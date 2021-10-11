# ---------------------------------------------------------------
# treelstm_util.py
# Set-up time: 2020/6/4 下午4:42
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .motif_util import get_dropout_mask, block_orthogonal
from .treelstm_util import TreeLSTM_IO
import math

class MultiLayer_HybridLSTM(nn.Module):
    """
    Multilayer Bidirectional hybrid LSTM
    Each layer contains one forward lstm(leaves to root) and one backward lstm(root to leaves)
    """

    def __init__(self, in_dim, out_dim, num_layer, dropout=0.0, pass_root=False, with_chain=True, chain_style='LSTM',
                 num_head=1, attn_style='cat'):
        super(MultiLayer_HybridLSTM, self).__init__()
        self.num_layer = num_layer
        layers = []
        for i in range(num_layer):
            input_dim = in_dim if i == 0 else out_dim
            layers.append(
                BidirectionalHybridLSTM(input_dim, out_dim, dropout, pass_root, with_chain, chain_style, num_head,
                                        attn_style))
        self.multi_layer_lstm = nn.ModuleList(layers)

    def forward(self, forest, features, num_obj):
        for i in range(self.num_layer):
            features = self.multi_layer_lstm[i](forest, features, num_obj)
        return features


class BidirectionalHybridLSTM(nn.Module):
    """
    Bidirectional Tree LSTM
    Contains one forward lstm(leaves to root) and one backward lstm(root to leaves)
    Dropout mask will be generated one time for all trees in the forest, to make sure the consistancy
    """

    def __init__(self, in_dim, out_dim, dropout=0.0, pass_root=False, with_chain=True, chain_style='LSTM', num_head=1,
                 attn_style='cat'):
        super(BidirectionalHybridLSTM, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim
        self.with_chain = with_chain
        self.chain_style = chain_style
        component_out_dim = out_dim // 4 if with_chain else out_dim // 2
        self.treeLSTM_forward = OneDirectionalTreeLSTM(in_dim, component_out_dim, 'forward', dropout, pass_root)
        self.treeLSTM_backward = OneDirectionalTreeLSTM(in_dim, component_out_dim, 'backward', dropout, pass_root)
        if self.with_chain:
            if chain_style == 'LSTM':
                self.chainLSTM_forward = OneDirectionalChainLSTM(in_dim, component_out_dim, 'forward', dropout)
                self.chainLSTM_backward = OneDirectionalChainLSTM(in_dim, component_out_dim, 'backward', dropout)
            elif chain_style == 'GNN':
                self.gnn = GNN(in_dim, component_out_dim * 2, num_head=num_head, attn_style=attn_style)

    def forward(self, forest, features, num_obj):
        tree_forward_output = self.treeLSTM_forward(forest, features, num_obj)
        tree_backward_output = self.treeLSTM_backward(forest, features, num_obj)

        if self.with_chain:
            if self.chain_style == 'LSTM':
                child_forward_output = self.chainLSTM_forward(forest, features, num_obj)
                child_backward_output = self.chainLSTM_backward(forest, features, num_obj)
                final_output = torch.cat((child_forward_output, child_backward_output,
                                          tree_forward_output, tree_backward_output), 1)
            elif self.chain_style == 'GNN':
                gnn_output = self.gnn(forest, features, num_obj)
                final_output = torch.cat((gnn_output, tree_forward_output, tree_backward_output), 1)
            else:
                raise not NotImplementedError
        else:
            final_output = torch.cat((tree_forward_output, tree_backward_output), 1)
        return final_output


class OneDirectionalTreeLSTM(nn.Module):
    """
    One Way Tree LSTM
    direction = forward | backward
    """

    def __init__(self, in_dim, out_dim, direction, dropout=0.0, pass_root=False):
        super(OneDirectionalTreeLSTM, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim
        if direction == 'forward':
            self.treeLSTM = TreeLSTM_Forward(in_dim, out_dim, pass_root)
        elif direction == 'backward':
            self.treeLSTM = TreeLSTM_Backward(in_dim, out_dim, pass_root)
        else:
            print('Error Tree LSTM Direction')

    def forward(self, forest, features, num_obj):
        # calc dropout mask, same for all
        if self.dropout > 0.0:
            dropout_mask = get_dropout_mask(self.dropout, self.out_dim, device=features.device)
        else:
            dropout_mask = None

        # tree lstm input
        out_h = None
        h_order = torch.LongTensor(num_obj).zero_().to(features.device)  # used to resume order
        order_idx = 0
        lstm_io = TreeLSTM_IO(out_h, h_order, order_idx, None, None, dropout_mask)
        # run tree lstm forward (leaves to root)
        for idx in range(len(forest)):
            self.treeLSTM(forest[idx], features, lstm_io)

        # resume order to the same as input
        output = torch.index_select(lstm_io.hidden, 0, lstm_io.order.long())
        return output


class TreeLSTM_Forward(nn.Module):
    """
    Child-sum LSTM
    From leaves to root
    """

    def __init__(self, feat_dim, h_dim, pass_root, is_pass_embed=False, embed_layer=None, embed_out_layer=None):
        super(TreeLSTM_Forward, self).__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.pass_root = pass_root
        self.is_pass_embed = is_pass_embed
        self.embed_layer = embed_layer
        self.embed_out_layer = embed_out_layer
        self.pooling_size = 7

        if self.is_pass_embed:
            assert self.embed_layer is not None
            self.p_embed_dim = self.embed_layer.weight.data.shape[1]
            self.p_embed = nn.Linear(self.p_embed_dim, self.p_embed_dim)
            block_orthogonal(self.p_embed.weight.data, [self.p_embed_dim, self.p_embed_dim])
            self.p_embed.bias.data.fill_(0.0)
            feat_forward_dim = self.feat_dim + self.p_embed_dim
        else:
            feat_forward_dim = self.feat_dim

        self.ioffux = nn.Linear(feat_forward_dim, 4 * self.h_dim)
        self.ioffuh = nn.Linear(self.h_dim, 4 * self.h_dim)
        self.px = nn.Linear(feat_forward_dim, self.h_dim)
        self.forget_x = nn.Linear(feat_forward_dim, self.h_dim)
        self.forget_h = nn.Linear(self.h_dim, self.h_dim)

        # init parameter
        block_orthogonal(self.px.weight.data, [self.h_dim, feat_forward_dim])
        block_orthogonal(self.ioffux.weight.data, [self.h_dim, feat_forward_dim])
        block_orthogonal(self.ioffuh.weight.data, [self.h_dim, self.h_dim])
        block_orthogonal(self.forget_x.weight.data, [self.h_dim, feat_forward_dim])
        block_orthogonal(self.forget_h.weight.data, [self.h_dim, self.h_dim])

        self.px.bias.data.fill_(0.0)
        self.ioffux.bias.data.fill_(0.0)
        self.ioffuh.bias.data.fill_(0.0)
        self.forget_x.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.forget_h.bias.data.fill_(1.0)

        # if self.saliency:
        #    sal_fmap = [Flattener(), nn.Linear(self.pooling_size * self.pooling_size + 3, 1)]
        #    self.sal_fmap = nn.Linear(self.pooling_size * self.pooling_size + 3, 1)

    def node_forward(self, feat_inp, collection_c, collection_h, dropout_mask):
        projected_x = self.px(feat_inp)
        h_sum = torch.sum(torch.cat(collection_h, 0), 0, keepdim=True)
        ioffu = self.ioffux(feat_inp) + self.ioffuh(h_sum)
        i, o, u, r = torch.split(ioffu, ioffu.size(1) // 4, dim=1)
        i, o, u, r = F.sigmoid(i), F.sigmoid(o), F.tanh(u), F.sigmoid(r)

        collection_f = [F.sigmoid(self.forget_x(feat_inp) + self.forget_h(col_h)) for col_h in collection_h]

        c = torch.mul(i, u) + torch.sum(torch.cat([torch.mul(f, c) for f, c in zip(collection_f, collection_c)], 0), 0,
                                        keepdim=True)
        h = torch.mul(o, F.tanh(c))
        h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            h_final = torch.mul(h_final, dropout_mask)
        return c, h_final

    def forward(self, tree, features, treelstm_io):
        """
        tree: The root for a tree
        features: [num_obj, featuresize]
        treelstm_io.hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
        treelstm_io.order: init as 0 for all [num_obj], update for recovering original order
        """
        # recursively search child
        num_child = len(tree.children)
        if num_child > 0:
            for i in range(num_child):
                self.forward(tree.children[i], features, treelstm_io)

        # get c,h from all children
        collection_c = []
        collection_h = []
        collection_embed = []
        if num_child:
            for i in range(num_child):
                collection_c.append(tree.children[i].tree_state_c.view(1, -1))
                collection_h.append(tree.children[i].tree_state_h.view(1, -1))
                if self.is_pass_embed:
                    collection_embed.append(tree.children[i].embeded_label.view(1, -1))
        else:
            collection_c.append(torch.FloatTensor(self.h_dim).to(features.device).fill_(0.0).view(1, -1))
            collection_h.append(torch.FloatTensor(self.h_dim).to(features.device).fill_(0.0).view(1, -1))
            if self.is_pass_embed:
                collection_embed.append(self.embed_layer.weight[0].view(1, -1))

        # calc
        if self.is_pass_embed:  # Only being used in decoder network
            embed = self.p_embed(torch.sum(torch.cat(collection_embed, 0), 0, keepdim=True))
            next_feature = torch.cat((features[tree.index].view(1, -1), embed.view(1, -1)), 1)
        else:
            next_feature = features[tree.index].view(1, -1)

        c, h = self.node_forward(next_feature, collection_c, collection_h, treelstm_io.dropout_mask)
        tree.tree_state_c = c
        tree.tree_state_h = h
        # record label prediction
        # Only being used in decoder network
        if self.is_pass_embed:
            pass_embed_postprocess(h, self.embed_out_layer, self.embed_layer, tree, treelstm_io, self.training)

        # record hidden state
        if treelstm_io.hidden is None:
            treelstm_io.hidden = h.view(1, -1)
        else:
            treelstm_io.hidden = torch.cat((treelstm_io.hidden, h.view(1, -1)), 0)

        treelstm_io.order[tree.index] = treelstm_io.order_count
        treelstm_io.order_count += 1
        return


class TreeLSTM_Backward(nn.Module):
    """
    from root to leaves
    """

    def __init__(self, feat_dim, h_dim, pass_root, is_pass_embed=False, embed_layer=None, embed_out_layer=None):
        super(TreeLSTM_Backward, self).__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.pass_root = pass_root
        self.is_pass_embed = is_pass_embed
        self.embed_layer = embed_layer
        self.embed_out_layer = embed_out_layer

        if self.is_pass_embed:
            assert self.embed_layer is not None
            self.p_embed_dim = self.embed_layer.weight.data.shape[1]
            feat_forward_dim = self.feat_dim + self.p_embed_dim
        else:
            feat_forward_dim = self.feat_dim

        self.iofux = nn.Linear(feat_forward_dim, 5 * self.h_dim)
        self.iofuh = nn.Linear(self.h_dim, 5 * self.h_dim)
        self.px = nn.Linear(feat_forward_dim, self.h_dim)

        # init parameter
        block_orthogonal(self.px.weight.data, [self.h_dim, feat_forward_dim])
        block_orthogonal(self.iofux.weight.data, [self.h_dim, feat_forward_dim])
        block_orthogonal(self.iofuh.weight.data, [self.h_dim, self.h_dim])

        self.px.bias.data.fill_(0.0)
        self.iofux.bias.data.fill_(0.0)
        self.iofuh.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.iofuh.bias.data[2 * self.h_dim:3 * self.h_dim].fill_(1.0)

    def node_backward(self, feat_inp, root_c, root_h, dropout_mask):

        projected_x = self.px(feat_inp)
        iofu = self.iofux(feat_inp) + self.iofuh(root_h)
        i, o, f, u, r = torch.split(iofu, iofu.size(1) // 5, dim=1)
        i, o, f, u, r = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f), F.tanh(u), F.sigmoid(r)

        c = torch.mul(i, u) + torch.mul(f, root_c)
        h = torch.mul(o, F.tanh(c))
        h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            h_final = torch.mul(h_final, dropout_mask)
        return c, h_final

    def forward(self, tree, features, treelstm_io):
        """
        tree: The root for a tree
        features: [num_obj, featuresize]
        treelstm_io.hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
        treelstm_io.order: init as 0 for all [num_obj], update for recovering original order
        """

        if tree.parent is None or (tree.parent.is_root and (not self.pass_root)):
            root_c = torch.FloatTensor(self.h_dim).to(features.device).fill_(0.0)
            root_h = torch.FloatTensor(self.h_dim).to(features.device).fill_(0.0)
            if self.is_pass_embed:
                root_embed = self.embed_layer.weight[0]
        else:
            root_c = tree.parent.tree_state_c_backward
            root_h = tree.parent.tree_state_h_backward
            if self.is_pass_embed:
                root_embed = tree.parent.embeded_label

        if self.is_pass_embed:
            next_features = torch.cat((features[tree.index].view(1, -1), root_embed.view(1, -1)), 1)
        else:
            next_features = features[tree.index].view(1, -1)

        c, h = self.node_backward(next_features, root_c, root_h, treelstm_io.dropout_mask)
        tree.tree_state_c_backward = c
        tree.tree_state_h_backward = h
        # record label prediction
        # Only being used in decoder network
        if self.is_pass_embed:
            pass_embed_postprocess(h, self.embed_out_layer, self.embed_layer, tree, treelstm_io, self.training)

        # record hidden state
        if treelstm_io.hidden is None:
            treelstm_io.hidden = h.view(1, -1)
        else:
            treelstm_io.hidden = torch.cat((treelstm_io.hidden, h.view(1, -1)), 0)

        treelstm_io.order[tree.index] = treelstm_io.order_count
        treelstm_io.order_count += 1

        # recursively update from root to leaves
        num_child = len(tree.children)
        if num_child:
            for i in range(num_child):
                self.forward(tree.children[i], features, treelstm_io)
        return


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_head=1, attn_style='cat'):
        super(GNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gnn_propagate = GNNPropagation(in_dim, out_dim, num_head, attn_style)

    def forward(self, forest, features, num_obj):
        out_h = None
        h_order = torch.LongTensor(num_obj).zero_().to(features.device)  # used to resume order
        order_idx = 0
        lstm_io = TreeLSTM_IO(out_h, h_order, order_idx, None, None, None)
        # run the gnn forward (root to leave, in each parent-son subgraph)
        for idx in range(len(forest)):
            self.gnn_propagate(forest[idx], features, lstm_io)

        # resume order to the same as input
        output = torch.index_select(lstm_io.hidden, 0, lstm_io.order.long())
        return output


class GNNPropagation(nn.Module):
    def __init__(self, feat_dim, h_dim, num_head=1, attn_style='cat'):
        super(GNNPropagation, self).__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.graph_scale = 2
        assert h_dim % num_head == 0
        self.num_head = num_head
        self.attn_style=attn_style
        if self.attn_style == 'cat':
            self.rel_weight = nn.Linear(feat_dim, h_dim, bias=False)
            self.attn_weight = nn.ModuleList([nn.Linear(2 * (h_dim // num_head), 1)] * num_head)
        else:
            self.rel_weight = nn.ModuleList([nn.Linear(feat_dim, h_dim)] * 3)

    def forward(self, tree, features, treelstm_io):
        """
        tree: The root for a tree
        features: [num_obj, featuresize]
        treelstm_io.hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
        treelstm_io.order: init as 0 for all [num_obj], update for recovering original order
        """
        # recursively search child
        num_child = len(tree.children)
        if num_child and num_child >= self.graph_scale - 1:
            # propagate inside a subgraph:
            children = tree.children
            node_index = [tree.index] + [children[i].index for i in range(num_child)]
            graph_feats = torch.cat([features[tree.index][None]] +
                                    [features[children[i].index][None] for i in range(num_child)], 0)
            num_nodes = graph_feats.size(0)

            if self.attn_style == 'cat':
                basic_transformed_feats = self.rel_weight(graph_feats)  # N * (h_dim)
                transformed_feats = basic_transformed_feats.view(num_nodes, self.num_head, -1).transpose(1,
                                                                                                         0)  # Head * N * (h/head)
                attn_transformed_feats = torch.cat((transformed_feats.unsqueeze(2).expand(-1, -1, num_nodes, -1),
                                                   transformed_feats.unsqueeze(1).expand(-1, num_nodes, -1, -1)),
                                                  dim=-1)  # Head * N * N * (h/head*2)
                hidden_feats = torch.stack(
                    [F.softmax(F.leaky_relu(aw(c).squeeze(), inplace=True)).mm(tf) for aw, c, tf in
                     zip(self.attn_weight,
                         attn_transformed_feats,
                         transformed_feats)],
                ).transpose(1, 0).contiguous().view(num_nodes, -1)
                hidden_feats = basic_transformed_feats + hidden_feats
            elif self.attn_style == 'dot':
                query, key, value = self.rel_weight[0](graph_feats).view(num_nodes, self.num_head, -1).transpose(1, 0), \
                                    self.rel_weight[1](graph_feats).view(num_nodes, self.num_head, -1).transpose(1, 0),\
                                    self.rel_weight[2](graph_feats).view(num_nodes, self.num_head, -1).transpose(1, 0)
                d_k = query.size(-1)
                scores = nn.Dropout(0.1)(F.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)))
                hidden_feats = torch.matmul(scores, value).transpose(1, 0).contiguous().view(num_nodes, -1)
            else:
                raise NotImplementedError
            # record hidden state: do not update the root, it has been updated when it was a child!
            if tree.parent is not None:
                hidden_feats = hidden_feats[1:]

            if treelstm_io.hidden is None:
                treelstm_io.hidden = hidden_feats
            else:
                treelstm_io.hidden = torch.cat((treelstm_io.hidden, hidden_feats), 0)
            node_update_pool = node_index if tree.parent is None else node_index[1:]
            for i in node_update_pool:
                treelstm_io.order[i] = treelstm_io.order_count
                treelstm_io.order_count += 1
        else:
            if tree.parent is None:
                # it means that this tree is empty!
                treelstm_io.hidden = features[tree.index][None]
                treelstm_io.order[tree.index] = treelstm_io.order_count
                treelstm_io.order_count += 1
            else:
                # otherwise it is a son of its parent, and do not has son itself, do nothing
                pass

        if num_child:
            for i in range(num_child):
                self.forward(tree.children[i], features, treelstm_io)
        return


class OneDirectionalChainLSTM(nn.Module):
    """
        One Way chain LSTM
        direction = forward | backward
        """

    def __init__(self, in_dim, out_dim, direction, dropout=0.0):
        super(OneDirectionalChainLSTM, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim
        if direction == 'forward':
            self.chainLSTM = ChainLSTM('forward', in_dim, out_dim)
        elif direction == 'backward':
            self.chainLSTM = ChainLSTM('backward', in_dim, out_dim)
        else:
            print('Error LSTM Direction')

    def forward(self, forest, features, num_obj):
        # calc dropout mask, same for all
        if self.dropout > 0.0:
            dropout_mask = get_dropout_mask(self.dropout, self.out_dim, device=features.device)
        else:
            dropout_mask = None

        # tree lstm input
        out_h = None
        h_order = torch.LongTensor(num_obj).zero_().to(features.device)  # used to resume order
        order_idx = 0
        lstm_io = TreeLSTM_IO(out_h, h_order, order_idx, None, None, dropout_mask)
        # run tree lstm forward (leaves to root)
        for idx in range(len(forest)):
            self.chainLSTM(forest[idx], features, lstm_io)

        # resume order to the same as input
        output = torch.index_select(lstm_io.hidden, 0, lstm_io.order.long())
        return output


class ChainLSTM(nn.Module):
    def __init__(self, direction, feat_dim, h_dim, is_pass_embed=False, embed_layer=None, embed_out_layer=None,
                 use_highway=True):
        super(ChainLSTM, self).__init__()
        assert direction in ('forward', 'backward')
        self.direction = direction
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.is_pass_embed = is_pass_embed
        self.embed_layer = embed_layer
        self.embed_out_layer = embed_out_layer
        self.use_highway = use_highway

        self.num_split = 5 if self.use_highway else 4
        self.ioffux = nn.Linear(self.feat_dim, self.num_split * self.h_dim)
        self.ioffuh = nn.Linear(self.h_dim, self.num_split * self.h_dim)
        self.px = nn.Linear(self.feat_dim, self.h_dim) if self.use_highway else None

        # init parameter
        if self.use_highway:
            block_orthogonal(self.px.weight.data, [self.h_dim, self.feat_dim])
            self.px.bias.data.fill_(0.0)

        block_orthogonal(self.ioffux.weight.data, [self.h_dim, self.feat_dim])
        block_orthogonal(self.ioffuh.weight.data, [self.h_dim, self.h_dim])

        self.ioffux.bias.data.fill_(0.0)
        self.ioffuh.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.ioffuh.bias.data[2 * self.h_dim:3 * self.h_dim].fill_(1.0)

    def node_forward(self, feat_inp, previous_state_c, previous_state_h, dropout_mask):
        ioffu = self.ioffux(feat_inp) + self.ioffuh(previous_state_h)
        packed = torch.split(ioffu, ioffu.size(1) // self.num_split, dim=1)
        if self.use_highway:
            i, o, f, u, r = packed
            i, o, f, u, r = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f), F.tanh(u), F.sigmoid(r)
        else:
            i, o, f, u = packed
            i, o, f, u = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f), F.tanh(u)

        c = torch.mul(i, u) + torch.mul(f, previous_state_c)
        h = torch.mul(o, F.tanh(c))
        if self.use_highway:
            projected_x = self.px(feat_inp)
            h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        else:
            h_final = h
        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            h_final = torch.mul(h_final, dropout_mask)
        return c, h_final

    def forward(self, tree, features, treelstm_io):
        """
        tree: The root for a tree
        features: [num_obj, featuresize]
        treelstm_io.hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
        treelstm_io.order: init as 0 for all [num_obj], update for recovering original order
        """
        # recursively search child
        num_child = len(tree.children)
        if num_child:
            for i in range(num_child):
                self.forward(tree.children[i], features, treelstm_io)

        if num_child >= 1:
            children = tree.children
            iterator = range(num_child) if self.direction == 'forward' else range(num_child - 1, -1, -1)
            step = -1 if self.direction == 'forward' else 1
            for i in iterator:
                if (i == 0 and self.direction == 'forward') or (i == num_child - 1 and self.direction == 'backward'):
                    previous_state_h = torch.FloatTensor(self.h_dim).to(features.device).fill_(0.0)
                    previous_state_c = torch.FloatTensor(self.h_dim).to(features.device).fill_(0.0)
                    if self.is_pass_embed:
                        previous_embed = self.embed_layer.weight[0]
                else:
                    if self.direction == 'forward':
                        previous_state_h = children[i + step].chain_state_h
                        previous_state_c = children[i + step].chain_state_c
                    else:
                        previous_state_h = children[i + step].chain_state_h_backward
                        previous_state_c = children[i + step].chain_state_c_backward
                    if self.is_pass_embed:
                        previous_embed = children[i + step].embedd_label
                if self.is_pass_embed:
                    next_feature = torch.cat((features[children[i].index].view(1, -1), previous_embed.view(1, -1)), 1)
                else:
                    next_feature = features[children[i].index].view(1, -1)
                c, h = self.node_forward(next_feature, previous_state_c, previous_state_h, treelstm_io.dropout_mask)
                if self.direction == 'forward':
                    children[i].chain_state_c = c
                    children[i].chain_state_h = h
                else:
                    children[i].chain_state_c_backward = c
                    children[i].chain_state_h_backward = h
                if self.is_pass_embed:
                    pass_embed_postprocess(h, self.embed_out_layer, self.embed_layer, tree, treelstm_io,
                                           self.training)

                # record hidden state
                if treelstm_io.hidden is None:
                    treelstm_io.hidden = h.view(1, -1)
                else:
                    treelstm_io.hidden = torch.cat((treelstm_io.hidden, h.view(1, -1)), 0)

                treelstm_io.order[children[i].index] = treelstm_io.order_count
                treelstm_io.order_count += 1
        return


def pass_embed_postprocess(h, embed_out_layer, embed_layer, tree, treelstm_io, is_training):
    """
    Calculate districution and predict/sample labels
    Add to lstm_IO
    """
    pred_dist = embed_out_layer(h)
    label_to_embed = F.softmax(pred_dist.view(-1), 0)[1:].max(0)[1] + 1
    if is_training:
        sampled_label = F.softmax(pred_dist.view(-1), 0)[1:].multinomial(1).detach() + 1
        tree.embeded_label = embed_layer(sampled_label + 1)
    else:
        tree.embeded_label = embed_layer(label_to_embed + 1)

    if treelstm_io.dists is None:
        treelstm_io.dists = pred_dist.view(1, -1)
    else:
        treelstm_io.dists = torch.cat((treelstm_io.dists, pred_dist.view(1, -1)), 0)

    if treelstm_io.commitments is None:
        treelstm_io.commitments = label_to_embed.view(-1)
    else:
        treelstm_io.commitments = torch.cat((treelstm_io.commitments, label_to_embed.view(-1)), 0)
