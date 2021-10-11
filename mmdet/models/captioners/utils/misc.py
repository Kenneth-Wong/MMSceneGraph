# ---------------------------------------------------------------
# misc.py
# Set-up time: 2021/1/3 12:59
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com

# Adapted from XLAN
# ---------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn


def activation(act, elu_alpha=1.3):
    if act == 'ReLU':
        return nn.ReLU()
    elif act == 'Tanh':
        return nn.Tanh()
    elif act == 'GLU':
        return nn.GLU()
    elif act == 'ELU':
        return nn.ELU(elu_alpha)
    elif act == 'CeLU':
        return nn.CELU(elu_alpha)
    else:
        return nn.Identity()


def expand_tensor(tensor, size, dim=1):
    if size == 1 or tensor is None:
        return tensor
    tensor = tensor.unsqueeze(dim)
    tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim + 1:])).contiguous()
    tensor = tensor.view(list(tensor.shape[:dim - 1]) + [-1] + list(tensor.shape[dim + 1:]))
    return tensor


def narrow_tensor(tensor, size, dim=1, narrow_method='avg'):
    if tensor is None:
        return tensor
    assert tensor.size(dim-1) % size == 0
    tensor = tensor.view(list(tensor.shape[:dim - 1]) + [tensor.size(dim-1) / size, size] + list(tensor.shape[dim + 1:]))
    if narrow_method == 'avg':
        tensor = torch.mean(tensor, dim=dim)
    elif narrow_method == 'mean':
        tensor = torch.max(tensor, dim=dim)
    else:
        raise NotImplementedError
    return tensor




def expand_numpy(x, seq_per_img=5):
    if seq_per_img == 1:
        return x
    x = x.reshape((-1, 1))
    x = np.repeat(x, seq_per_img, axis=1)
    x = x.reshape((-1))
    return x


def load_ids(path):
    with open(path, 'r') as fid:
        lines = [int(line.strip()) for line in fid]
    return lines


def load_lines(path):
    with open(path, 'r') as fid:
        lines = [line.strip() for line in fid]
    return lines


def load_vocab(path):
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab


# torch.nn.utils.clip_grad_norm
# https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L84-L91
# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
def clip_gradient(optimizer, model, grad_clip_type, grad_clip):
    if grad_clip_type == 'Clamp':
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-grad_clip, grad_clip)
    elif grad_clip_type == 'Norm':
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        raise NotImplementedError


def decode_sequence(vocab, seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(vocab[ix])
        sent = ' '.join(words)
        sents.append(sent)
    return sents


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float(-1e9)).type_as(t)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
