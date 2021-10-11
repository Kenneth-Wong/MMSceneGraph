import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, get_point_from_mask
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
from mmdet.models.relation_heads.approaches import Result


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 relation_head=None,
                 attr_head=None,
                 relcaption_head=None,
                 saliency_detector=None,
                 downstream_caption_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 with_point=False):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        if relation_head is not None:
            self.relation_head = builder.build_head(relation_head)

        if attr_head is not None:
            self.attr_head = builder.build_head(attr_head)

        # For relational caption / caption
        if relcaption_head is not None:
            self.relcaption_head = builder.build_head(relcaption_head)

        # For saliency detection
        if saliency_detector is not None:
            self.saliency_detector = builder.build_saliency_detector(saliency_detector)

        # For downstream SG caption
        if downstream_caption_head is not None:
            self.downstream_caption_head = builder.build_head(downstream_caption_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.with_point = with_point

        # cache the detection results to speed up the sgdet training.
        self.rpn_results = dict()
        self.det_results = dict()

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        if self.with_relation:
            self.relation_head.init_weights()
        if self.with_attr:
            self.attr_head.init_weights()
        if self.with_relcaption:
            self.relcaption_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(device=img.device)
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            if self.bbox_head.__class__.__name__ == 'ExtrDetWeightSharedFCBBoxHead':
                det_weight = self.bbox_head.det_weight_hook()
            else:
                det_weight = None
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            if self.mask_head.__class__.__name__ == 'TransferMaskHead':
                assert det_weight is not None
                mask_input = (mask_feats, det_weight)
            else:
                mask_input = (mask_feats,)
            mask_pred = self.mask_head(*mask_input)
            outs = outs + (mask_pred,)
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_rels=None,
                      gt_keyrels=None,
                      gt_relmaps=None,
                      gt_scenes=None,
                      gt_attrs=None,
                      gt_rel_inputs=None,  # for rel caption
                      gt_rel_targets=None,  # for rel caption
                      gt_rel_ipt_scores=None,  # for rel caption
                      gt_cap_inputs=None,  # for caption
                      gt_cap_targets=None,  # for caption
                      proposals=None,
                      rescale=False):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
            gt_rels :
            gt_relmaps:
            rescale:


        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        ################################################################
        #        Specifically for downstream Captioning Caption        #
        #        Use the result from relation head or relcaption_head. #
        ################################################################
        if self.with_downstream_caption:
            assert self.with_relation or self.with_relcaption
            if self.with_relation:
                pass
            elif self.with_relcaption:
                bboxes, labels, target_labels, \
                dists, _, _ = self.detector_simple_test(x, img_meta, gt_bboxes, gt_labels,
                                                        gt_masks,
                                                        proposals,
                                                        use_gt_box=self.relcaption_head.use_gt_box,
                                                        use_gt_label=self.relcaption_head.use_gt_label,
                                                        rescale=rescale)

                det_result = Result(bboxes=bboxes, labels=labels, dists=dists,
                                    target_labels=target_labels, target_scenes=gt_scenes,
                                    img_shape=[meta['img_shape'] for meta in img_meta])

                det_result = self.relcaption_head(x, img_meta, det_result, is_testing=True, downstreaming=True)
                roi_feats = self.relcaption_head.bbox_roi_extractor.roi_feats

                gt_result = Result(bboxes=gt_bboxes, labels=gt_labels, rels=gt_rels, relmaps=gt_relmaps,
                                   cap_inputs=[cap_input.clone() for cap_input in
                                               gt_cap_inputs] if gt_cap_inputs is not None else None,
                                   cap_targets=[cap_target.clone() for cap_target in
                                                gt_cap_targets] if gt_cap_targets is not None else None,
                                   img_shape=[meta['img_shape'] for meta in img_meta])
                # generate the new cap_scores to cover the cap_scores from relational_caption_head
                det_result = self.downstream_caption_head(x, img_meta, roi_feats, det_result, gt_result)
                return self.downstream_caption_head.loss(det_result)

        ################################################################
        #        Specifically for Relation Prediction / SGG.           #
        #        The detector part must perform as if at test mode.    #
        ################################################################
        if self.with_relation:
            # assert gt_rels is not None and gt_relmaps is not None
            if self.relation_head.with_visual_mask and (not self.with_mask):
                raise ValueError('The basic detector did not provide masks.')

            """
            NOTE: (for VG) When the gt masks is None, but the head needs mask, 
            we use the gt_box and gt_label (if needed) to generate the fake mask. 
            """
            bboxes, labels, target_labels, \
            dists, masks, points = self.detector_simple_test(x, img_meta, gt_bboxes, gt_labels,
                                                             gt_masks,
                                                             proposals,
                                                             use_gt_box=self.relation_head.use_gt_box,
                                                             use_gt_label=self.relation_head.use_gt_label,
                                                             rescale=rescale)

            saliency_maps = self.saliency_detector_test(img, img_meta) if self.with_saliency else None

            gt_result = Result(bboxes=gt_bboxes, labels=gt_labels, rels=gt_rels, relmaps=gt_relmaps, masks=gt_masks,
                               rel_pair_idxes=[rel[:, :2].clone() for rel in gt_rels] if gt_rels is not None else None,
                               rel_labels=[rel[:, -1].clone() for rel in gt_rels] if gt_rels is not None else None,
                               key_rels=gt_keyrels if gt_keyrels is not None else None,
                               img_shape=[meta['img_shape'] for meta in img_meta], scenes=gt_scenes)

            det_result = Result(bboxes=bboxes, labels=labels, dists=dists, masks=masks, points=points,
                                target_labels=target_labels, target_scenes=gt_scenes,
                                saliency_maps = saliency_maps,
                                img_shape=[meta['img_shape'] for meta in img_meta])

            det_result = self.relation_head(x, img_meta, det_result, gt_result)
            return self.relation_head.loss(det_result)

        ################################################################
        #        Specifically for Caption/relational Caption           #
        #        The detector part must perform as if at test mode.    #
        ################################################################
        if self.with_relcaption:
            bboxes, labels, target_labels, \
            dists, _, _ = self.detector_simple_test(x, img_meta, gt_bboxes, gt_labels,
                                                    gt_masks,
                                                    proposals,
                                                    use_gt_box=self.relcaption_head.use_gt_box,
                                                    use_gt_label=self.relcaption_head.use_gt_label,
                                                    rescale=rescale)

            gt_result = Result(bboxes=gt_bboxes, labels=gt_labels, rels=gt_rels, relmaps=gt_relmaps,
                               rel_pair_idxes=[rel[:, :2].clone() for rel in gt_rels] if gt_rels is not None else None,
                               rel_cap_inputs=[rel_input.clone() for rel_input in
                                               gt_rel_inputs] if gt_rel_inputs is not None else None,
                               rel_cap_targets=[rel_target.clone() for rel_target in
                                                gt_rel_targets] if gt_rel_targets is not None else None,
                               rel_ipts=[rel_ipt.clone() for rel_ipt
                                         in gt_rel_ipt_scores] if gt_rel_ipt_scores is not None else None,
                               cap_inputs=[cap_input.clone() for cap_input in
                                           gt_cap_inputs] if gt_cap_inputs is not None else None,
                               cap_targets=[cap_target.clone() for cap_target in
                                            gt_cap_targets] if gt_cap_targets is not None else None,
                               img_shape=[meta['img_shape'] for meta in img_meta], scenes=gt_scenes)

            det_result = Result(bboxes=bboxes, labels=labels, dists=dists,
                                target_labels=target_labels, target_scenes=gt_scenes,
                                img_shape=[meta['img_shape'] for meta in img_meta])

            det_result = self.relcaption_head(x, img_meta, det_result, gt_result)
            return self.relcaption_head.loss(det_result)



        ################################################################
        #        Original object detector running procedure.           #
        ################################################################

        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            if self.bbox_head.__class__.__name__ == 'ExtrDetWeightSharedFCBBoxHead':
                det_weight = self.bbox_head.det_weight_hook()
            else:
                det_weight = None

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

            if self.with_attr:
                attr_score, _ = self.attr_head(bbox_feats)
                # transform the gt_attrs
                attr_targets = self.attr_head.get_attr_target(sampling_results, gt_attrs)
                loss_attr = self.attr_head.loss(attr_score, None, bbox_targets[0], None, None, None,
                                                attr_targets=attr_targets)
                losses.update(loss_attr)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            if mask_feats.shape[0] > 0:
                if self.mask_head.__class__.__name__ == 'TransferMaskHead':
                    assert det_weight is not None
                    mask_input = (mask_feats, det_weight)
                else:
                    mask_input = (mask_feats,)
                mask_pred = self.mask_head(*mask_input)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta,
                                                      self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        if self.bbox_head.__class__.__name__ == 'ExtrDetWeightSharedFCBBoxHead':
            det_weight = self.bbox_head.det_weight_hook()
        else:
            det_weight = None

        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_meta,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'),
                det_weight=det_weight)
            return bbox_results, segm_results

    def saliency_detector_test(self, img, img_meta):
        return self.saliency_detector(img, img_meta, return_loss=False)

    # TODO
    def detector_simple_test(self,
                             x,
                             img_meta,
                             gt_bboxes,
                             gt_labels,
                             gt_masks,
                             proposals=None,
                             use_gt_box=False,
                             use_gt_label=False,
                             rescale=False,
                             is_testing=False):
        """Test without augmentation. Used in SGG.

        Return:
            det_bboxes: (list[Tensor]): The boxes may have 5 columns (sgdet) or 4 columns (predcls/sgcls).
            det_labels: (list[Tensor]): 1D tensor, det_labels (sgdet) or gt_labels (predcls/sgcls).
            det_dists: (list[Tensor]): 2D tensor, N x Nc, the bg column is 0. detected dists (sgdet/sgcls), or
                None (predcls).
            masks: (list[list[Tensor]]): Mask is associated with box. Thus, in predcls/sgcls mode, it will
                firstly return the gt_masks. But some datasets do not contain gt_masks. We try to use the gt box
                to obtain the masks.

        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.with_mask:
            if use_gt_box and use_gt_label:  # predcls
                target_labels = gt_labels
                if gt_masks is None:
                    result = self.detector_simple_test_gt_mask(x, img_meta, gt_bboxes, gt_labels, rescale=rescale)
                    if self.with_point:
                        masks, points = result
                    else:
                        masks, points = result, None
                else:
                    # temporarily not used, no gt masks.
                    masks = gt_masks
                    points = None
                    if self.with_point:
                        points = get_point_from_mask(masks, gt_bboxes,
                                                     self.test_cfg.rcnn.get('mask_size', 56),
                                                     self.test_cfg.rcnn.get('sample_num', 729),
                                                     self.test_cfg.rcnn.get('dist_sample_thr', 3))
                return gt_bboxes, gt_labels, target_labels, None, masks, points

            elif use_gt_box and not use_gt_label:  # sgcls
                """The self implementation return 1-based det_labels"""
                target_labels = gt_labels
                _, det_labels, det_dists = self.detector_simple_test_det_bbox(x, img_meta,
                                                                              proposals=gt_bboxes, rescale=rescale)
                if gt_masks is None:
                    result = self.detector_simple_test_gt_mask(x, img_meta, gt_bboxes, gt_labels,
                                                               det_labels=det_labels, use_gt_label=use_gt_label,
                                                               rescale=rescale)
                    if self.with_point:
                        masks, points = result
                    else:
                        masks, points = result, None
                else:
                    # temporarily not used, no gt masks.
                    masks = gt_masks
                    points = None
                    if self.with_point:
                        points = get_point_from_mask(masks, gt_bboxes,
                                                     self.test_cfg.rcnn.get('mask_size', 56),
                                                     self.test_cfg.rcnn.get('sample_num', 729),
                                                     self.test_cfg.rcnn.get('dist_sample_thr', 3))
                return gt_bboxes, det_labels, target_labels, det_dists, masks, points

            elif not use_gt_box and not use_gt_label:
                """It returns 1-based det_labels"""
                det_bboxes, det_labels, det_dists, masks, points = self.detector_simple_test_det_bbox_mask(x, img_meta,
                                                                                                           rescale=rescale)
                # get target labels for the det bboxes: make use of the bbox head assigner
                if not is_testing:  # excluding the testing phase
                    target_labels = []
                    bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                    for i in range(len(img_meta)):
                        assign_result = bbox_assigner.assign(det_bboxes[i],
                                                             gt_bboxes[i],
                                                             gt_labels=gt_labels[i])
                        target_labels.append(assign_result.labels)
                else:
                    target_labels = None

                return det_bboxes, det_labels, target_labels, det_dists, masks, points

        # General type: do not need mask and points
        else:
            if use_gt_box and use_gt_label:  # predcls
                target_labels = gt_labels
                return gt_bboxes, gt_labels, target_labels, None, None, None

            elif use_gt_box and not use_gt_label:  # sgcls
                """The self implementation return 1-based det_labels"""
                target_labels = gt_labels
                _, det_labels, det_dists = self.detector_simple_test_det_bbox(x, img_meta,
                                                                              proposals=gt_bboxes, rescale=rescale)
                return gt_bboxes, det_labels, target_labels, det_dists, None, None

            elif not use_gt_box and not use_gt_label:
                """It returns 1-based det_labels"""
                det_bboxes, det_labels, det_dists, _, _ = self.detector_simple_test_det_bbox_mask(x, img_meta,
                                                                                                  rescale=rescale)
                # get target labels for the det bboxes: make use of the bbox head assigner
                if not is_testing:  # excluding the testing phase
                    target_labels = []
                    bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                    for i in range(len(img_meta)):
                        assign_result = bbox_assigner.assign(det_bboxes[i],
                                                             gt_bboxes[i],
                                                             gt_labels=gt_labels[i])
                        target_labels.append(assign_result.labels)
                else:
                    target_labels = None

                return det_bboxes, det_labels, target_labels, det_dists, None, None

    def detector_simple_test_gt_mask(self,
                                     x,
                                     img_meta,
                                     gt_bboxes,
                                     gt_labels,
                                     det_labels=None,
                                     use_gt_label=True,
                                     rescale=False):
        """
        Specifically for VG which does not contain gt mask and use Mask-X-RCNN to get masks with gt_bboxes,
        but may be with det_labels or gt_labels.
        Args:
            det_labels / gt_labels: (list[Tensor]): must be 0 based.
        """
        assert self.with_mask
        num_levels = len(x)
        if self.bbox_head.__class__.__name__ == 'ExtrDetWeightSharedFCBBoxHead':
            det_weight = self.bbox_head.det_weight_hook()
        else:
            det_weight = None

        segm_masks = []
        points = []
        for img_id in range(len(img_meta)):
            x_i = tuple([x[i][img_id][None] for i in range(num_levels)])
            img_meta_i = [img_meta[img_id]]

            if use_gt_label:
                label_input = gt_labels[img_id] - 1
            else:
                assert det_labels is not None
                label_input = det_labels[img_id] - 1
            test_result = self.simple_test_mask(
                x_i, img_meta_i, gt_bboxes[img_id], label_input, rescale=rescale, det_weight=det_weight,
                with_point=self.with_point)
            if isinstance(test_result, tuple):
                segm_masks_i, points_i = test_result
                points.append(points_i)
            else:
                segm_masks_i = test_result
            segm_masks.append(segm_masks_i)
        if self.with_point:
            return segm_masks, points
        else:
            return segm_masks

    def detector_simple_test_det_bbox(self,
                                      x,
                                      img_meta,
                                      proposals=None,
                                      rescale=False):
        """Run the detector in test mode, given gt_bboxes, return the labels, dists
        Return:
            det_labels: 1 based.
        """
        num_levels = len(x)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals
        """Support multi-image per batch"""
        det_bboxes, det_labels, score_dists = [], [], []
        for img_id in range(len(img_meta)):
            x_i = tuple([x[i][img_id][None] for i in range(num_levels)])
            img_meta_i = [img_meta[img_id]]
            proposal_list_i = [proposal_list[img_id]]
            det_labels_i, score_dists_i = self.simple_test_given_bboxes(x_i, proposal_list_i)
            det_bboxes.append(proposal_list[img_id])
            det_labels.append(det_labels_i)
            score_dists.append(score_dists_i)
        return det_bboxes, det_labels, score_dists

    def detector_simple_test_det_bbox_mask(self,
                                           x,
                                           img_meta,
                                           rescale=False):
        """Run the detector in test mode, return the detected boxes, labels, dists, and masks"""

        """RPN phase"""
        num_levels = len(x)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)

        """Support multi-image per batch"""
        det_bboxes, det_labels, score_dists = [], [], []
        for img_id in range(len(img_meta)):
            x_i = tuple([x[i][img_id][None] for i in range(num_levels)])
            img_meta_i = [img_meta[img_id]]
            proposal_list_i = [proposal_list[img_id]]
            det_bboxes_i, det_labels_i, score_dists_i = self.simple_test_bboxes(
                x_i, img_meta_i, proposal_list_i, self.test_cfg.rcnn, rescale=rescale, return_dist=True)
            det_bboxes.append(det_bboxes_i)
            det_labels.append(det_labels_i + 1)
            score_dists.append(score_dists_i)

        if not self.with_mask:
            return det_bboxes, det_labels, score_dists, None, None
        else:
            if self.bbox_head.__class__.__name__ == 'ExtrDetWeightSharedFCBBoxHead':
                det_weight = self.bbox_head.det_weight_hook()
            else:
                det_weight = None
            segm_masks = []
            points = []
            for img_id in range(len(img_meta)):
                x_i = tuple([x[i][img_id][None] for i in range(num_levels)])
                img_meta_i = [img_meta[img_id]]
                test_result = self.simple_test_mask(
                    x_i, img_meta_i, det_bboxes[img_id], det_labels[img_id] - 1, rescale=rescale, det_weight=det_weight,
                    with_point=self.with_point)
                if isinstance(test_result, tuple):
                    segm_masks_i, points_i = test_result
                    points.append(points_i)
                else:
                    segm_masks_i = test_result
                segm_masks.append(segm_masks_i)
            return det_bboxes, det_labels, score_dists, segm_masks, points

    def relation_simple_test(self,
                             img,
                             img_meta,
                             gt_bboxes=None,
                             gt_labels=None,
                             gt_rels=None,
                             gt_masks=None,
                             gt_scenes=None,
                             rescale=False,
                             ignore_classes=None,
                             key_first=False):
        """
        :param img:
        :param img_meta:
        :param gt_bboxes: Usually, under the forward (train/val/test), it should not be None. But
        when for demo (inference), it should be None. The same for gt_labels.
        :param gt_labels:
        :param gt_rels: You should make sure that the gt_rels should not be passed into the forward
        process in any mode. It is only used to visualize the results.
        :param gt_masks:
        :param rescale:
        :param ignore_classes: For practice, you may want to ignore some object classes
        :return:
        """
        # Extract the outer list: Since the aug test is temporarily not supported.
        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes[0]
        if gt_labels is not None:
            gt_labels = gt_labels[0]
        if gt_masks is not None:
            gt_masks = gt_masks[0]

        x = self.extract_feat(img)

        if self.relation_head.with_visual_mask and (not self.with_mask):
            raise ValueError('The basic detector did not provide masks.')

        """
        NOTE: (for VG) When the gt masks is None, but the head needs mask, 
        we use the gt_box and gt_label (if needed) to generate the fake mask. 
        """

        # Rescale should be forbidden here since the bboxes and masks will be used in relation module.
        bboxes, labels, target_labels, \
        dists, masks, points = self.detector_simple_test(x, img_meta, gt_bboxes, gt_labels,
                                                         gt_masks,
                                                         use_gt_box=self.relation_head.use_gt_box,
                                                         use_gt_label=self.relation_head.use_gt_label,
                                                         rescale=False, is_testing=True)

        saliency_maps = self.saliency_detector_test(img, img_meta) if self.with_saliency else None

        det_result = Result(bboxes=bboxes, labels=labels, dists=dists, masks=masks, points=points,
                            target_labels=target_labels, saliency_maps=saliency_maps,
                            img_shape=[meta['img_shape'] for meta in img_meta])

        det_result = self.relation_head(x, img_meta, det_result, is_testing=True, ignore_classes=ignore_classes)

        """
        Transform the data type, and rescale the bboxes and masks if needed 
        (for visual, do not rescale, for evaluation, rescale). 
        """
        scale_factor = img_meta[0]['scale_factor']
        return self.relation_head.get_result(det_result, scale_factor, rescale=rescale, key_first=key_first)

    def relcaption_simple_test(self,
                               img,
                               img_meta,
                               gt_bboxes=None,
                               gt_labels=None,
                               gt_masks=None,
                               gt_rels=None,
                               gt_relmaps=None,
                               gt_scenes=None,
                               gt_attrs=None,
                               gt_rel_inputs=None,  # for rel caption
                               gt_rel_targets=None,  # for rel caption
                               gt_rel_ipt_scores=None,  # for rel caption
                               gt_cap_inputs=None,  # for caption
                               gt_cap_targets=None,  # for caption
                               rescale=False
                               ):
        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes[0]
        if gt_labels is not None:
            gt_labels = gt_labels[0]

        x = self.extract_feat(img)

        bboxes, labels, target_labels, \
        dists, _, _ = self.detector_simple_test(x, img_meta, gt_bboxes, gt_labels,
                                                gt_masks,
                                                use_gt_box=self.relcaption_head.use_gt_box,
                                                use_gt_label=self.relcaption_head.use_gt_label,
                                                rescale=rescale, is_testing=True)

        det_result = Result(bboxes=bboxes, labels=labels, dists=dists,
                            target_labels=target_labels, target_scenes=gt_scenes,
                            img_shape=[meta['img_shape'] for meta in img_meta])

        det_result = self.relcaption_head(x, img_meta, det_result, is_testing=True)
        scale_factor = img_meta[0]['scale_factor']
        return self.relcaption_head.get_result(det_result, scale_factor, rescale=rescale)

    def downstream_caption_simple_test(self,
                                       img,
                                       img_meta,
                                       gt_bboxes=None,
                                       gt_labels=None,
                                       gt_masks=None,
                                       gt_rels=None,
                                       gt_relmaps=None,
                                       gt_scenes=None,
                                       gt_attrs=None,
                                       gt_rel_inputs=None,  # for rel caption
                                       gt_rel_targets=None,  # for rel caption
                                       gt_rel_ipt_scores=None,  # for rel caption
                                       gt_cap_inputs=None,  # for caption
                                       gt_cap_targets=None,  # for caption
                                       rescale=False
                                       ):
        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes[0]
        if gt_labels is not None:
            gt_labels = gt_labels[0]

        x = self.extract_feat(img)

        bboxes, labels, target_labels, \
        dists, _, _ = self.detector_simple_test(x, img_meta, gt_bboxes, gt_labels,
                                                gt_masks,
                                                use_gt_box=self.relcaption_head.use_gt_box,
                                                use_gt_label=self.relcaption_head.use_gt_label,
                                                rescale=rescale, is_testing=True)

        det_result = Result(bboxes=bboxes, labels=labels, dists=dists,
                            target_labels=target_labels, target_scenes=gt_scenes,
                            img_shape=[meta['img_shape'] for meta in img_meta])

        det_result = self.relcaption_head(x, img_meta, det_result, is_testing=True, downstreaming=True)
        roi_feats = self.relcaption_head.bbox_roi_extractor.roi_feats

        # generate the new cap_scores to cover the cap_scores from relational_caption_head
        det_result = self.downstream_caption_head(x, img_meta, roi_feats, det_result, is_testing=True)
        return self.downstream_caption_head.get_result(det_result)



    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        if self.bbox_head.__class__.__name__ == 'ExtrDetWeightSharedFCBBoxHead':
            det_weight = self.bbox_head.det_weight_hook()
        else:
            det_weight = None
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale, det_weight=det_weight, with_point=self.with_point)
            if len(results) == 2:
                segm_results, point_results = results
                return bbox_results, segm_results, point_results
            else:
                segm_results = results
                return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)
        if self.bbox_head.__class__.__name__ == 'ExtrDetWeightSharedFCBBoxHead':
            det_weight = self.bbox_head.det_weight_hook()
        else:
            det_weight = None

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels, det_weight)
            return bbox_results, segm_results
        else:
            return bbox_results
