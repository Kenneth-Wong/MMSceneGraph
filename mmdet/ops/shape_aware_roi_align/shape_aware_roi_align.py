# ---------------------------------------------------------------
# shape_aware_roi_align.py
# Set-up time: 2020/4/16 下午9:57
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import shape_aware_roi_align_cuda
import numpy as np


class ShapeAwareRoIAlignFunction(Function):

    @staticmethod
    def forward(ctx,
                features,
                rois,
                masks,
                out_size,
                spatial_scale,
                sample_num=0):
        """
        rois: the final detected boxes, thus the coordinates must be integer.
        masks: a list of mask, [(mask_h, mask_w), ... ]
        """
        out_h, out_w = _pair(out_size)
        num_rois = rois.size(0)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        assert isinstance(masks, list) and len(masks) == num_rois
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        rois_int = rois[:, 1:].cpu().numpy().astype(np.int32)
        roi_widths = (rois_int[:, 2] - rois_int[:, 0] + 1)
        roi_heights = (rois_int[:, 3] - rois_int[:, 1] + 1)
        max_roi_width = int(roi_widths.max())
        max_roi_height = int(roi_heights.max())
        batch_masks = rois.new_zeros(num_rois, max_roi_height, max_roi_width)
        for i in range(num_rois):
            batch_masks[i, :roi_heights[i], :roi_widths[i]] = masks[i]

        ctx.save_for_backward(rois, batch_masks)
        ctx.feature_size = features.size()

        if features.is_cuda:
            (batch_size, num_channels, data_height,
             data_width) = features.size()

            output = features.new_zeros(num_rois, num_channels, out_h,
                                        out_w)
            shape_aware_roi_align_cuda.forward(features, rois, batch_masks, out_h, out_w,
                                               max_roi_height, max_roi_width,
                                               spatial_scale, sample_num, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        masks = ctx.saved_tensors[1]
        bs_mask_height = masks.size(1)
        bs_mask_width = masks.size(2)

        assert (feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = grad_masks = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels,
                                        data_height, data_width)
            shape_aware_roi_align_cuda.backward(grad_output.contiguous(), rois, masks,
                                       out_h, out_w, bs_mask_height, bs_mask_width, spatial_scale,
                                       sample_num, grad_input)

        return grad_input, grad_rois, grad_masks, None, None, None


shape_aware_roi_align = ShapeAwareRoIAlignFunction.apply


class ShapeAwareRoIAlign(nn.Module):

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0):
        """
        Args:
            out_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sample_num (int): number of inputs samples to take for each
                output sample. 2 to take samples densely for current models.

        Note:
            The implementation of RoIAlign when aligned=True is modified from
            https://github.com/facebookresearch/detectron2/

            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel
            indices (in our pixel model) are computed by floor(c - 0.5) and
            ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal
            at continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing
            neighboring pixel indices and therefore it uses pixels with a
            slightly incorrect alignment (relative to our pixel model) when
            performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors;

            The difference does not make a difference to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ShapeAwareRoIAlign, self).__init__()
        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois, masks):
        """
        Args:
            features: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4
            columns are xyxy.
            masks: lists, (B,)

        """
        assert rois.dim() == 2 and rois.size(1) == 5
        assert len(masks) == rois.size(0)

        return shape_aware_roi_align(features, rois, masks, self.out_size, self.spatial_scale, self.sample_num)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(
            self.out_size, self.spatial_scale, self.sample_num)
        return format_str
