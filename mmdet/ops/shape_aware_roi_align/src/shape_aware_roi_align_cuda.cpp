#include <torch/extension.h>

#include <ATen/ATen.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int ShapeAwareROIAlignForwardLaucher(const at::Tensor features, const at::Tensor rois,
                                     const at::Tensor masks,
                                     const float spatial_scale, const int sample_num,
                                     const int channels, const int height,
                                     const int width, const int num_rois,
                                     const int pooled_height, const int pooled_width,
                                     const int bs_mask_height, const int bs_mask_width,
                                     at::Tensor output);

int ShapeAwareROIAlignBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
                                      const at::Tensor masks,
                                      const float spatial_scale, const int sample_num,
                                      const int channels, const int height,
                                      const int width, const int num_rois,
                                      const int pooled_height, const int pooled_width,
                                      const int bs_mask_height, const int bs_mask_width,
                                      at::Tensor bottom_grad);
#endif

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int ShapeAwareROIAlign_forward(at::Tensor features, at::Tensor rois, at::Tensor masks,
                              int pooled_height, int pooled_width, int bs_mask_height, int bs_mask_width,
                              float spatial_scale, int sample_num, at::Tensor output) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(masks);
  CHECK_INPUT(output);
  at::DeviceGuard guard(features.device());

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 5) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);

  ShapeAwareROIAlignForwardLaucher(features, rois, masks, spatial_scale, sample_num,
                                   num_channels, data_height, data_width, num_rois,
                                   pooled_height, pooled_width, bs_mask_height, bs_mask_width,
                                   output);

  return 1;
}

int ShapeAwareROIAlign_backward(at::Tensor top_grad, at::Tensor rois, at::Tensor masks, int pooled_height,
                                int pooled_width, int bs_mask_height, int bs_mask_width,
                                float spatial_scale, int sample_num,
                                at::Tensor bottom_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(rois);
  CHECK_INPUT(masks);
  CHECK_INPUT(bottom_grad);
  at::DeviceGuard guard(top_grad.device());

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = bottom_grad.size(1);
  int data_height = bottom_grad.size(2);
  int data_width = bottom_grad.size(3);

  ShapeAwareROIAlignBackwardLaucher(top_grad, rois, masks, spatial_scale, sample_num,
                                    num_channels, data_height, data_width, num_rois,
                                    pooled_height, pooled_width, bs_mask_height, bs_mask_width,
                                    bottom_grad);

  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ShapeAwareROIAlign_forward, "Shape_Aware_Roi_Align forward (CUDA)");
  m.def("backward", &ShapeAwareROIAlign_backward, "Shape_Aware_Roi_Align backward (CUDA)");
}
