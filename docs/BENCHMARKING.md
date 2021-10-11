# Object Detection Model Benchmark

Here is the benchmark for some detection models used in scene graph generation.

## Environment

### Hardware

- Totally 4 NVIDIA Titan X  GPUs
- 32x Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz

### Software environment

- Python 3.6 / 3.7 / 3.8
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.5
- NCCL 2.1.15

## Common settings

- All FPN baselines and RPN-C4 baselines were trained with 2 images per GPU. Other C4 baselines were trained with 1 image per GPU.
- The base-lr is 0.02 for batchsize 16 (8 GPUs x 2), and you should adjust it linearly for the actual number of GPUs you use.
- We use distributed training and BN layer stats are fixed.
- We adopt the same training schedules as Detectron. 1x indicates 12 epochs and 2x indicates 24 epochs, which corresponds to slightly less iterations than Detectron and the difference can be ignored.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the overall time including data loading, network forwarding and post processing (on a single GPU).
- The results in a same grid stand for (train/val/test) splits. 




##  Faster R-CNN (default: test on all classes)

- -ftCOCO / -ftVG: it means that the model is previously finetuned on COCO or VG. 

|    Backbone     |       method       |   dataset    | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) |       AP       |      AP50      |      AP75      |     AP_S     |     AP_M      |      AP_L      |
| :-------------: | :----------------: | :----------: | :-----: | :------: | :-----------------: | :------------: | :------------: | :------------: | :------------: | :----------: | :-----------: | :------------: |
| X-101-64x4d-FPN |    Faster-RCNN     |      VG      |   1x    |   10.2   |        1.17         |      6.2       | 25.1/13.8/14.7 | 45.9/27.0/29.9 | 25.0/12.6/12.7 | 17.7/7.1/5.6 | 23.7/9.6/12.2 | 26.1/17.5/17.9 |
| X-101-64x4d-FPN |    Faster-RCNN     | VG\COCOval17 |   1x    |   10.2   |        1.17         |      6.2       |  25.9/-/15.2   |  45.1/-/29.5   |  26.9/-/14.0   |  17.5/-/5.9  |  24.3/-/12.6  |  27.4/-/18.7   |
| X-101-64x4d-FPN |    Faster-RCNN     |     VRD      |   1x    |   10.2   |        1.31         |      6.0       |  42.7/-/13.3   |  72.1/-/27.7   |  46.7/-/10.9   |  33.8/-/5.0  |  44.5/-/9.1   |  43.4/-/16.3   |
| X-101-64x4d-FPN | Faster-RCNN-ftCOCO |     VRD      |   1x    |   10.2   |        1.12         |      6.0       |  63.0/-/19.2   |  87.4/-/33.2   |  76.1/-/20.6   |  55.2/-/5.8  |  63.1/-/11.3  |  64.7/-/23.0   |
| X-101-64x4d-FPN |  Faster-RCNN-ftVG  |     VRD      |   1x    |   10.2   |        1.12         |      6.0       |  62.1/-/17.8   |  88.6/-/33.2   |  75.2/-/17.1   |  55.5/-/3.6  |  62.7/-/11.5  |  63.5/-/21.4   |
| X-101-64x4d-FPN |    Faster-RCNN     |     VGKR     |   1x    |   10.2   |        1.12         |      6.0       |    -/-/13.9    |    -/-/27.0    |    -/-/12.8    |   -/-/5.4    |   -/-/11.3    |    -/-/16.9    |

## RepPoints

|        Backbone        |  method   | dataset | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) |  AP  | AP50 | AP75 | AP_S | AP_M | AP_L |
| :--------------------: | :-------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :--: | :--: | :--: | :--: | :--: | :--: |
| X-101-dcn-32x4d-FPN-mt | RepPoints |   VG    |   2x    |   8.5    |         1.6         |      6.65      | 15.6 | 29.3 | 14.7 | 5.7  | 12.7 | 19.5 |



## Reimplemented Methods

###  1. Mask-X-RCNN (CVPR 2018)

The Training procedure is:
- stage 1: Use the config: visualgenome/MASKTRANS_faster_rcnn_x101_64x4d_fpn_1x.py to train a faster RCNN on VG150.
- stage 2: Use the config: visualgenome/MASKTRANS_mask_rcnn_x101_64x4d_fpn_1x.py, which freezes the faster-RCNN 
  parameters, and only train the mask head and transfer function on the COCO17 training set with 53 categories. 
- stage 3: Visualization: use the config visualgenome/MASKTRANS_VISVG_mask_rcnn_x101_64x4d_fpn_1x.py to visualize 
  the segmentation result on VG.

###  2.  Mask R-CNN trained for Transfering Instance Segmentation (box AP on 53 categories).

Note: This model is the same as the Faster-RCNN trained on VG\COCOval17, and it's evaluated on COCO17val's 53 categories which can be mapped to VG150.

|    Backbone     |        method        | dataset  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) |  AP  | AP50 | AP75 | AP_S | AP_M | AP_L |
| :-------------: | :------------------: | :------: | :-----: | :------: | :-----------------: | :------------: | :--: | :--: | :--: | :--: | :--: | :--: |
| X-101-64x4d-FPN | Mask-RCNN(frz FRCNN) | COCO(53) |   1x    |   5.4    |        1.13         |      6.2       | 24.4 | 44.8 | 24.1 | 10.1 | 27.9 | 35.9 |

###  3. Mask R-CNN trained for Transfering Instance Segmentation (mask AP on 53 categories).

- The Faster-RCNN part of this model is the same as the one trained on VG\COCOval17, and it's evaluated on COCO17val's 53 categories which can be mapped to VG150. 
- The Mask-RCNN is slower since it has another RoI Align Module.

|    Backbone     |        method        | dataset  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) |  AP  | AP50 | AP75 | AP_S | AP_M | AP_L |
| :-------------: | :------------------: | :------: | :-----: | :------: | :-----------------: | :------------: | :--: | :--: | :--: | :--: | :--: | :--: |
| X-101-64x4d-FPN | Mask-RCNN(frz FRCNN) | COCO(53) |   1x    |   5.4    |        1.13         |      3.5       | 25.0 | 43.3 | 25.9 | 11.7 | 28.6 | 36.2 |



## Caption Graph (Relational Caption)

1. Use the Generalized VG (3,000 classes, 800 attributes) to pretrain the Faster RCNN detector. 1,000 images are used for validation. We use the `filter_coco_karpathycap_testval_split`. See more details in the config file [configs/generalized_visualgenome/faster\_rcnn\_x101\_64x4d\_fpn\_1x.py](https://github.com/Kenneth-Wong/Scene-Graph-Benchmark-mmdet.pytorch/blob/master/configs/generalized_visualgenome/faster_rcnn_x101_64x4d_fpn_1x.py). We train it on 4 TITAN RTX GPUs. 
  ​

|    Backbone     |   method    | dataset | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) |   AP    |   AP50   |  AP75   |  AP_S   |  AP_M   |   AP_L   |   AR_1   |  AR_10   |  AR_100  |   AR_S   |   AR_M   |   AR_L   |
| :-------------: | :---------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :-----: | :------: | :-----: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| X-101-64x4d-FPN | Faster-RCNN |  VGGN   |   1x    |   10.5   |        0.92         |      6.9       | 7.1/2.4 | 12.8/4.9 | 6.8/2.1 | 7.9/1.8 | 6.6/2.5 | 10.6/3.4 | 11.8/6.2 | 15.4/8.4 | 15.5/8.5 | 12.1/6.4 | 13.8/8.4 | 19.7/9.4 |
