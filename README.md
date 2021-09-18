# MMSceneGraph
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/Kenneth-Wong/MMSceneGraph/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.4.0-%237732a8)

## Introduction

MMSceneneGraph is an open source code hub for scene graph generation as well as supporting downstream tasks based on the scene graph on PyTorch. The frontend object detector is supported by [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection). 


### Major features

- **Modular Design**

  We decompose the framework into different components and one can easily construct a customized scene graph generation framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

## License

This project is released under the [MIT license](LICENSE).

## Changelog



## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] VGG (ICLR'2015)
- [x] HRNet (CVPR'2019)
- [x] RegNet (CVPR'2020)
- [x] Res2Net (TPAMI'2020)
- [x] ResNeSt (ArXiv'2020)

Supported methods:

- [x] [RPN (NeurIPS'2015)](configs/rpn)


## Installation

- mmcv and mmdetection: Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection) for installation. 

## Getting Started

## Contributing

## Acknowledgement

We appreciate the contributors of the [mmdetection](https://github.com/open-mmlab/mmdetection) project and [Scene-Graph-Benchmark.pytorch](https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/master/README.md) which inspires our design. 

## Citation

If you find this code hub or our works useful in your research works, please consider citing:

```
@inproceedings{wang2021topic,
  title={Topic Scene Graph Generation by Attention Distillation from Caption},
  author={Wang, Wenbin and Wang, Ruiping and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year={2021}
}


@inproceedings{wang2020sketching,
  title={Sketching Image Gist: Human-Mimetic Hierarchical Scene Graph Generation},
  author={Wang, Wenbin and Wang, Ruiping and Shan, Shiguang and Chen, Xilin},
  booktitle={Proceedings of European Conference on Computer Vision (ECCV)},
  pages={222--239},
  year={2020},
  volume={12358},
  doi={10.1007/978-3-030-58601-0_14},
  publisher={Springer}
}

@InProceedings{Wang_2019_CVPR,
author = {Wang, Wenbin and Wang, Ruiping and Shan, Shiguang and Chen, Xilin},
title = {Exploring Context and Visual Pattern of Relationship for Scene Graph Generation},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
pages = {8188-8197},
month = {June},
address = {Long Beach, California, USA},
doi = {10.1109/CVPR.2019.00838},
year = {2019}
}
```
