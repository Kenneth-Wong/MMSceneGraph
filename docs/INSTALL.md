## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.4 or higher
- CUDA 9.0 or higher (tested on 10.1)
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04
- CUDA: 10.1
- GCC(G++): 5.4.0

### Install mmdetection

a. Create a conda virtual environment and activate it.

```shell
conda create -n mmsg python=3.8 -y
conda activate mmsg
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the MMSceneGraph repository.

```shell
git clone https://github.com/Kenneth-Wong/MMSceneGraph.git
cd MMSceneGraph
```

d. Install build requirements and then install mmdetection.
(We install pycocotools via the github repo instead of pypi because the pypi version is old and not compatible with the latest numpy.)

```shell
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .  # or "python setup.py develop"
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, mmdetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.


### Prepare datasets

It is recommended to symlink the dataset root to `$MMSCENEGRAPH/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.
Please refer to [DATA_INFO.md](DATA_INFO.md) to access the details of the datasets.

```
MMSceneGraph
├── mmdet
├── tools
├── configs
├── factories
├── requirements
├── tests
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```

### A from-scratch setup script

Here is a full script for setting up MMSceneGraph with conda and link the dataset path (supposing that your COCO dataset path is $COCO_ROOT).

```shell
conda create -n mmsg python=3.8 -y
conda activate mmsg

conda install -c pytorch pytorch torchvision -y
git clone https://github.com/Kenneth-Wong/MMSceneGraph.git
cd MMSceneGraph
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .

mkdir data
ln -s $COCO_ROOT data
```

### Using multiple MMDetection versions

If there are more than one mmdetection on your machine, and you want to use them alternatively, the recommended way is to create multiple conda environments and use different environments for different versions.

Another way is to insert the following code to the main scripts (`train.py`, `test.py` or any other scripts you run)
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```

Or run the following command in the terminal of corresponding folder to temporally use the current one.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
