## Dataset Information
This document is for recording the information of data under `data/`. You can create the symlinks whose names are the following datasets under the `data/`. 



### coco
This is a MSCOCO (coco17) dataset mainly for object detection bechmark. You can follow the official guidance from MSCOCO to prepare the dataset. 

### caption\_coco
This is a MSCOCO dataset for image captioning. 

### coco\_vg\_mapping
It contains the class mapping from coco (consecutive: 1\~90; discrete: 1\~80) to visualgenome (150).

### glove
The GloVe vectors. 

### vg_evaluation
It is from [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) used for 
some scene graph evaluation protocols referred in [Unbiased Scene Graph Generation from Biased Training (CVPR 2020)](https://arxiv.org/abs/2002.11949)

### visualgenome
This is the mainly dataset for scene graph generation. Refer to [factories/README.md](../factories/README.md) for details. 

### visualgenomekr
This is the dataset developed based on the visualgenome. Please refer to our paper [Sketching Image Gist: Human-Mimetic Hierarchical Scene Graph Generation (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580222.pdf)
and [factories/README.md](../factories/README.md) for details about accessing the dataset. 

### visualgenomegn
This is the dataset developed based on the visualgenome and used by our TopicSG in ICCV 2021. Refer to [factories/README.md](../factories/README.md) for details about accessing the dataset


### vrd
This is a dataset for visual relationship detection and an auxiliary dataset for evaluating the SGG algorithms. 


### SOC and MSRA10K
The datasets for salient object detection.
