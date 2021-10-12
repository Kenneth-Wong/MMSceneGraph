# Data Settings
We provide the information of the settings of data.

## Stanford Filtered data (VG150)
Adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
Follow the steps to get the dataset set up.
1. Create the data folder `data/visualgenome` and put all of the following items under it. 

2. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images and put the two parts, `VG_100K` and `VG_100K_2` into folder `data/visualgenome/Images`. 

3. Download the [recsize_image_data.json](https://drive.google.com/file/d/14S8LUfdhAj_DyaUv5rBbMgw33TIF2FYq/view?usp=sharing) (The corrected version of original image_data.json, because there exists something wrong about the image sizes) [~~image_data.json~~](http://cvgl.stanford.edu/scene-graph/VG/image_data.json)
and put in to `data/visualgenome/recsize_image_data.json`.

4. Download the [VG-SGG-with-attri.h5](https://drive.google.com/file/d/1cKC2nqFyHs6VQXga9dLWCHXYef7Xpl45/view?usp=sharing) (Kaihua's version) and extract it to `data/visualgenome/VG-SGG-with-attri.h5`. 

5. Download the [VG-SGG-dicts-with-attri.json](https://drive.google.com/file/d/1yHnG1_YdlPzpMq5AuPuxRGpr4jUwtjH_/view?usp=sharing) and extract it to `data/stanford_filtered/VG-SGG-dicts-with-attri.json`.

6. (Optional, used in HET(ECCV2020) ) The saliency map: We use [DSS](https://github.com/Andrew-Qibin/DSS) to generate the saliency map. 
Please refer to the DSS and follow their setup and generate it yourself and load the images directly with opencv or PIL, etc. 
You may put it under `data/visualgenome` and name it as `saliency_512.imdb`.

## VGKR_v1
It uses the data under `data/visualgenomekr_ingredients/v1` to generate the visualgenomekr dataset used in 
our paper [Sketching Image Gist: Human-Mimetic Hierarchical Scene Graph Generation (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580222.pdf) (conference version) and the journal version (**published soon**).
1. Follow the settings of **Stanford filtered data**.
2. Create the data folder `data/visualgenomekr`, `data/visualgenomekr_ingredients`, `data/visualgenomekr_ingredients/public_inputs`, and `data/visualgenomekr_ingredients/v1`.

3. Create some necessary soft links under `data/visualgenomekr`, such as `data/visualgenomekr/Images`, `data/visualgenomekr/recsize_image_data.json`, and `data/visualgenomekr/saliency_512.h5`. 

4. Download the [VGKR](https://drive.google.com/drive/folders/1g7Fmfm64Ja1cXCo1Pv0ZvreaLQpdkBQ3?usp=sharing) annotation. It contains two files: `VG200-SGG-dicts.json` and `VG200-SGG.h5` and put them under `data/visualgenomekr`. In the `VG200-SGG.h5`, there exist indicative key relation annotations.
Put them under `data/visualgenomekr`.


5. (Optional) You can also create the **VGKR** yourself. We provide the scripts and raw data here. Before running the scripts, remember to set your PYTHONPATH: ```export PYTHONPATH=/home/YourName/ThePathOfYourProject```. All the scripts should be run from the project root. 
    1. Create the data folder `data/visualgenomekr_ingredients`, `data/visualgenomekr_ingredients/public_inputs`, and `data/visualgenomekr_ingredients/v1`.

    2. Prepare the raw VG data (The first three items can be found on the Visual Genome site) and our provided [VG raw data][1] (the 4th \~ 6th item) and put them under `data/visualgenomekr_ingredients/public_inputs`:

        - `imdb_1024.h5`, `imdb_512.h5` (This is optional and you can also use the raw images).

        - `object_alias.txt`, `relationship_alias.txt`.

        - `objects.json`, `relationships.json`.
        
        - MSCOCO captions [captions_vgOVcoco.json][1]. It contains the captions of 51,498 images that in both the VG and COCO dataset.
        
        - [object_list.txt][1], [predicate_list.txt][1], [predicate_stem.txt][1].
        
        - Use the [Stanford Scene Graph Parser](https://nlp.stanford.edu/software/scenegraph-parser.shtml) to transform the captions into scene graphs, or download our file [captions_to_sg_v1.json][1].
    3. Prepare the word embedding vectors from GloVe. Put the data files under the folder `data/GloVe`. 

    4. Run the script [cleanse_raw_vg.py](vgkr_v1/cleanse_raw_vg.py) to generate two files or download our files:
     [cleanse\_objects.json][1], [cleanse_relationships.json][1]. It is expected to be under `data/visualgenomekr_ingredients/v1`.
    
    5. Run the script [triplet_match.py](vgkr_v1/triplet_match.py) to generate or download the file
    [cleanse\_triplet\_match.json][1]. It is expected to be under `data/visualgenomekr_ingredients/v1`.
        
    6. Run the script [vg_to_roidb.py](vgkr_v1/vg_to_roidb.py) to generate the final annotation files: `VG200-SGG-dicts.json` and `VG200-SGG.h5` under `data/visualgenomekr/`. 

## VGKR_v2

The VGKR\_v2 is a developing dataset. There exist some differences from the VGKR_v1.
1. Create the folder `data/visualgenomekr_ingredients/v2`.

2. In the step of VGKR_v1.5.ii, we additionally make use of the [coco_entities_release.json](http://ailb-web.ing.unimore.it/releases/show-control-and-tell/coco_entities_release.json) (put it under `data/visualgenomekr_ingredients/public_inputs`) from [aimagelab/show-control-and-tell](https://github.com/aimagelab/show-control-and-tell).
Thanks for their contributions. We provide jupyter notebooks ([extract_vgset.ipynb](extract_vgset.ipynb) to get different vg splits 
and [extract_vgkeyrel.ipynb](extract_vgkeyrel.ipynb)) for generating the following related files. 
    1. The original captions are added to the `captions_to_sg_v1.json` for convenience. We also add the attributes. This resulted in the [captions_to_sg_v2.json][2] under `data/visualgenomekr_ingredients/public_inputs`.

    2. ~~\[Deprecated\] In the step of VGKR_v1.5.iv, we use the [vacancy/SceneGraphParser](https://github.com/vacancy/SceneGraphParser) and put this parser under `factories/utils/sng_parser`.
Thanks for their contributions. Use this file `captions_to_sg.json` instead, or use [transform_captions_to_sg.py](vgkr_v2/deprecated_transform_captions_to_sg.py) to generate it.~~
    
    3. Using the `captions_to_sg_v2.json` and `coco_entities_release.json`, we finally obtained the `cap_sgentities_vgcoco.json` (put it under `data/visualgenomekr_ingredients/v2`),
    which contains 51,208 images with both scene graphs from captions and the entities of captions (NOTE: 16 images do not contain entities, because they are not in `coco_entities_release.json`). 
    
        
3. In VGKR_v1 step 5.iv, we run the script [cleanse_raw_vg.py](vgkr_v2/cleanse_raw_vg.py) to generate two files instead or download our files:
     `cleanse_objects.json`, `cleanse_relationships.json`, and the additional `cleanse_attributes.json`. It is expected to be under `data/visualgenomekr_ingredients/v2`.
     In this v2 script, we do not use the most frequent name to replace the original name of each object. 

4. Finally, through matching (see [extract_vgkeyrel.ipynb](extract_vgkeyrel.ipynb) ),
we have 26,234 images (446,117 relationships, 68,937 key relationships) with key relations among 51,208 images (746,018 relationships).
    
    - **NOTE**: since there are various splits for visualgenome, we generate the `meta_form.csv` (put it under `data/visualgenome/`) to record the most completed information of each image.
    
            - meta_dbidx: int, consecutive from 0 to 108,072
            - path_vgids: extracted from the paths from the `recsize_image_data.json`	
            - meta_vgids: str	
            - meta_cocoids: str or 'None'	
            - meta_flickr_ids: str or 'None'	
            - meta_paths: 'VG_100K/XXX.jpg' or 'VG_100K_2/XXX.jpg'	
            - meta_heights: int	
            - meta_widths: int	
            - vg150_split: train(75,651) + test(32,422)	= 108,073
            - vgcoco_split: train(35,155) + test(16,053) = 51,208
            - vgcoco_entity_split: train(35,145) + test(16,047) = 51,192
            - vgcoco_keyrel_split: train(18,197) + test(7,734) = 25,931
            - filter_coco_det_val_split: train(74,134) + test(32,422) = 106,556 (filter the images in COCO17 detection val set, 118,287(train) + 5,000(val), filtered 1,517 images in the VG)
            - filter_coco_karpathycap_testval_split: train(72,697) + test(32,422) = 105,119 (filter the images in COCO karpathy captioning test/val set, 113,287(train) + 5,000(val) + 5,000(test), filtered 2,954 images in the VG)

## caption_coco

This is the MSCOCO for image captioning. As the [karpathy split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) has become the standard of image captioning, 
we follow the process of previous works. (113,287 train + 5,000 val + 5,000 test)

1. Create the folder `data/caption_coco`, `data/caption_coco/bottomup`, `data/caption_coco/karpathy_captions`.

2. Download the bottom-up features and put them under `data/caption_coco/bottomup`. Run the script [factories/caption_coco/transform_bottomup_feats.py](caption_coco/transform_bottomup_feats.py)
and it will produce the `up_down_10_100` folder under `data/caption_coco` by default. It contains three split, train (113,287), val (5,000), and test (5,000).
Each of them contains `feature` and `boxsize` sub-folders. 

3. We use the processed [annotations](https://drive.google.com/open?id=1i5YJRSZtpov0nOtRyfM0OS1n0tPCGiCS) from [X-LAN](https://github.com/JDAI-CV/image-captioning).
extract it and put all the files under `data/caption_coco/karpathy_captions`. 
```
karpathy_captions
├── captions_test5k.json
├── captions_val5k.json
├── coco_test4w_image_id.txt
├── coco_test_image_id.txt
├── coco_train_cider.pkl
├── coco_train_gts.pkl
├── coco_train_image_id.txt
├── coco_train_input.pkl
├── coco_train_target.pkl
├── coco_val_image_id.txt
└── coco_vocabulary.txt
```

### Generate yourself!
As the files above are directly used from XLAN project and they do not provide the scripts for obtaining these files, 
we create some scripts for generating them. Specifically, we additionally generate the `coco_val_input.pkl` and `coco_val_target.pkl` 
because in this project, we also want to monitor the loss of the validation set during training, keeping coordinate with the mmdet project!

1. Download the [karpathy split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and extract it. 
you will get `dataset_coco.json`. It contains totally 123,287 images and their raw captions, the tokenized sequences. The structure is:
```
{
'dataset': 'coco'
'images': [
    {
        'filepath': 'val2014'/....,
        'sentids': list(int),
        'filename': 'COCO_val2014_XXXXXXXXXXXX.jpg',
        'imgid': int,
        'split': val/restval/.....
        'sentences': [
            {
            'tokens': list(str),
            'raw': str,
            'imgid': int,
            'sentid': int
            },
           ...
        ],
        'cocoid': int
    },
...
]
}
```
Run the script [generate_labels_for_train.py](caption_coco/generate_labels_for_train.py)
to generate it. You should set the proper file paths in it. E.g., it need the `dataset_coco.json`, `coco_vocabulary.txt`, and `coco_val_img_id.txt` as input. 
You will obtain `coco_val_input.pkl`, and `coco_val_target.pkl`. You can also run the script to get `coco_test_input.pkl` and `coco_test_target.pkl`.


## Relational Caption
This is the dataset from the work: Dense Relational Image Captioning
via Multi-task Triple-Stream Networks, CVPR19.

## Generalized Visual Genome
This is the byproduct of VGKR\_V2. We provide the script [get_generalized_roidb.py](vgkr_v2/get_generalized_roidb.py)
for generate the data. The outcome dataset can be placed under ```data/visualgenomegn```, which includes
```
visualgenomegn
├── meta_form.csv
├── VGGN-SGG-dicts.json
├── VGGN-SGG-sentences.json
├── VGGN-SGG.h5
└── Images/
```
[1]: https://drive.google.com/drive/folders/16ZHThqz72yQTh4759qMphwVp8VjgWg-Y?usp=sharing
[2]: https://drive.google.com/drive/folders/1H3w7KjZBXobscz1rScsj0Fe7qEO2LaPz?usp=sharing