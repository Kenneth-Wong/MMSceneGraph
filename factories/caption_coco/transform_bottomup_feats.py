# ---------------------------------------------------------------
# transform_bottomup_feats.py
# Set-up time: 2020/12/29 11:32
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import os
import base64
import numpy as np
import csv
import sys
import argparse
import os.path as osp
import mmcv

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def main(args):
    # coco test 4w: 300104, 147295, 321486 are corrupted, ignore them
    count = 0
    with open(args.infeats, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            if count % 1000 == 0:
                print(count)
            count += 1

            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                try:
                    item[field] = np.frombuffer(base64.urlsafe_b64decode(item[field]),
                                                dtype=np.float32).reshape((item['num_boxes'], -1))
                except:
                    print('error:', item['image_id'])
                finally:
                    pass
            image_id = item['image_id']

            feats = item['features']
            boxes = item['boxes']
            np.savez_compressed(osp.join(osp.join(args.outfolder, args.subfolder, args.split, 'feature'), str(image_id)),
                                feat=feats)
            box_and_size = {'boxes': boxes, 'h': item['image_h'], 'w': item['image_w']}
            mmcv.dump(box_and_size, osp.join(args.outfolder, args.subfolder, args.split, 'boxsize', str(image_id)+'.pickle'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--infeats', default='data/caption_coco/bottomup/test2014/test2014_resnet101_faster_rcnn_genome.tsv.2', help='image features')
    parser.add_argument('--outfolder', default='data/caption_coco/bottomup/', help='output folder')
    parser.add_argument('--subfolder', default='up_down_10_100', help='output sub folder')
    parser.add_argument('--split', default='test4w')

    args = parser.parse_args()
    main(args)