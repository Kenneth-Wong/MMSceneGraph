import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from .custom import CustomDataset
from .coco import CocoDataset
from .registry import DATASETS
from mmdet.core import vg_evaluation
from mmdet.models.relation_heads.approaches import Result


@DATASETS.register_module
class AithorDataset(CocoDataset):
    CLASSES = ('AlarmClock', 'AluminumFoil', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub',
               'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'BreadSliced',
               'ButterKnife', 'Cabinet', 'Candle', 'CD', 'CellPhone', 'Chair', 'Cloth', 'CoffeeMachine', 'CoffeeTable',
               'CounterTop', 'CreditCard', 'Cup', 'Curtains', 'Desk', 'DeskLamp', 'Desktop', 'DiningTable',
               'DishSponge', 'DogBed', 'Drawer', 'Dresser', 'Dumbbell', 'Egg', 'EggCracked', 'Faucet', 'Floor',
               'FloorLamp', 'Footstool', 'Fork', 'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowel', 'HandTowelHolder',
               'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'Lettuce',
               'LettuceSliced', 'LightSwitch', 'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan',
               'PaperTowel', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Poster', 'Pot', 'Potato',
               'PotatoSliced', 'RemoteControl', 'RoomDecor', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf',
               'ShelvingUnit', 'ShowerCurtain', 'ShowerDoor', 'ShowerGlass', 'ShowerHead', 'SideTable', 'Sink',
               'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'Stool',
               'StoveBurner', 'StoveKnob', 'TableTopDecor', 'TargetCircle', 'TeddyBear', 'Television', 'TennisRacket',
               'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'Tomato', 'TomatoSliced', 'Towel',
               'TowelHolder', 'TVStand', 'VacuumCleaner', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['file_name'] = '_'.join(info['file_name'].split('/')[0::2])
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        # NOTE: For SGG, since the forward process may need gt_bboxes/gt_labels,
        # we should also load annotation as if in the training mode.
        anno_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=anno_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0] + 1,
            _bbox[3] - _bbox[1] + 1,
        ]

    def _proposal2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            result_files['segm'] = '{}.{}.json'.format(outfile_prefix, 'segm')
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'proposal')
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='predcls',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 multiple_preds=False,
                 iou_thrs=0.5,
                 nogc_thres_num=None,
                 **kwargs):
        """
        **kwargs: contain the paramteters specifically for OD, e.g., proposal_nums.
        Overwritten evaluate API:
            For each metric in metrics, it checks whether to invoke od or sg evaluation.
            if the metric is not 'sg', the evaluate method of super class is invoked
            to perform Object Detection evaluation.
            else, perform scene graph evaluation.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_sg_metrics = ['predcls', 'sgcls', 'sgdet']
        allowed_od_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        sg_metrics, od_metrics = [], []
        for m in metrics:
            if m in allowed_od_metrics:
                od_metrics.append(m)
            elif m in allowed_sg_metrics:
                sg_metrics.append(m)
            else:
                raise ValueError("Unknown metric {}.".format(m))

        if len(od_metrics) > 0:
            # invoke object detection evaluation.
            # Temporarily for bbox
            od_results = [r.formatted_bboxes for r in results]
            return super(AithorDataset, self).evaluate(od_results,
                                                       metric,
                                                       logger,
                                                       jsonfile_prefix,
                                                       classwise=classwise,
                                                       iou_thrs=iou_thrs,
                                                       **kwargs)
        if len(sg_metrics) > 0:
            """ 
                Invoke scenen graph evaluation. prepare the groundtruth and predictions.
                Transform the predictions of key-wise to image-wise.
                Both the value in gt_results and det_results are numpy array.
            """
            if not hasattr(self, 'test_gt_results'):
                print('\nLooading testing groundtruth...\n')
                prog_bar = mmcv.ProgressBar(len(self))
                gt_results = []
                for i in range(len(self)):
                    ann = self.get_ann_info(i)
                    gt_results.append(Result(bboxes=ann['bboxes'],
                                             labels=ann['labels'],
                                             rels=ann['rels'],
                                             relmaps=ann['rel_maps'],
                                             rel_pair_idxes=ann['rels'][:, :2],
                                             rel_labels=ann['rels'][:, -1],
                                             attrs=ann['attrs']))
                    prog_bar.update()
                print('\n')
                self.test_gt_results = gt_results

            return vg_evaluation(sg_metrics,
                                 groundtruths=self.test_gt_results,
                                 predictions=results,
                                 iou_thrs=iou_thrs,
                                 logger=logger,
                                 ind_to_predicates=self.ind_to_predicates,
                                 multiple_preds=multiple_preds,
                                 predicate_freq=self.predicate_freq,
                                 nogc_thres_num=nogc_thres_num)
