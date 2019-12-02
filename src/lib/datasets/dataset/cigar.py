from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class Cigar(data.Dataset):
    num_classes = 20
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(Cigar, self).__init__()
        dt_name = 'cigar_box'
        self.data_dir = os.path.join(opt.data_dir, dt_name)
        self.img_dir = ''  # as file_name is absolute path
        # self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                '{}_{}_{}.json').format(Cigar.num_classes, dt_name, split)
        else:
            if opt.task == 'exdet':  # train or val
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    '{}_{}_{}.json').format(Cigar.num_classes, dt_name, split)
            else:  # ctdet,..?
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    '{}_{}_{}.json').format(Cigar.num_classes, dt_name, split)

        self.max_objs = 10
        # self.max_objs = 128
        self.class_name = [
            '__background__',
            'DaZhongJiu_A', 'YunYan_a', 'JiaoZi_B', 'ZhongHua_B',
            'LiQun_a', 'HuangHeLou_e', 'JiaoZi_F', 'YunYan_A',
            'HuangHeLou_h', 'HuangHeLou_E', 'HuangJinYe_C', '555_a',
            'HongTaShan_b', 'YuXi_A', 'JiaoZi_K', 'HuangHeLou_A',
            'JiaoZi_E', 'TianZi_a', 'TianZi_A', 'WangGuan_A'
        ]
        # _valid_ids same to real cat_id in xx.json
        self._valid_ids = np.arange(1, 21, dtype=np.int32)

        self.cat_ids = {
            v: i for i, v in enumerate(self._valid_ids)
        }
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing cigar {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # save det results to local, then load
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(cocoGt=self.coco,
                             cocoDt=coco_dets,
                             iouType="bbox")  # initialize CocoEval object

        coco_eval.evaluate()  # run per image evaluation
        coco_eval.accumulate()  # accumulate per image results
        coco_eval.summarize()  # display summary metrics of results
