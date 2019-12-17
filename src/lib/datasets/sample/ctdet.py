from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math


class CTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        # xywh -> x1y1x2y2
        bbox = np.array([box[0], box[1],
                         box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        # all anns of one img
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        # height, width
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # ori img center

        if self.opt.keep_res:  # False
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            # not keep_res, use opt.input_h, w
            # note: h != w, ori not keep_res, then set w=h=512
            # s = max(img.shape[0], img.shape[1]) * 1.0
            s = np.array([width, height], dtype=np.float32)  # ori img size?
            input_h, input_w = self.opt.input_h, self.opt.input_w

        # flip
        flipped = False

        if self.split == 'train':
            # random scale
            if not self.opt.not_rand_crop:
                # train set opt.not_rand_crop=False, so will use default random scale
                # s = s * np.random.choice(np.arange(0.4, 0.6, 0.1))  # (1920,1080) -> (640)
                # note: restrict the img center translate range, lrtb 1/2
                # w_border = self._get_border(img.shape[1] // 4, img.shape[1])
                # h_border = self._get_border(img.shape[0] // 4, img.shape[0])
                # random center, this may translate img so far
                w_range, h_range = img.shape[1] // 8, img.shape[0] // 8
                c[0] = np.random.randint(low=img.shape[1] // 2 - w_range,
                                         high=img.shape[1] // 2 + w_range)
                c[1] = np.random.randint(low=img.shape[0] // 2 - h_range,
                                         high=img.shape[0] // 2 + h_range)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            # random flip
            if np.random.random() < self.opt.flip:  # 0.5
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        # trans ori img to input size
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        # use generated trans_input matrix to trans img
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        # note: see trans img
        # print('scale:', s, 'center:', c)
        # cv2.imwrite('{}_img_trans.png'.format(img_id), inp)
        inp = (inp.astype(np.float32) / 255.)

        # color augment
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        # normalize
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        # down sample
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes

        # trans ori img box to output size
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)  # 20
        # todo: dense or sparse wh
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)  # (10,2) sparse!
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)  # dense!
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)  # (10,2)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        # msra, umich
        # opt.mse_loss = False
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        # GT
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])  # xywh -> x1y1x2y2
            # map ori cat_id (whatever) to [0, num_class-1]
            cls_id = int(self.cat_ids[ann['category_id']])  # self.cat_ids in cigar.py
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            # transform box 2 pts to output
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            # todo: redefine the center and w,h
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]  # x1y1x2y2
            if h > 0 and w > 0:
                # note: radius generated with spatial extent info from h,w
                radius = gaussian_radius(det_size=(math.ceil(h), math.ceil(w)))
                radius = max(0, int(math.ceil(radius / 3)))
                # radius = max(0, int(radius))
                # opt.mse_loss = False
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                # center
                ct = np.array([(bbox[0] + bbox[2]) / 2,
                               (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                # label of w,h
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]  # 1D ind
                reg[k] = ct - ct_int  # float - int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        ret = {
            'input': inp,
            'hm': hm,
            'reg_mask': reg_mask,
            'ind': ind,
            'wh': wh
        }

        # from utils.plt_utils import plt_heatmaps
        # note: see heatmaps
        # plt_heatmaps(hm, basename='{}_hm'.format(img_id))
        # print(wh)

        if self.opt.dense_wh:  # False
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret
