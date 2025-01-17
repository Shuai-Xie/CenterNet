from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        # create, load model in BaseDetector, with opt args
        # self.opt in BaseDetector
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        """
        :param images: images tensor
        :param return_time: whether return forward time
        :return: if flip_test, dim0 = 2
            output = {
                'hm': [1, 20, 88, 160] (B, C, out_h, out_w)
                'wh': [1, 2, 88, 160]
                'reg': [1, 2, 88, 160]
            }
            dets: [B,K,6]
        """
        with torch.no_grad():
            # forward, define in lib.models.networks
            output = self.model(images)[-1]  # forward return [ret]
            # print(type(output))  # dict
            # for k, v in output.items():
            #     print(k, v.size())
            hm = output['hm'].sigmoid_()  # in-place sigmoid, activation fn
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None  # center offset

            if self.opt.flip_test:  # get mean
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2  # dim0=2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None  # not like wh

            torch.cuda.synchronize()
            forward_time = time.time()
            # [B,K,6] box,score,cls
            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            # print(dets)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        """
        recover result to ori img size with meta
        """
        dets = dets.detach().cpu().numpy()  # [B,K,6] box,score,cls
        dets = dets.reshape(1, -1, dets.shape[2])  # [1,B*K,6]
        dets = ctdet_post_process(  # affine transform
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    # Note: debug and show_results have 2 different thresh
    # debug: opt.center_thresh = 0.1
    # show_results: opt.vis_thresh = 0.3

    def debug(self, debugger, images, dets, output, scale=1, img_name=None):
        detection = dets.detach().cpu().numpy().copy()
        # scale to 4x [input_size], but not to ori img size
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # h,w,3
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)  # recover to 255 img
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            blend_img_id = '{}_pred_hm_{:.1f}'.format(img_name, scale) if img_name else 'pred_hm_{:.1f}'.format(scale)
            out_img_id = '{}_out_pred_{:.1f}'.format(img_name, scale) if img_name else 'out_pred_{:.1f}'.format(scale)
            # add: blend_img, out_img
            debugger.add_blend_img(back=img, fore=pred, img_id=blend_img_id)
            debugger.add_img(img, img_id=out_img_id)
            for k in range(len(dets[i])):  # i:img, k:box, 4 score
                if detection[i, k, 4] > self.opt.center_thresh:  # center_thresh=0.1
                    debugger.add_coco_bbox(bbox=detection[i, k, :4],
                                           cat=detection[i, k, -1],
                                           conf=detection[i, k, 4],
                                           img_id=out_img_id)
            # add: save all
            debugger.save_img(imgId=blend_img_id, path=self.opt.debug_dir)
            debugger.save_img(imgId=out_img_id, path=self.opt.debug_dir)

    def show_results(self, debugger, image, results, img_name=None):
        img_id = img_name if img_name else 'ctdet'  # todo: set img_id with img_name
        debugger.add_img(image, img_id=img_id)
        # draw results on image
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id=img_id)
        # change show to save
        # debugger.show_all_imgs(pause=self.pause)  # not show
        debugger.save_img(imgId=img_id, path=self.opt.debug_dir)
