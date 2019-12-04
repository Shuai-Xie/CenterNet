from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger


class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')  # can be multi gpu
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)  # load pretrain
        self.model = self.model.to(opt.device)
        self.model.eval()  # set BN, Dropout constant

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = opt.K  # 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

    def pre_process(self, image, scale, meta=None):
        """
        :param image: ndarray img
        :param scale: scale
        :param meta:
        :returns
            images: tensor, maybe with filp image
            meta: {'c', 's', 'out_height', 'out_width'} to recover img to ori
        """
        height, width = image.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)  # affine_transform center
            s = max(height, width) * 1.0
        else:
            # opt.pad = 127 if 'hourglass' in opt.arch else 31
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)  # affine_transform center
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))  # by scale
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # tensor shape
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {
            'c': c,
            's': s,
            'out_height': inp_height // self.opt.down_ratio,
            'out_width': inp_width // self.opt.down_ratio
        }
        return images, meta

    # abstract func will implement in real detectors
    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1, img_name=None):
        raise NotImplementedError

    def show_results(self, debugger, image, results, img_name=None):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(
            dataset=self.opt.dataset,
            ipynb=(self.opt.debug == 3),
            theme=self.opt.debugger_theme  # white
        )
        start_time = time.time()
        pre_processed = False
        img_name = None
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif isinstance(image_or_path_or_tensor, str):  # img name
            image = cv2.imread(image_or_path_or_tensor)
            img_name = os.path.basename(image_or_path_or_tensor)
        else:  # pre_processed_images
            image = image_or_path_or_tensor['image'][0].numpy()  # prefetch_test, PrefetchDataset
            # print(image.shape)  # (1080, 1920, 3), ori img size
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:  # default [1.0]
            scale_start_time = time.time()

            # pre_process input images
            if not pre_processed:
                # totensor, affine meta params
                images, meta = self.pre_process(image, scale, meta)
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {
                    k: v.numpy()[0] for k, v in meta.items()
                }

            images = images.to(self.opt.device)  # tensor to device

            torch.cuda.synchronize()  # sync and cal time
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            # model forward
            #
            output, dets, forward_time = self.process(images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 2:
                # debug img, add pred_blend_img
                self.debug(debugger, images, dets, output, scale, img_name)

            # affine transform
            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        # merge_outputs, use soft nms
        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            # pass one more param: img_name, to save img
            self.show_results(debugger, image, results, img_name)

        return {
            'results': results,
            'tot': tot_time,
            'load': load_time,
            'pre': pre_time,
            'net': net_time,
            'dec': dec_time,
            'post': post_time,
            'merge': merge_time
        }
