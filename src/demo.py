from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
from utils.debugger import Debugger

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # even in default o mode, debug = 1
    opt.debug = max(opt.debug, 1)  # 1: only show the final detection results
    Detector = detector_factory[opt.task]  # choose a detector
    detector = Detector(opt)

    if opt.demo == 'webcam' or opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                # ext = file_name.split('.')[-1].lower()
                ext = file_name[file_name.rfind('.') + 1:].lower()  # img ext
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]  # one img

        for image_name in image_names:
            # process one img a time
            ret = detector.run(image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)


from pprint import pprint

if __name__ == '__main__':
    resdcn18_args = [
        'ctdet',  # detector
        '--exp_id', 'coco_resdcn18',  # experiment, wrt trained model
        '--keep_res',  # keep ori img resolution
        '--load_model', '../models/ctdet_coco_resdcn18.pth',
        '--gpus', '0',
        '--arch', 'resdcn_18',  # num_layer = 18
        '--demo', '../images',  # path to demo images
        # '--cat_spec_wh',  # not use class-aware
        # '--head_conv', '-1',  # 64, default setting, heat_conv is ok
        # '--heads' # heads {'hm': 80, 'wh': 2, 'reg': 2}, has set in opts.update_dataset_info_and_set_heads()
    ]
    dla34_args = [
        'ctdet',  # detector
        '--exp_id', 'coco_dla_2x_demo',
        '--keep_res',  # keep ori img resolution
        '--load_model', '../models/ctdet_coco_dla_2x.pth',
        '--gpus', '0',
        '--arch', 'dla_34',
        '--demo', '../images',
    ]
    opt = opts().init(resdcn18_args)
    pprint(vars(opt))
    exit(0)
    demo(opt)  # change debug from 0->1
