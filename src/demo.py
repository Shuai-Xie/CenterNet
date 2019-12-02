from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from pprint import pprint

import os
import cv2

from opts import opts
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # even in default o mode, debug = 1
    opt.debug = max(opt.debug, 1)  # 1: only show the final detection results

    # update opt info with dataset
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    pprint(vars(opt))

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


if __name__ == '__main__':
    # all args define in sh
    opt = opts().parse()  # parse can use cmd args, init() can't
    demo(opt)  # change debug from 0->1
