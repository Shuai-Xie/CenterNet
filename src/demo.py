from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from pprint import pprint

import os
import cv2
import time

from opts import opts
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from utils.debugger import Debugger

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
img_id = 'video'


# test on video
def draw_res_box(debugger, img, results, num_classes, vis_thresh):
    debugger.add_img(img, img_id=img_id)
    # draw results on image
    for j in range(1, num_classes + 1):
        for bbox in results[j]:
            if bbox[4] > vis_thresh:
                debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id=img_id)
    return debugger.imgs[img_id]


def process_img(img_path, detector):
    """
    use default debugger in detector.run()
    :param img_path:
    :param detector:
    :return:
    """
    # process one img a time
    ret = detector.run(img_path)
    time_str = ''
    for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    print(time_str)


def process_video(video_path, detector, debugger, num_classes):
    """
    use self-defined debugger in draw_res_box()
    :param video_path:
    :param detector:
    :param debugger: used to add_coco_bbox
    :param num_classes: dataset classes
    :return:
    """
    cam = cv2.VideoCapture(video_path)
    video_w, video_h, fps = int(cam.get(3)), int(cam.get(4)), int(cam.get(5))
    total_frames = int(cam.get(7))
    # write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_res = cv2.VideoWriter(video_path.replace('cigar_videos', 'cigar_videos_res'), fourcc,
                             fps, (video_w, video_h))
    cnt = 0
    detector.pause = False
    begin_time = time.time()
    while True:
        ok, img = cam.read()
        if not ok:
            break
        # cv2.imshow('input', img)
        ret = detector.run(img)
        res_img = draw_res_box(debugger, img, ret['results'],
                               num_classes, opt.vis_thresh)  # opt in main, so can use
        vw_res.write(res_img)
        cnt += 1
        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print('\r{}/{}'.format(cnt, total_frames), time_str, end='')
        # if cv2.waitKey(1) == 27:
        #     return  # esc to quit
    print()
    total_time = time.time() - begin_time
    print('total time:', total_time)
    print('fps:', total_frames / total_time)
    vw_res.release()
    cam.release()


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # even in default o mode, debug = 1
    # opt.debug = max(opt.debug, 1)  # 1: only show the final detection results

    # update opt info with dataset
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    pprint(vars(opt))

    Detector = detector_factory[opt.task]  # choose a detector
    detector = Detector(opt)

    debugger = Debugger(
        dataset=opt.dataset,
        ipynb=False,
        theme=opt.debugger_theme
    )  # debugger.imgs is a dict

    image_names, video_names = [], []
    if os.path.isdir(opt.demo):  # image dir or video drr
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()  # img ext
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
            if ext in video_ext:
                video_names.append(os.path.join(opt.demo, file_name))

    elif opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:  # one video
        video_names = [opt.demo]
    elif opt.demo[opt.demo.rfind('.') + 1:].lower() in image_ext:  # one image
        image_names = [opt.demo]

    elif opt.demo == 'webcam':  # cam
        pass

    for image_name in image_names:
        print(image_name)
        process_img(image_name, detector)

    opt.debug = 0  # when test video, use draw_res_box

    for video_name in video_names:
        print(video_name)
        process_video(video_name, detector, debugger, Dataset.num_classes)


if __name__ == '__main__':
    # all args define in sh
    opt = opts().parse()  # parse can use cmd args, init() can't
    demo(opt)  # change debug from 0->1
