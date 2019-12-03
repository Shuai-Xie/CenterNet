import cv2
import numpy as np


# Yes I Do
# convert float points of image to int points in a image
def pt_float2int(pt, img_w, img_h):
    # x,y
    return max(0, min(round(pt[0] * img_w), img_w - 1)), \
           max(0, min(round(pt[1] * img_h), img_h - 1))


def pt_int2float(pt, img_w, img_h):
    # x,y
    return max(0, min(pt[0] / img_w, 1)), \
           max(0, min(pt[1] / img_h, 1))


# convert normalized pts to (x1,y1,x2,y2)
def cvt_rect_fpts_to_x1y1x2y2(points, img_w, img_h):
    x1, y1 = pt_float2int(points[0], img_w, img_h)  # 左上
    x2, y2 = pt_float2int(points[1], img_w, img_h)  # 右上
    x3, y3 = pt_float2int(points[2], img_w, img_h)  # 右下
    x4, y4 = pt_float2int(points[3], img_w, img_h)  # 右下

    xmin, ymin = min([x1, x2, x3, x4]), min([y1, y2, y3, y4])
    xmax, ymax = max([x1, x2, x3, x4]), max([y1, y2, y3, y4])

    return [xmin, ymin, xmax, ymax]


# convet points of bbox to (x,y,w,h) bbox
def cvt_rect_fpts_to_xywh(points, img_w, img_h):
    xmin, ymin, xmax, ymax = cvt_rect_fpts_to_x1y1x2y2(points, img_w, img_h)
    return [xmin, ymin, xmax - xmin, ymax - ymin]  # [x,y,w,h]


def cvt_rect_ftps_tp_center_xywh(points, img_w, img_h):
    xmin, ymin, xmax, ymax = cvt_rect_fpts_to_x1y1x2y2(points, img_w, img_h)
    return [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]  # [center_x,y,w,h]


def cvt_poly_fpts_to_center_xywh_angle(points, img_w, img_h):
    # cvt float to int
    poly_pts = np.array([pt_float2int(pt, img_w, img_h) for pt in points])
    rect = cv2.minAreaRect(poly_pts)  # ((x,y), (w,h), angle)
    box, angle = rect[0] + rect[1], rect[2]
    return box, angle


# convert (x1,y1,x2,y2) to normalized pts
# 左上角 (x1,y1), 顺时针
def cvt_box_to_rect_fpts(box, img_w, img_h):
    x1, y1 = pt_int2float((box[0], box[1]), img_w, img_h)
    x2, y2 = pt_int2float((box[2], box[1]), img_w, img_h)
    x3, y3 = pt_int2float((box[2], box[3]), img_w, img_h)
    x4, y4 = pt_int2float((box[0], box[3]), img_w, img_h)

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
