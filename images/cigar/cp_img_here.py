import shutil
from pycocotools.coco import COCO
from pprint import pprint
import numpy as np
import os


def random_cp_imgs(num_sample=10):
    json_file = '../../data/cigar_box/annotations/20_cigar_box_train.json'

    coco = COCO(annotation_file=json_file)

    choose_ids = np.random.choice(coco.getImgIds(), num_sample)  # list

    for idx in choose_ids:
        file_name = coco.imgs[idx]['file_name']  # get file name is enough
        shutil.copyfile(src=file_name,  # cp to cur dir
                        dst=os.path.basename(file_name))


if __name__ == '__main__':
    random_cp_imgs()
