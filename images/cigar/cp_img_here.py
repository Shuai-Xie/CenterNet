import shutil
from pycocotools.coco import COCO
from pprint import pprint
import numpy as np
import os

json_file = '../../data/cigar_box/annotations/20_cigar_box_train.json'

coco = COCO(annotation_file=json_file)

num_sample = 10
choose_ids = np.random.choice(coco.getImgIds(), num_sample)  # list

for idx in choose_ids:
    file_name = coco.imgs[idx]['file_name']  # get file name is enough
    shutil.copyfile(src=file_name,  # cp to cur dir
                    dst=os.path.basename(file_name))
