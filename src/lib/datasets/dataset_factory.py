from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.cigar import Cigar  # add cigar
from .dataset.cigar_Aa import Cigar_Aa
from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.ctdet_offset import CTDetOffset_Dataset
from .sample.multi_pose import MultiPoseDataset

dataset_factory = {
    'cigar': Cigar,
    'cigar_Aa': Cigar_Aa,
    'coco': COCO,
    'pascal': PascalVOC,
    'kitti': KITTI,
    'coco_hp': COCOHP
}

# fetch image from dataset no change, but change the annotation
# sample annotation <-> task
_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ctdet_offset': CTDetOffset_Dataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
    # match dataset and task
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
