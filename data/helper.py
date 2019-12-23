from pycocotools.coco import COCO
import re
from utils_me.io_utils import load_json, dump_json


# A 156 225
# a 104 127
def see_cat_stats(coco_json='./cigar_box/annotations/20_cigar_box_train.json'):
    coco = COCO(annotation_file=coco_json)
    cats = coco.getCatIds()
    print(cats)
    for cat_id in cats:
        imgs = coco.getImgIds(catIds=[cat_id])
        anns = coco.getAnnIds(catIds=[cat_id])
        print(coco.cats[cat_id], len(imgs), len(anns))


def get_Aa_idxs(cats):
    A_cats, a_cats = [], []
    A_idxs, a_idxs = [], []
    for idx, cat in enumerate(cats):
        if re.match('.+_[A-Z]', cat):
            A_idxs.append(idx + 1)  # from 1
            A_cats.append(cat)
        elif re.match('.+_[a-z]', cat):
            a_idxs.append(idx + 1)
            a_cats.append(cat)
    # print(A_cats)
    # print(A_idxs)
    # print(a_cats)
    # print(a_idxs)
    return A_idxs, A_cats, a_idxs, a_cats


def cvt_json_super(in_json, A_idxs, a_idxs):
    train_dict = load_json(in_json)  # coco format json
    for ann in train_dict['annotations']:
        if ann['category_id'] in A_idxs:
            ann['category_id'] = 1
        elif ann['category_id'] in a_idxs:
            ann['category_id'] = 2
    train_dict['categories'] = [
        {"id": 1, "name": "条装", "supercategory": ""},
        {"id": 2, "name": "包装", "supercategory": ""},
    ]
    dump_json(train_dict, in_json.replace('cigar_box', 'cigar_Aa'))


def cvt_to_Aa(num_class=None):
    Aa_cfg = {
        'name': 'Cigar Aa',
        'cats_num': {'条装': 0, '包装': 0},
        'classes': 2,
        'train': 0,
        'valid': 0,
        'test': 0
    }
    if num_class:
        sub_cfg = load_json(f'./cigar_box/annotations/{num_class}_cigar_box_cfg.json')
        sub_cats = list(sub_cfg['cats_num'].keys())
        A_idxs, A_cats, a_idxs, a_cats = get_Aa_idxs(sub_cats)
        cvt_json_super(f'./cigar_box/annotations/{num_class}_cigar_box_train.json', A_idxs, a_idxs)
        cvt_json_super(f'./cigar_box/annotations/{num_class}_cigar_box_val.json', A_idxs, a_idxs)
        cvt_json_super(f'./cigar_box/annotations/{num_class}_cigar_box_test.json', A_idxs, a_idxs)
    else:
        sub_cfg = load_json('./cigar_box/annotations/cigar_box_cfg.json')
        sub_cats = list(sub_cfg['cats_num'].keys())
        A_idxs, A_cats, a_idxs, a_cats = get_Aa_idxs(sub_cats)
        cvt_json_super('./cigar_box/annotations/cigar_box_train.json', A_idxs, a_idxs)
        cvt_json_super('./cigar_box/annotations/cigar_box_val.json', A_idxs, a_idxs)
        cvt_json_super('./cigar_box/annotations/cigar_box_test.json', A_idxs, a_idxs)

    Aa_cfg['cats_num']['条装'] = sum([sub_cfg['cats_num'][cat] for cat in A_cats])
    Aa_cfg['cats_num']['包装'] = sum([sub_cfg['cats_num'][cat] for cat in a_cats])
    Aa_cfg['train'] = sub_cfg['train']
    Aa_cfg['valid'] = sub_cfg['valid']
    Aa_cfg['test'] = sub_cfg['test']

    cfg_path = './cigar_Aa/annotations/' + (f'{num_class}_cigar_Aa_cfg.json' if num_class else 'cigar_Aa_cfg.json')
    dump_json(Aa_cfg, cfg_path)


if __name__ == '__main__':
    cvt_to_Aa()
    # see_cat_stats(coco_json='cigar_Aa/annotations/20_cigar_Aa_train.json')
    # see_cat_stats(coco_json='cigar_box/annotations/20_cigar_box_train.json')
