import matplotlib.pyplot as plt
import torch
import math
import numpy as np


def cat_conf_point_stats(hm, cat_idx=1):
    """
    points statistics of the cat_idx class
    """
    a = hm[0][cat_idx].detach().cpu().numpy()  # yunyan_a hm
    print('mean:', np.mean(a))
    print('max:', np.max(a))
    for i in range(11):
        thre = i / 10
        print('P > {:.1f}: '.format(thre), len(np.where(a > thre)[0]))


cigar_classnames = [
    'DaZhongJiu_A', 'YunYan_a', 'JiaoZi_B', 'ZhongHua_B', 'LiQun_a', 'HuangHeLou_e', 'YunYan_A', 'JiaoZi_F', 'HuangHeLou_h', 'HuangHeLou_E',
    'HuangJinYe_C', '555_a', 'HongTaShan_b', 'YuXi_A', 'HuangGuoShu_a', 'JiaoZi_K', 'HuangHeLou_A', 'JiaoZi_E', 'TianZi_a', 'TianZi_A'
]


def plt_heatmaps(hm, basename='heatmap'):
    if isinstance(hm, torch.Tensor):
        hm = hm.detach().cpu().numpy()
    elif isinstance(hm, np.ndarray) and len(hm.shape) == 3:
        hm = np.expand_dims(hm, axis=0)  # add batch=1

    batch, cat, height, width = hm.shape

    cols = 5
    rows = int(math.ceil(cat / cols))

    classnames = cigar_classnames

    # a batch of imgs
    for img_i in range(batch):
        f, axs = plt.subplots(rows, cols)
        # f.set_size_inches((cols * 3, rows * 3))  # 1:1 todo
        f.set_size_inches((cols * 4, rows * 2))  # 1:1 todo

        for cat_i in range(cat):  # plt heatmap of each cat
            ax = axs.flat[cat_i]
            ax.axis('off')
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.imshow(hm[img_i][cat_i], cmap=plt.get_cmap('jet'))
            ax.set_title(classnames[cat_i])

        # plt.suptitle('img: {}'.format(img_i))  # big title, so high!
        img_name = '{}_{}.png'.format(basename, img_i)
        plt.savefig(img_name, bbox_inches='tight', pad_inches=0.0)
        print('save', img_name)
