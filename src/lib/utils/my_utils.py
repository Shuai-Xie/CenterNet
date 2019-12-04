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
    'DaChongJiu_A', 'YunYan_a', 'JiaoZi_B', 'ZhongHua_B',
    'LiQun_a', 'HuangHeLou_e', 'JiaoZi_F', 'YunYan_A',
    'HuangHeLou_h', 'HuangHeLou_E', 'HuangJinYe_C', '555_a',
    'HongTaShan_b', 'YuXi_A', 'JiaoZi_K', 'HuangHeLou_A',
    'JiaoZi_E', 'TianZi_a', 'TianZi_A', 'WangGuan_A'
]


def plt_heatmaps(hm):
    assert isinstance(hm, torch.Tensor)

    hm = hm.detach().cpu().numpy()
    batch, cat, height, width = hm.shape

    cols = 5
    rows = int(math.ceil(cat / cols))

    classnames = cigar_classnames

    # a batch of imgs
    for img_i in range(batch):
        f, axs = plt.subplots(rows, cols)
        f.set_size_inches((cols * 3, rows * 3))  # 1:1

        for cat_i in range(cat):  # plt heatmap of each cat
            ax = axs.flat[cat_i]
            ax.axis('off')
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.imshow(hm[img_i][cat_i], cmap=plt.get_cmap('jet'))
            ax.set_title(classnames[cat_i])

        # plt.suptitle('img: {}'.format(img_i))  # big title, so high!
        # plt.savefig('img_{}'.format(img_i), bbox_inches='tight', pad_inches=0.0)
        plt.savefig('img_{}'.format(2), bbox_inches='tight', pad_inches=0.0)
