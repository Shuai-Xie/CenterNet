from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchsummary import summary

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net

"""
1.create base keypoint based models
2.load model from checkpoint
3.save model (convert data_parallal to model)
"""

# base model backbone factory
_model_factory = {
    'res': get_pose_net,  # default Resnet with deconv
    'dlav0': get_dlav0,  # default DLAup
    'dla': get_dla_dcn,
    'resdcn': get_pose_net_dcn,
    'hourglass': get_large_hourglass_net,
}


def model_layers(model):
    print('{:>4} {:<50} {:<30} {}'.format('idx', 'param', 'size', 'grad'))
    for idx, (name, param) in enumerate(model.named_parameters()):  # recurse=True, as net itself is a big module
        print('{:>4} {:<50} {:<30} {}'.format(idx, name, str(param.size()), param.requires_grad))


def model_summary(model, input_size):
    summary(model, input_size, device='cpu')


def create_model(arch, heads, head_conv):
    """
    all params are set in opt
    :param arch: resdcn_18
        'res_18 | res_101 | resdcn_18 | resdcn_101 |'
        'dlav0_34 | dla_34 | hourglass'
    :param heads: {'hm': 80, 'wh': 2, 'reg': 2}, 80 on coco
    :param head_conv: 64, if set>0, do one more conv in head, not directly conv to result
    """
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0  # 18, find return val + list
    arch = arch[:arch.find('_')] if '_' in arch else arch  # dla
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers,  # total layers of a model, layer = CBR, POOL
                      heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path,
               optimizer=None, resume=False,
               lr=None, lr_step=None):
    # 1.load ckpt
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))  # resdcn18, epoch=140, 1x; 2x, epoch=230

    # 2.judge model and ckpt state_dict, so that ckpt can be loaded

    # from checkpoint
    state_dict_ = checkpoint['state_dict']  # just an OrderedDict
    state_dict = {}
    # convert data_parallal to model! a way to deal with data_parallal weights
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]  # k[7:] remove module.
        else:
            state_dict[k] = state_dict_[k]

    # from defined model
    model_state_dict = model.state_dict()

    msg = 'msg'
    # msg = 'If you see this, your model does not fully load the ' + \
    #       'pre-trained weight. Please make sure ' + \
    #       'you have correctly specified --arch xxx ' + \
    #       'or set the correct --num_classes for your own dataset.'

    # check loaded parameters and created model parameters
    # judge whether: model_state_dict = ckpt state_dict
    for k in state_dict:
        if k in model_state_dict:
            # if ckpt has this k, but shape different, maybe different num_classes
            if state_dict[k].shape != model_state_dict[k].shape:  # may be shape diff
                print('Skip loading parameter {}, required shape {}, loaded shape {}. {}'.
                      format(k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            # different task, miss heads keys
            print('Drop parameter {}.'.format(k) + msg)

    for k in model_state_dict:
        if k not in state_dict:  # reverse of last else, complement state_dict so that it can be load!
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]

    # model_layers(model)

    # load ckpt done!
    model.load_state_dict(state_dict, strict=False)

    # 3.resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:  # if ckpt has saved optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']  # resume epoch
            start_lr = lr
            for step in lr_step:  # traverse each lr_step, get the last start_lr
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr  # set new lr for optimizer
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        start_epoch = 0
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()  # convert data_parallal to model
    else:
        state_dict = model.state_dict()
    data = {
        'epoch': epoch,
        'state_dict': state_dict
    }
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
