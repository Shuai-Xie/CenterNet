from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from pprint import pprint
import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory  # ModelTrainer class
from models.model import model_layers, model_summary


def main(opt):
    torch.manual_seed(opt.seed)
    # if add --not_cuda_benchmark, opt.not_cuda_benchmark=True
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    # return Dataset class by dataset and task name
    # one dataset can do multiple tasks by different annotation settings
    Dataset = get_dataset(opt.dataset, opt.task)
    # update opt [ input|ouput res, opt.heads ] with Dataset info
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    pprint(vars(opt))

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    # opt.arch: --arch dla_34
    # opt.heads: set heads by task in opts().update_dataset_info_and_set_heads()
    # opt.head_conv: 256, one more layer btw features and final_class, number defined by opt.arch
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)  # optimize all params
    start_epoch = 0

    # load pretrain model
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    # choose trainer by opt.task
    Trainer = train_factory[opt.task]
    # define trainer
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')

    # val dataset
    val_loader = torch.utils.data.DataLoader(Dataset(opt, 'val'),
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True)
    if opt.test:  # test on val dataset
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return  # end program here

    train_loader = torch.utils.data.DataLoader(Dataset(opt, 'train'),  # split, load json
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers,  # multi-process read data, wrt batch_size
                                               pin_memory=True,
                                               drop_last=True)

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'  # save all middle model or last
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            # default will USE_TENSORBOARD to log scalars
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        # default val/save intervals = 5
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),  # path
                       epoch, model, optimizer)  # save model dict keys
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                # metric:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
