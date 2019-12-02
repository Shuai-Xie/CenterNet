#!/usr/bin/env bash
cd src
# train todo: add input_h, input_w
python main.py ctdet --exp_id cigar_resdcn50 --arch resdcn_50 \
--dataset cigar --batch_size 16 --lr 1e-4 --num_epochs 70 --lr_step 45,60 \
--gpus 0 --num_workers 2 --save_all

# test
# exp_id has define the dir where we find model_last.pth
python test.py ctdet --exp_id cigar_resdcn50 --arch resdcn_50 \
--dataset cigar --keep_res --resume --use best --gpus 0

# flip test
python test.py ctdet --exp_id cigar_resdcn50 --arch resdcn_50 --keep_res --resume --flip_test --gpus 0
# multi scale test
python test.py ctdet --exp_id cigar_resdcn50 --arch resdcn_50 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5 --gpus 0

# demo
python demo.py ctdet --exp_id coco_hg_demo --arch hourglass --keep_res --load_model ../models/ctdet_coco_hg.pth --demo ../images --debug 2 --gpus 0

cd ..