#!/usr/bin/env bash
cd src
# train todo: add input_h, input_w
# if add --load_model, must add resume can the epoches is ok
python main.py ctdet --exp_id cigar_dla_1x --arch dla_34 \
--dataset cigar --batch_size 16 --lr 1e-4 --num_epochs 70 --lr_step 45,60 \
--gpus 0 --num_workers 2 --save_all

# test
# --gpus default '0'
python test.py ctdet --exp_id cigar_dla_1x --arch dla_34 \
--dataset cigar --batch_size 1 --keep_res --resume --use best --gpus 0

# flip test
python test.py ctdet --exp_id cigar_resdcn50_test --arch resdcn_50 --keep_res --resume --flip_test --gpus 0
# multi scale test
python test.py ctdet --exp_id cigar_resdcn50_test --arch resdcn_50 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5 --gpus 0

# demo
python demo.py ctdet --exp_id coco_hg_demo --arch hourglass --keep_res --load_model ../models/ctdet_coco_hg.pth --demo ../images --debug 2 --gpus 0

cd ..