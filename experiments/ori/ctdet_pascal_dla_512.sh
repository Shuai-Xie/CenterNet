#!/usr/bin/env bash
cd src
# train
python main.py ctdet --exp_id pascal_dla_512 --dataset pascal --input_res 512 --num_epochs 70 --lr_step 45,60 --gpus 0,1

# test
python test.py ctdet --exp_id pascal_dla_512 --dataset pascal --input_res 512 --resume
# flip test
python test.py ctdet --exp_id pascal_dla_512 --dataset pascal --input_res 512 --resume --flip_test

# demo
python demo.py ctdet --exp_id coco_dla_2x_demo --arch dla_34 --keep_res --load_model ../models/ctdet_coco_dla_2x.pth --demo ../images --debug 2 --gpus 0

cd ..
