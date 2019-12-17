#!/usr/bin/env bash
cd src

# train
# if add --load_model, must add --resume, begin epoch is still 1
python main.py ctdet --exp_id cigar_Aa_dla_640_3r --arch dla_34 \
--dataset cigar_Aa --batch_size 16 --lr 1e-4 --num_epochs 40 --lr_step 25,35 \
--K 10 \
--input_h 352 --input_w 640 \
--gpus 0 --num_workers 4

# test
# --gpus default '0'
# keep_res will get very low acc!
python test.py ctdet --exp_id cigar_Aa_dla_640 --arch dla_34 \
--dataset cigar_Aa --resume --use best --gpus 0 \
--input_h 352 --input_w 640 \
--keep_res

# demo
python demo.py ctdet --exp_id cigar_Aa_dla_640 --arch dla_34 \
--dataset cigar_Aa --load_model ../exp/ctdet/cigar_Aa_dla_640/model_best.pth \
--demo ../images/cigar --debug 2 --gpus 0 \
--input_h 352 --input_w 640 \
--keep_res

##################################

# half r
python main.py ctdet --exp_id cigar_Aa_dla_640_0.5r --arch dla_34 \
--dataset cigar_Aa --batch_size 16 --lr 1e-4 --num_epochs 30 --lr_step 15,25 \
--input_h 352 --input_w 640 \
--gpus 0 --num_workers 4


python test.py ctdet --exp_id cigar_Aa_dla_640_0.5r --arch dla_34 \
--dataset cigar_Aa --resume --use last --gpus 0 \
--input_h 352 --input_w 640 \
--keep_res

# demo
python demo.py ctdet --exp_id cigar_Aa_dla_640_0.5r --arch dla_34 \
--dataset cigar_Aa --resume --use last --gpus 0 \
--input_h 352 --input_w 640 \
--demo ../images/cigar --debug 2 \
--keep_res

##################################

# flip test
python test.py ctdet --exp_id cigar_Aa_dla_640 --arch dla_34 \
--dataset cigar_Aa --resume --use last --gpus 0 \
--input_h 352 --input_w 640 \
--flip_test \
--keep_res


# multi scale test
python test.py ctdet --exp_id cigar_Aa_dla_640 --arch dla_34 \
--dataset cigar_Aa --resume --use last --gpus 0 \
--input_h 352 --input_w 640 \
--flip_test --test_scales 0.5,0.75,1,1.25,1.5 \
--keep_res




python demo.py ctdet --exp_id cigar_Aa_dla_640 --arch dla_34 \
--dataset cigar_Aa --load_model ../exp/ctdet/cigar_dla_640/model_last.pth \
--demo ../images/cigar/6ce746ea-72a1-4b2a-9f10-7b249e856838___2019-09-19_191844_cloud_video_1907.jpg \
--debug 2 --gpus 1 \
--input_h 352 --input_w 640 \
--flip_test

cd ..