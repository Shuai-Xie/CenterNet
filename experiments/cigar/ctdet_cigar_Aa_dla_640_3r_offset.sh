#!/usr/bin/env bash
cd src

# train
# if add --load_model, must add --resume, begin epoch is still 1
python main.py ctdet_offset --exp_id cigar_Aa_dla_640_3r_offset --arch dla_34 \
--dataset cigar_Aa --batch_size 16 --lr 1e-4 --num_epochs 70 --lr_step 45,60 \
--input_h 352 --input_w 640 \
--gpus 0 --num_workers 4

# test
# --gpus default '0'
# keep_res will get very low acc!
python test.py ctdet --exp_id cigar_Aa_dla_640_3r --arch dla_34 \
--dataset cigar_Aa --resume --use best --gpus 0 \
--input_h 352 --input_w 640 \
--flip_test \
--keep_res

# demo
# img dir
python demo.py ctdet --exp_id cigar_Aa_dla_640_3r --arch dla_34 \
--dataset cigar_Aa --resume --use best --gpus 0 \
--input_h 352 --input_w 640 \
--demo ../images/cigar \
--debug 2 \
--flip_test \
--keep_res

# video dir
python demo.py ctdet --exp_id cigar_Aa_dla_640_3r --arch dla_34 \
--dataset cigar_Aa --resume --use best --gpus 0 \
--input_h 352 --input_w 640 \
--demo ../images/cigar_videos

# video
python demo.py ctdet --exp_id cigar_Aa_dla_640_3r --arch dla_34 \
--dataset cigar_Aa --resume --use best --gpus 0 \
--input_h 352 --input_w 640 \
--demo ../images/cigar_videos/20191216-154731_comp_Trim.mp4

# img
python demo.py ctdet --exp_id cigar_Aa_dla_640_3r --arch dla_34 \
--dataset cigar_Aa --resume --use best --gpus 0 \
--input_h 352 --input_w 640 \
--demo ../images/cigar/adaa8f79-fac0-4144-8bc2-ecc1273af72b___vlcsnap-2019-12-20-16h06m49s132.png \
--debug 2 \
--flip_test

cd ..