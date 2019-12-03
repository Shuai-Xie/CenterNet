#!/usr/bin/env bash
cd src
# train todo: add input_h, input_w
# if add --load_model, must add resume can the epoches is ok
python main.py ctdet --exp_id cigar_dla_1x --arch dla_34 \
--dataset cigar --batch_size 16 --lr 1e-4 --num_epochs 70 --lr_step 45,60 \
--gpus 0 --num_workers 2 --save_all

# test
# --gpus default '0'
# keep_res will get very low acc!
python test.py ctdet --exp_id cigar_dla_1x --arch dla_34 \
--dataset cigar --keep_res --resume --use last --gpus 0

# flip test
python test.py ctdet --exp_id cigar_dla_1x --arch dla_34 \
--dataset cigar --keep_res --resume --flip_test --use last --gpus 0

# multi scale test
python test.py ctdet --exp_id cigar_dla_1x --arch dla_34 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5 --gpus 0

# demo
python demo.py ctdet --exp_id cigar_dla_1x --arch dla_34 \
--dataset cigar --load_model ../exp/ctdet/cigar_dla_1x/model_last.pth \
--demo ../images/cigar --debug 2 --gpus 0 \
--keep_res

python demo.py ctdet --exp_id cigar_dla_1x --arch dla_34 \
--dataset cigar --load_model ../exp/ctdet/cigar_dla_1x/model_last.pth \
--demo ../images/cigar/db534d21-8b6d-4921-86a3-b678632e1cd3___vlcsnap-2019-11-21-18h59m09s722.png \
--debug 2 --gpus 0


cd ..