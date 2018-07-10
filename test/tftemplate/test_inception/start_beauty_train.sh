#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python MainEntryTrainBeauty.py \
    --task=train_beauty \
    --is_training=True \
    --max_iter=50000 \
    --batch_size=32 \
    --learning_rate=0.00008 \
    --decay_step=1000 \
    --decay_rate=0.9 \
    --dropout=0.85 \
    --input_dir=/alexwang/data/beauty \
    --save_model_dir=/alexwang/workspace/beauty_model_2 \
    --save_model_freq=500 \
    --print_info_freq=100 \
    --valid_freq=500 \
    --summary_dir=/alexwang/workspace/beauty_summary_2 \
    --checkpoint=/alexwang/data/inception_v4.ckpt