#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python MainEntryTrainBeauty.py \
    --task=train_beauty_class \
    --is_training=True \
    --max_iter=200000 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --decay_step=10000 \
    --decay_rate=0.9 \
    --dropout=0.5 \
    --input_dir=/u02/alexwang/data/beauty \
    --save_model_dir=/alexwang/workspace/beauty_model_1 \
    --save_model_freq=1000 \
    --print_info_freq=100 \
    --valid_freq=1000 \
    --summary_dir=/alexwang/workspace/beauty_summary_1 \
    --checkpoint=/alexwang/data/inception_v4.ckpt

# tensorboard --logdir=/alexwang/workspace/beauty_summary --port=10013