#!/bin/sh

emb_dim=128
train_itrs=100000
train_batch_size=32
logdir=runs/pretrain/emb_${emb_dim}
device="cuda:0"

python pud/pretrain_encoder.py \
    --logdir $logdir \
    --device $device \
    --train_itrs $train_itrs \
    --train_batch_size $train_batch_size \
    --emb_dim $emb_dim