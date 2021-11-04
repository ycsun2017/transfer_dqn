#!/bin/bash

# train on source
# python source_dqn.py --env-name cart_save --name source_save --episodes 200 \
#         --head-layers 1 --feature-size 16

for (( i=5; i<20; i++ ))
do
    # pretraining encoder with time alignment
    python pixel_dqn_pretrain.py --env-name cart_pretrain --name pretrain_run${i} \
        -no-reg -detach-next --head-layers 1 --feature-size 16 \
        --load-from ./learned_models/cart_save/source_save.pt

    # used pretrained encoder
    python pixel_dqn.py --env-name cart_pretrain --name learn_run${i} --episodes 200 \
        -no-reg -detach-next --head-layers 1 --feature-size 16 -aligned \
        --load-from ./learned_models/cart_pretrain/pretrain_run${i}.pt
done