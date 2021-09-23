#!/bin/bash
for (( i=0; i<5; i++ ))
do
    ### IN SOURCE
    # source task learning with auxiliary task
    python source_dqn.py --env-name cart_fix --name source_l2_run${i} --episodes 200 \
        --head-layers 1 --feature-size 16
    # source task learning without auxiliary task
    python source_dqn.py --env-name cart_fix --name source_noreg_l2_run${i} --episodes 200 \
        -no-reg --head-layers 1 --feature-size 16
    # source task transfer to itself
    python source_dqn.py --env-name cart_fix --name source_transfer_c1_run${i} --episodes 200 -target \
        --load-from learned_models/cart_fix/source_l2_run0.pt --coeff 1 -detach-next \
        --head-layers 1 --feature-size 16

    ### IN TARGET
    # single task learning without auxiliary task
    python pixel_dqn.py --env-name cart_fix_pixel --name single_noreg_l2_run${i} --episodes 200 \
        -no-reg -detach-next --head-layers 1
    # single task learning with auxiliary task
    python pixel_dqn.py --env-name cart_fix_pixel --name single_l2_run${i} --episodes 200 \
        -detach-next --head-layers 1
    # transfer from source task
    python pixel_dqn.py --env-name cart_fix_pixel --name transfer_l2_c1_run${i} --episodes 200 \
        -transfer -detach-next --coeff 1 --head-layers 1 \
        --load-from learned_models/cart_fix/source_l2_run0.pt
    # learn with randomly initialized model
    python pixel_dqn.py --env-name cart_fix_pixel --name transfer_l2_c1_run${i} --episodes 200 \
        -transfer -detach-next --coeff 1 --head-layers 1 
done