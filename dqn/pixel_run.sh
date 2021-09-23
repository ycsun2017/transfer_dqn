#!/bin/bash
for (( i=5; i<10; i++ ))
do
    # source task learning with auxiliary task
    # python source_dqn.py --name source_detach_run${i} --episodes 200 --lr 1e-3
    # single task learning without auxiliary task
    python pixel_dqn.py --name single_noreg_detach_run${i} --episodes 200 -no-reg -detach-next
    # # single task learning with auxiliary task
    python pixel_dqn.py --name single_detach_run${i} --episodes 200 -detach-next
    # # transfer from source task
    python pixel_dqn.py --name transfer_detach_c1_run${i} --episodes 200 -transfer -detach-next \
        --load-from learned_models/cartpole/source_detach_run0.pt --coeff 1
    # python pixel_dqn.py --name transfer_c10decay_run${i} --episodes 200 -transfer -decay-coeff \
    #     --load-from learned_models/cartpole/source_lr3e4_run4.pt --coeff 10
done