#!/bin/bash
for (( i=0; i<5; i++ ))
do
    # source task learning with auxiliary task
    python source_dqn.py --name source_run${i} --episodes 200
    # single task learning without auxiliary task
    python pixel_dqn.py --name single_noreg_run${i} --episodes 300 -no-reg
    # single task learning with auxiliary task
    python pixel_dqn.py --name single_run${i} --episodes 300
    # transfer from source task
    python pixel_dqn.py --name transfer_c1_run${i} --episodes 300 -transfer \
        --load-from learned_models/cartpole/source_run${i}.pt --coeff 1
done