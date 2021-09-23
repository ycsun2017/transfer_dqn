#!/bin/bash
for (( i=0; i<2; i++ ))
do
    python source_dqn.py --name source_noreg_detach_run${i} --episodes 200 -no-reg -detach-next
    python source_dqn.py --name source_detach_run${i} --episodes 200 -detach-next
    python source_dqn.py --name source_transfer_detach_c1_run${i} --episodes 200 -target \
        --load-from learned_models/cartpole/source_detach_run${i}.pt --coeff 1 -detach-next
    python source_dqn.py --name source_transfer_fromnoreg_detach_c1_run${i} --episodes 200 -target \
        --load-from learned_models/cartpole/source_noreg_detach_run${i}.pt --coeff 1 -detach-next 
done