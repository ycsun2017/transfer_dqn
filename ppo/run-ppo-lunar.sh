#! /bin/bash

CUDA=$1
ENV="LunarLander-normal"
TENV="LunarLander-noisy"
N=3000000

for FS in 1000
do
    for USE in both
    do
#         python main.py --env ${ENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --steps ${N} --lr 5e-4 > out/${ENV}_single_${FS}_${USE}
#         python main.py --env ${TENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --steps ${N} --lr 5e-4 > out/${TENV}_single_${FS}_${USE}
        for COEFF in 2 4
        do
            python main.py --env ${TENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --transfer --source ${ENV}_ppo_n${N}_f${FS}_lr0.0005_${USE}_single --steps ${N} --coeff-decay --coeff ${COEFF} --no-detach --exp-name "_nd"  --lr 5e-4 > out/${TENV}_transfer_${FS}_${USE}_c${COEFF}_nd
        done
    done
done
