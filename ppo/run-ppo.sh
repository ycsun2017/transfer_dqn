#! /bin/bash

CUDA=$1
ENV="LunarLander-v2"
TENV="LunarLander-v2"
N=5000000

for FS in 1000
do
    for USE in both
    do
#         python main.py --env ${ENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --steps ${N} > out/${ENV}_single_${FS}_${USE}
#         python main.py --env ${TENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} > out/${TENV}_single_${FS}_${USE}
        for COEFF in 1 2 3 4 5
        do
            python main.py --env ${TENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --transfer --source ${ENV}_ppo_n${N}_f${FS}_${USE}_single --steps ${N} --coeff-decay --coeff ${COEFF} --lr 5e-4 > out/${TENV}_transfer_${FS}_${USE}_c${COEFF}
        done
    done
done
