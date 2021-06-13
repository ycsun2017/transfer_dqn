#! /bin/bash

CUDA=$1
ENV="CartPole-normal"
TENV="CartPole-hard"
N=300000

for FS in 256
do
    for USE in both
    do
#         python main.py --env ${ENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --steps ${N} > out/${ENV}_single_${FS}_${USE}
#         python main.py --env ${TENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --steps ${N} > out/${TENV}_single_${FS}_${USE}
        for COEFF in 2
        do
            python main.py --env ${TENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --transfer --source ${ENV}_ppo_n${N}_f${FS}_${USE}_single --steps ${N} --coeff-decay --coeff ${COEFF} --no-detach --exp-name "_nd" > out/${TENV}_transfer_${FS}_${USE}_c${COEFF}_nd
        done
    done
done
