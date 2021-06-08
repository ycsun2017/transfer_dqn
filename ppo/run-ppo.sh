#! /bin/bash

CUDA=$1
ENV="LunarLander-v2"
TENV="LunarLander-v2"
N=1000000

for FS in 1000
do
    for USE in both
    do
        python main.py --env ${ENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --steps ${N} > out/${ENV}_single_${FS}_${USE}
#         python main.py --env ${TENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} > out/${TENV}_single_${FS}_${USE}
        python main.py --env ${TENV} --learner ppo --feature-size ${FS} --device cuda:${CUDA} --use ${USE} --transfer --source ${ENV} --coeff-decay --steps ${N} > out/${TENV}_transfer_${FS}_${USE}
    done
done
