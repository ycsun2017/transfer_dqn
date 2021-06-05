#! /bin/bash

CUDA=0
ENV="LunarLander-v2"
N=3000

for FS in 8 16 32 48 64
do
# 	python main.py --env ${ENV} --feature-size ${FS} --episodes ${N} --exp-name _single_${FS} --device cuda:${CUDA} > out/single_${FS}
	python main.py --env ${ENV} --feature-size ${FS} --episodes ${N} --exp-name _transfer_${FS} --device cuda:${CUDA} --transfer --loadfile ${ENV}_vpg_n${N}_f${FS}_single_${FS} > out/transfer_${FS}
done
