### To run experiment with new linear model
In source task: \n
  `python main.py --env-name HalfCheetah-v3 --device cuda:0 --encoder-layers 2 --feature-size 64 --policy-layers 1 --act-encoder --epochs 1000 --log-epochs 50 &`\n
So every 50 epochs, this will store the model in learned_models/ \n

In target task:\n
  `python main.py --env-name HalfCheetahTest-v3 --device cuda:0 --transfer --encoder-layers 2 --feature-size 64 --policy-layers 0 --act-encoder --load-path ./learned_models/HalfCheetah-v3_64_source_both --epochs 1000 --log-epochs 50 &  `
