import argparse
import numpy as np
from transfer_ppo.main import main
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch
import pandas as pd
import os
import seaborn as sns

def process_seed(cuda_n, env_name, seed, statistics, lock):
    print("Process ID:{}, GPU ID:{}".format(os.getpid(), cuda_n), flush=True)
    args = get_args(cuda_n, seed, env_name, transfer=False)
    smooth_return, n_interactions = main(args)
    lock.acquire()
    statistics['smooth_return'] = statistics['smooth_return']+smooth_return
    statistics['n_interactions'] = statistics['n_interactions']+n_interactions
    statistics['seed'] = statistics['seed'] + [seed]*len(smooth_return)
    statistics['task'] = statistics['task']+['source']*len(smooth_return)
    lock.release()
    
    args = get_args(cuda_n, seed, env_name, transfer=True)
    smooth_return, n_interactions = main(args)
    lock.acquire()
    statistics['smooth_return'] = statistics['smooth_return']+smooth_return
    statistics['n_interactions'] = statistics['n_interactions']+n_interactions
    statistics['seed'] = statistics['seed'] + [seed]*len(smooth_return)
    statistics['task'] = statistics['task']+['target_{}'.format(args.coeff)]*len(smooth_return)
    lock.release()

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default="HalfCheetah")
    parser.add_argument('--n-seed',   type=int, default=8)
    args = parser.parse_args()
    
    device_count = torch.cuda.device_count()
    seeds = np.random.randint(0,1000,[args.n_seed,])
    
    processes = []
    mp.set_start_method('spawn')
    lock = mp.Lock() 
    ret_dict = mp.Manager().dict(smooth_return=[], seed=[], n_interactions=[], task=[])
    
    for i in range(args.n_seed):
        process = mp.Process(target=process_seed, args=(i%device_count, args.env_name,
                                                        int(seeds[i]), ret_dict, lock))
        processes.append(process)
        process.start()
    for i in range(args.n_seed):
        processes[i].join()
    statistics = pd.DataFrame(dict(n_interactions=ret_dict['n_interactions'], smooth_return=ret_dict['smooth_return'],
                                   seed=ret_dict['seed'], task=ret_dict['task']))
    
    statistics.to_pickle('./data/{}_linear.pkl'.format(args.env_name))
    plt.figure(figsize=(8,5))
    ax = sns.lineplot(data=statistics, x='n_interactions', y='smooth_return', hue='task')
    plt.xlabel("Number of interactions", fontsize=12)
    plt.ylabel("Average Trajectory Return", fontsize=12)
    #plt.xticks([0,0.3,0.6,0.9,1.2,1.5])
    #plt.xticks([0,0.2,0.4,0.6,0.8,1.0])
    #plt.yticks([200,150,100,50,0,-50,-100])
    plt.savefig('./plot/{}_linear.png'.format(args.env_name))
    
    
    
def get_args(cuda_id, seed, name, transfer):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--env-name', type=str, default="Hopper")
    parser.add_argument('--frame-stack', action='store_true', default=False)
    parser.add_argument('--reward-normalize', action='store_true', default=False)
    parser.add_argument('--exp-name', type=str, default="")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--max-ep-len', type=int, default=1000)
    parser.add_argument('--log-epochs', type=int, default=50)
    parser.add_argument('--steps-per-epoch', type=int, default=4000)
    parser.add_argument('--eval-trajectories', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--coeff', type=float, default=5)

    parser.add_argument('--disable-encoder', action='store_true', default=False)
    parser.add_argument('--feature-size', type=int, default=64)
    parser.add_argument('--hidden-units', type=int, default=64)
    parser.add_argument('--encoder-layers', type=int, default=2)
    parser.add_argument('--policy-layers', type=int, default=0)
    parser.add_argument('--model-layers',  type=int, default=1)
    parser.add_argument('--value-layers', type=int, default=2)
    parser.add_argument('--no-delta', action='store_true', default=False, 
                        help='If no-delta is False, the transition model predict s_{t+1}-s{t} instead of s_{t+1}')
    parser.add_argument('--obs-only', action='store_true', default=False,
                        help='If obs-only is True, then we will only have the transition dynamics and no reward model.')

    parser.add_argument('--buffer-size', type=int, default=500000)
    parser.add_argument('--batch-size',  type=int, default=256)
    parser.add_argument('--transfer', action='store_true', default=False)
    parser.add_argument('--load-path', type=str, default=None)

    parser.add_argument('--no-detach', action='store_true', default=False)
    parser.add_argument('--source-aux', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--act-encoder', action='store_true', default=False)

    # file settings
    parser.add_argument('--no-log', action='store_true', default=False)
    parser.add_argument('--logdir', type=str, default="logs/")
    parser.add_argument('--resdir', type=str, default="results/")
    parser.add_argument('--plotdir', type=str, default="plot/")
    parser.add_argument('--moddir', type=str, default="models/")

    ### PPO learning hyperparameters
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--model_lr', type=float, default=1e-4)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--train_pi_iters', type=int, default=80)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--train_dynamic_iters', type=int, default=400)
    
    feature_size_dict = {"Swimmer": 8, "Hopper": 11, "HalfCheetah": 17, "Walker2d": 17}
    feature_size = feature_size_dict[name]
    env_name = name + 'Test-v3'
    
    if not transfer:
        ### Substitute this if you need to change the command-line argument
        args = parser.parse_args(["--device", "cuda:{}".format(cuda_id), "--env-name", 
                                 "{}".format(env_name), "--disable-encoder", 
                                 "--encoder-layers", "2", "--feature-size", "64",
                                  "--policy-layers", "1", "--act-encoder",
                                 "--seed", "{}".format(seed), "--no-log", "--verbose"])
    else:
        args = parser.parse_args(["--device", "cuda:{}".format(cuda_id), "--env-name", 
                            "{}".format(env_name), "--transfer", "--load-path", 
                            "./learned_models/{}-v3_64_source_both".format(name), 
                            "--encoder-layers", "2", "--policy-layers", "1",
                            "--feature-size", "64", "--seed", "{}".format(seed), 
                            "--no-log", "--verbose", "--pretrain", "--act-encoder"])
                              
    return args
