import sys
import pickle
import numpy as np
from collections import namedtuple, deque
import time

from utils.monitor import Monitor
from utils.dqn_core import *
from utils.atari_utils import *
from utils.replay_buffer import *
from utils.schedule import *
from utils.param import Param
#from experiment.envs.test_env import TestEnv
from env.new_lunar import NewLunarLander

import random
import gym
import gym.spaces
gym.logger.set_level(40)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def dqn(
    env_name,
    atari,
    exploration,
    trained_dir=None,
    frame_total=3e+6,
    episode_total=1000,
    replay_buffer_size=1e+5,
    batch_size=256,
    lr=0.0025,
    model_lr=0.0025,
    gamma=0.99,
    tau=1e-3,
    update_freq = None,
    learning_starts=1000,
    learning_freq=4,
    max_steps=10000,
    log_steps=1000,
    exp_name="dqn",
    seed=0,
    doubleQ=False,
    duel=False,
    plot=False,
    prioritized_replay=False,
    beta_schedule=None,
    no_save=False,
    feature_size=8,
    hidden_units=64,
    transfer=False):

    """
    learning_starts: 
        After how many environment steps to start replaying experiences
    learning_freq: 
        How many steps of environment to take between every experience replay
    log_steps:
        How many environment steps to log the progress of the DQN agent
    target_update_freq: 
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    update:
        Whether we use soft or hard update for target Q network.
        Default: soft
    duel, doubleQ:
        Whether we use dueling dqn or double dqn
    """
    dtype = Param.dtype
    device = Param.device
    ram = "ram" in env_name
    exp_name = env_name + "_" + exp_name
    if (env_name == 'VelLunarLander-v2'):
        env = NewLunarLander(obs_type='vel')
    elif (env_name == 'PosLunarLander-v2'):
        env = NewLunarLander(obs_type='pos')
    elif (env_name == 'NoisyLunarLander-v2'):
        env = NewLunarLander(obs_type='noisy')
    elif ram:
        env = gym.make(env_name)
    elif atari == True:
        env = gym.make(env_name)
        env = Monitor(env)
        env = make_env(env, frame_stack=True, scale=False)
#         q_func = model_get('Atari', num_actions = env.action_space.n, duel=duel)
    else:
        raise Exception('Environments not supported')
        
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete
    num_actions = env.action_space.n  
    
    dqn_agent = DQN_Agent(num_states=env.observation_space.shape[0], 
                          num_actions=env.action_space.n, 
                          feature_size=feature_size, 
                          learning_rate=lr, 
                          model_lr=model_lr, 
                          doubleQ=doubleQ, 
                          atari=atari, 
                          ram=ram,
                          update_freq=update_freq,
                          hidden_units=hidden_units)
    print(dqn_agent.Q)
    print(dqn_agent.dynamic_model)
    
    if trained_dir is not None:
        saved_q, saved_model = torch.load(trained_dir, map_location=Param.device)
        dqn_agent.dynamic_model.load_state_dict(saved_model)
        if not transfer:
            dqn_agent.Q.load_state_dict(saved_q)
        for name, pam in dqn_agent.dynamic_model.named_parameters():
            print("loaded", name, pam)
            
    if not prioritized_replay:
        replay_buffer = DQNReplayBuffer(num_actions,replay_buffer_size,batch_size,seed)
    else:
        replay_buffer = DQNPrioritizedBuffer(replay_buffer_size, batch_size=batch_size,seed=seed)
    
    ### Set up logger file
    logger_file = open(os.path.join(Param.data_dir, r"logger_{}.txt".format(exp_name)), "wt")
    
    rew_file = open(os.path.join(Param.data_dir, r"reward_{}.txt".format(exp_name)), "wt")
    
    ### Prepareing for running on the environment
    t, counter = 0, 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    scores = []
    scores_window = deque(maxlen=log_steps)
    model_loss = deque(maxlen=log_steps)
    ep_len_window = deque(maxlen=log_steps)
    time_window = deque(maxlen=log_steps)
    
    while (t<episode_total or counter<frame_total):
        t+=1
        score, steps = 0, 0
        last_obs = env.reset()
        start = time.time()
        while (True):
            if (counter>learning_starts or trained_dir is not None):
                ### Epsilon Greedy Policy
                last_obs_normalized = last_obs/255. if atari else last_obs
                action = dqn_agent.select_epilson_greedy_action(last_obs_normalized, counter, exploration, num_actions)
            else: ### Randomly Select an action before the learning starts
                action = random.randrange(num_actions)
            # Advance one step
            obs, reward, done, info = env.step(action)
            steps += 1
            counter+=1
            score += reward
            
            replay_buffer.add(last_obs, action, reward, obs, done)
            ### Update last observation
            last_obs = obs
            
            if (counter > learning_starts and
                    counter % learning_freq == 0):
                ### sample a batch of transitions from replay buffer and train the Q network
                if not prioritized_replay:
                    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample()
                    not_done_mask = 1 - done_mask
                    obs_batch = obs_batch/255. if atari else obs_batch
                    next_obs_batch = next_obs_batch/255. if atari else next_obs_batch
                    m_loss = dqn_agent.update(obs_batch, act_batch, rew_batch, \
                                     next_obs_batch, not_done_mask, gamma, tau)  
                    model_loss.append(m_loss)
                else:
                    obs_batch, act_batch, rew_batch, next_obs_batch, \
                    done_mask, indices, weights = replay_buffer.sample(beta=beta_schedule.value(counter))
                    obs_batch, next_obs_batch = obs_batch.squeeze(1), next_obs_batch.squeeze(1)
                    not_done_mask = (1 - done_mask).unsqueeze(1)
                    obs_batch = obs_batch/255. if atari else obs_batch
                    next_obs_batch = next_obs_batch/255. if atari else next_obs_batch
                    priority = dqn_agent.update(obs_batch, act_batch, rew_batch, \
                                     next_obs_batch, not_done_mask, gamma, tau, weights)   
                    replay_buffer.update_priorities(indices, priority.cpu().numpy())
            #if atari:
                #if 'episode' in info.keys() or steps>max_steps:
                    #score = info['episode']['r'] if 'episode' in info.keys() else score
                    #ep_len_window.append(steps)
                    #scores_window.append(score)
                    #steps = 0
                    #break
            #else:
            if done or steps>max_steps:
                ep_len_window.append(steps)
                scores_window.append(score)
                steps = 0
                break
            
            if counter % log_steps == 0 and len(scores_window)>1:
                rew_file.write('Step {:.2f}: Reward {:.2f}\n'.format(counter, np.mean(scores_window)))
         
        scores.append(score)
        time_window.append(time.time()-start)
        
        if t % log_steps == 0 and counter > learning_starts:
            print("------------------------------Episode {}------------------------------------".format(t))
            logger_file.write("------------------------------Episode {}------------------------------------\n".format(t))
            print('Num of Interactions with Environment:{:.2f}k'.format(counter/1000))
            logger_file.write('Num of Interactions with Environment:{:.2f}k\n'.format(counter/1000))
            print('Mean Training Reward per episode: {:.2f}'.format(np.mean(scores_window)))
            logger_file.write('Mean Training Reward per episode: {:.2f}\n'.format(np.mean(scores_window)))
            print('Average Episode Length: {:.2f}'.format(np.mean(ep_len_window)))
            logger_file.write('Average Episode Length: {:.2f}\n'.format(np.mean(ep_len_window)))
            print('Average Time: {:.2f}'.format(np.mean(time_window)))
            logger_file.write('Average Time: {:.2f}\n'.format(np.mean(time_window)))
            print('Model Loss: {:.2f}'.format(np.mean(model_loss)))
            logger_file.write('Model Loss: {:.2f}\n'.format(np.mean(model_loss)))
            
            if atari:
                eval_reward = roll_out_atari(dqn_agent, env)
            else:
                eval_reward = rollout(dqn_agent, env)
            print('Eval Reward:{:.2f}'.format(eval_reward))
            logger_file.write('Eval Reward:{:.2f}\n'.format(eval_reward))
            logger_file.flush()
            if not no_save:
                dqn_agent.save(exp_name=exp_name)
        
    logger_file.close()
    
    if plot:
        plt.plot(exponential_smoothing(scores,alpha=0.05))
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.savefig(r'./plot/{}.png'.format(exp_name))
        plt.close()
    return scores
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-no_cuda', action="store_true")
    parser.add_argument('-cuda', '--which_cuda', type=int, default=1)
    parser.add_argument('-no_save', action="store_true")
    
    parser.add_argument('--trained_dir', type=str, default=None)
    parser.add_argument('--episode_total', type=int, default=5000)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--frame_total', type=int, default=3e+6)
    parser.add_argument('-atari',action='store_true')
    parser.add_argument('-transfer',action='store_true')
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--exp_initp', type=float, default=1.0)
    parser.add_argument('--exp_finalp', type=float, default=0.01)
    
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--model_lr', type=float, default=1e-4)
    parser.add_argument('--feature_size', type=int, default=16)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('-weight_decay', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--update_freq', type=int, default=None)
    
    parser.add_argument('-doubleQ', action="store_true")
    parser.add_argument('-duel', action='store_true')
    
    parser.add_argument('--learning_starts', type=int, default=50000)
    parser.add_argument('--learning_freq', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='dqn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-plot', action='store_true')
    
    parser.add_argument('-prioritized_replay', action='store_true')
    parser.add_argument('--beta_start', type=float, default=0.4)
    parser.add_argument('--beta_steps', type=int, default=1e+6)
    parser.add_argument('--exploration_frames', type=int, default=1e+6)
    
    args = parser.parse_args()
    
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.which_cuda)))
    
    if args.prioritized_replay:
        print("Prioritized Experience Replay")
        beta_schedule = PiecewiseSchedule([(0,args.beta_start), (args.frame_total, 1.0)], 
                                                outside_value=1.0)
    else:
        beta_schedule = None
        
    exploration = PiecewiseSchedule([(0, args.exp_initp), (args.exploration_frames, args.exp_finalp)], 
                                    outside_value=args.exp_finalp)
    #exploration = ExponentialSchedule(final_p=args.exp_finalp)
    
    dqn(env_name=args.env,
        atari=args.atari,
        plot=args.plot,
        trained_dir = args.trained_dir,
        exploration=exploration,
        episode_total=args.episode_total,
        frame_total=args.frame_total,
        replay_buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        lr=args.lr,
        model_lr = args.model_lr,
        gamma=args.gamma,
        tau = args.tau,
        learning_starts = args.learning_starts,
        learning_freq=args.learning_freq,
        log_steps=args.log_steps,
        exp_name=args.exp_name,
        seed=args.seed,
        doubleQ=args.doubleQ,
        duel = args.duel, 
        update_freq = args.update_freq,
        prioritized_replay = args.prioritized_replay,
        beta_schedule = beta_schedule,
        max_steps=args.max_steps,
        no_save=args.no_save,
        feature_size=args.feature_size,
        hidden_units=args.hidden_units,
        transfer=args.transfer)
    
        
