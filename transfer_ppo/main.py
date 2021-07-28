import numpy as np
import torch 
import gym
from gym.spaces import Box, Discrete
import time
import pickle
from transfer_ppo.utils.ppo_core import rollout
from transfer_ppo.learners.ppo import PPO, MovingMeanStd
from transfer_ppo.utils.param import Param
from transfer_ppo.utils.memory import PPOBuffer, ModelReplayBuffer
from transfer_ppo.utils.wrapper import FrameStack, RewardNormalize
from transfer_ppo.envs.cart import NewCartPoleEnv
from transfer_ppo.envs.lunar import NewLunarLanderEnv
import argparse
import os

def main(args):
    torch.autograd.set_detect_anomaly(True)
    ### Setup Torch Device
    if args.device=='cpu':
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device(args.device))
    if args.plotdir is not None:
        Param.plot_dir = args.plotdir
    
    ### Create Environment
    if args.env_name == "CartPole-normal":
        env = NewCartPoleEnv(obs_type="normal")
    elif args.env_name == "CartPole-hard":
        env = NewCartPoleEnv(obs_type="hard")
    elif args.env_name == "LunarLander-normal":
        env = NewLunarLanderEnv()
    elif args.env_name == "LunarLander-noisy":
        env = NewLunarLanderEnv(obs_type="noisy")
    else:
        env = gym.make(args.env_name)
    if args.frame_stack:
        env = FrameStack(env, 4)
    if args.reward_normalize:
        env = RewardNormalize(env)
    ### Set-up Random Seed
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    
    ### Create PPO agent
    agent = PPO(env.observation_space, env.action_space, args.feature_size, 
                hidden_units=args.hidden_units, encoder_layers=args.encoder_layers, model_layers=args.model_layers,
                value_layers=args.value_layers, policy_layers=args.policy_layers, clip_ratio=args.clip_ratio, 
                device=args.device, train_pi_iters=args.train_pi_iters, 
                train_v_iters=args.train_v_iters, train_dynamic_iters=args.train_dynamic_iters, 
                target_kl=args.target_kl, transfer=args.transfer, no_detach=args.no_detach, 
                pi_lr=args.pi_lr, vf_lr=args.vf_lr, model_lr=args.model_lr, 
                coeff=args.coeff, delta=not args.no_delta, obs_only = args.obs_only, 
                env=env, disable_encoder=args.disable_encoder, source_aux=args.source_aux, act_encoder=args.act_encoder)
    
    ### Create Two Buffers
    obs_dim   = env.observation_space.shape
    act_dim   = env.action_space.shape
    cont      = True if isinstance(env.action_space, Box) else False
    memory    = PPOBuffer(obs_dim, act_dim, args.steps_per_epoch, args.gamma, args.lam)
    op_memory = ModelReplayBuffer(obs_dim, act_dim)
    
    ### If it's the target environment, load the pretrained environment model
    if args.transfer:
        agent.ac.load_dynamics(args.load_path)
        if args.pretrain:
            agent.pretrain_env_model(env, num_t=100)
        
    if not args.no_log:
        ### Set up logging path
        model_name = args.env_name + '_' + '{}'.format(args.feature_size)
        model_name += ('_target_{}'.format(args.coeff) if args.transfer else '_source')
        model_name += ('_obs' if args.obs_only else '_both')
        model_name += '_reward_normalize' if args.reward_normalize else ''
        model_path = os.path.join(Param.model_dir, model_name)
        log_path = os.path.join(Param.log_dir, model_name+'.txt')
        logger_file = open(log_path, 'wt')
        data_path = os.path.join(Param.data_dir, model_name+'.pkl')
        data_file = open(data_path, 'wb') ### Store the reward from every epoch

        ### Log Network Info
        logger_file.write('--------------------\nNumber of parameters: \t pi: %d, \t v: %d\n------------------'%agent.var_counts)
        logger_file.write(str(agent))
        logger_file.flush()
    
    epoch_reward, max_eval_reward = [], -np.inf
    o, next_o = env.reset(), None
    ep_ret, ep_len = 0, 0
    
    ### Keep track of moving mean and standard deviation of the feature vector
    latent_vec = agent.ac.pi.encoder(torch.from_numpy(o).to(Param.device).type(Param.dtype))
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(args.epochs):
        episode_reward, episode_len, start_time = [], [], time.time()
        for t in range(args.steps_per_epoch):
            a, v, logp, latent_vec_new = agent.step(torch.from_numpy(o).to(Param.device).type(Param.dtype))
            a = a.cpu().numpy()
            next_o, r, d, _ = env.step(a)
            if not args.reward_normalize:
                ep_ret += r
            else:
                ### If we normalize the reward, add the original unnormalized reward
                ep_ret += env.r
            ep_len += 1

            # Add to buffer
            memory.store(o, a, r, v, logp)
            op_memory.store(o, a, next_o, np.array(r), cont=cont)
            
            # Update obs (critical!)
            o, latent_vec = next_o, latent_vec_new

            timeout = ep_len == args.max_ep_len
            terminal = d or timeout
            epoch_ended = t== args.steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal) and not(args.verbose):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _ = agent.step(torch.from_numpy(o).to(Param.device).type(Param.dtype))
                else:
                    v = torch.tensor(0)
                memory.finish_path(v.item())
                if terminal:
                    episode_reward.append(ep_ret)
                    episode_len.append(ep_len)
                ### Reset the environment
                o, ep_ret, ep_len = env.reset(), 0, 0
        epoch_reward.append(sum(episode_reward)/len(episode_reward))
        
        if not args.obs_only:
            pi_info, v_info, loss_obs, loss_reward = agent.update(memory, op_memory)
        else:
            pi_info, v_info, loss_obs              = agent.update(memory, op_memory)
        
        if not args.verbose and not args.no_log:
            print("----------------------Epoch {}----------------------------".format(epoch))
            logger_file.write("----------------------Epoch {}----------------------------\n".format(epoch))
            print("EpRet:{}".format(sum(episode_reward)/len(episode_reward) ))
            logger_file.write("EpRet:{}\n".format(sum(episode_reward)/len(episode_reward)))
            print("EpLen:{}".format(sum(episode_len)/len(episode_len)))
            logger_file.write("EpLen:{}\n".format(sum(episode_len)/len(episode_len)))
            print('Total Interaction with Environment:{}'.format((epoch+1)*args.steps_per_epoch))
            logger_file.write('Total Interaction with Environment:{}\n'.format((epoch+1)*args.steps_per_epoch))
            print('Loss Policy:{}'.format(pi_info['LossPi']))
            logger_file.write('LossPi:{}\n'.format(pi_info['LossPi']))
            print('Loss V:{}'.format(v_info['LossV']))
            logger_file.write('LossV:{}\n'.format(v_info['LossV']))
            print('Loss Observation:{}'.format(loss_obs))
            logger_file.write('Loss Observation:{}\n'.format(loss_obs))
            if not args.obs_only:
                print('Loss Reward:{}'.format(loss_reward))
                logger_file.write('LossR:{}\n'.format(loss_reward))
            print('V Values:{}'.format(v))
            logger_file.write('V Values:{}\n'.format(v))
            print('DeltaLossPi:{}'.format(pi_info['DeltaLossPi']))
            logger_file.write('DeltaLossPi:{}\n'.format(pi_info['DeltaLossPi']))
            print('DeltaLossV:{}'.format(v_info['DeltaLossV']))
            logger_file.write('DeltaLossV:{}\n'.format(v_info['DeltaLossV']))
            print('Entropy:{}'.format(pi_info['Entropy']))
            logger_file.write('Entropy:{}\n'.format(pi_info['Entropy']))
            print('ClipFrac:{}'.format(pi_info['ClipFrac']))
            logger_file.write('ClipFrac:{}\n'.format(pi_info['ClipFrac']))
            print('KL:{}'.format(pi_info['KL']))
            logger_file.write('KL:{}\n'.format(pi_info['KL']))
            print('StopIter:{}'.format(pi_info['StopIter']))
            logger_file.write('StopIter:{}\n'.format(pi_info['StopIter']))
            print('Time:{}'.format(time.time()-start_time))
            logger_file.write('Time:{}\n'.format(time.time()-start_time))
            
            logger_file.flush()
            
        if epoch% args.log_epochs==0 and epoch>0 and not args.no_log:
            eval_return = rollout(env, agent, num_trajectories = args.eval_trajectories, 
                                  max_steps = args.max_ep_len, reward_normalize=args.reward_normalize)
            print("---------------------Eval Reward:{}-----------------------".format(eval_return))
            logger_file.write("---------------Epoch: {}  Eval Reward:{}--------------\n".format(epoch, eval_return))
            if eval_return > max_eval_reward:
                max_eval_reward = eval_return
                agent.save(model_path)
        
            
            
    if not args.no_log:
        logger_file.close()
        pickle.dump(epoch_reward, data_file)
        data_file.close()
    
    ### The smoothed return is averaged over 20 epochs
    smoothed_return = []
    epoch_smoothed  = 20
    for i in range(args.epochs):
        if (i+epoch_smoothed>args.epochs):
            smoothed_return.append(sum(epoch_reward[i:])/(args.epochs-i))
        else:
            smoothed_return.append(sum(epoch_reward[i:i+epoch_smoothed])/epoch_smoothed)
    n_interactions = [(i*args.steps_per_epoch) for i in range(args.epochs)]
    
    if args.plot:
        import matplotlib.pyplot as plt
        ### While plotting, episode return is smoothed over a window of 10 epochs
        plt.figure(figsize=(8,5))
        plt.plot(n_interactions, epoch_reward)
        plt.xlabel("Number of interactions(1e+6)", fontsize=12)
        plt.ylabel("Average Trajectory Return", fontsize=12)
        plt.xticks(np.arange(0.,4.01,0.5))
        plt.yticks(np.arange(0, 7001,1000))
        plt.savefig('./plot/ppo_{}_{}.png'.format(args.env_name, 'source' if not args.transfer else 'transfer'))
    
    return smoothed_return, n_interactions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--env-name', type=str, default="CartPole-v0")
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
    parser.add_argument('--no-detach', action='store_true', default=False)
    parser.add_argument('--source-aux', action='store_true', default=False)
    parser.add_argument('--pretrain', action='store_true', default=False)

    parser.add_argument('--buffer-size', type=int, default=500000)
    parser.add_argument('--batch-size',  type=int, default=256)
    parser.add_argument('--transfer', action='store_true', default=False)
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--act-encoder', action='store_true', default=False)

    parser.add_argument('--no-log', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)

    # file settings
    parser.add_argument('--logdir', type=str, default="logs/")
    parser.add_argument('--resdir', type=str, default="results/")
    parser.add_argument('--plotdir', type=str, default="plot/")
    parser.add_argument('--moddir', type=str, default="models/")

    ### PPO learning hyperparameters
    parser.add_argument('--vf-lr', type=float, default=1e-3)
    parser.add_argument('--pi-lr', type=float, default=3e-4)
    parser.add_argument('--model-lr', type=float, default=0.001)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--train_pi_iters', type=int, default=80)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--train_dynamic_iters', type=int, default=500)
    args = parser.parse_args()
    main(args)

    