import torch
import argparse
import gym
from gym.spaces import Box, Discrete
import numpy as np
import setup
from learners.memory import Memory, OPMemory 
from learners.agents.vpg import VPG
from learners.agents.ppo import PPO
from envs.cart import NewCartPoleEnv

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cpu")

# env settings
parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--exp-name', type=str, default="")
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--steps', type=int, default=300)

parser.add_argument('--log-interval', type=int, default=10)

# learner settings
parser.add_argument('--learner', type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--feature-size', type=int, default=16)
parser.add_argument('--transfer', action='store_true', default=False)
parser.add_argument('--use', type=str, default="both", help="both, actor, critic")
# file settings
parser.add_argument('--logdir', type=str, default="logs/")
parser.add_argument('--resdir', type=str, default="results/")
parser.add_argument('--moddir', type=str, default="models/")
parser.add_argument('--loadfile', type=str, default="")

args = parser.parse_args()

def get_log(file_name):
    logger = logging.getLogger('train') 
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(file_name, mode='a') 
    fh.setLevel(logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    return logger


if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env #"LunarLander-v2"
    
    max_episodes = args.episodes        # max training episodes
    max_steps = args.steps         # max timesteps in one episode
    lr = args.lr
    device = args.device
    ############ For All #########################
    gamma = 0.99                # discount factor
    random_seed = 0 
    render = False
    update_every = 300
    save_every = 100
    ############ For PPO #########################
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    
    # creating environment
    if env_name == "CartPole-normal":
        env = NewCartPoleEnv(obs_type="normal")
    elif env_name == "CartPole-hard":
        env = NewCartPoleEnv(obs_type="hard")
    else:
        env = gym.make(env_name)
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    filename = env_name + "_" + args.learner + "_n" + str(max_episodes) + \
        "_f" + str(args.feature_size) 
    
    if args.learner == "ppo":
        filename += "_" + args.use

    if args.transfer:
        filename += "_transfer"
    else:
        filename += "_single"

    logger = get_log(args.logdir + filename)
    logger.info(args)
    
    rew_file = open(args.resdir + filename + ".txt", "w")

    if args.learner == "vpg":
        policy_net = VPG(env.observation_space, env.action_space, args.feature_size, gamma=gamma, 
                    device=device, learning_rate=lr, transfer=args.transfer)
    elif args.learner == "ppo":
        policy_net = PPO(env.observation_space, env.action_space, args.feature_size, gamma=gamma, 
                    device=device, learning_rate=lr, transfer=args.transfer, use_model=args.use)
    
    
    start_episode = 0
    # load learner from checkpoint
    if args.loadfile != "":
        policy_net.load_models(args.moddir + args.loadfile)
    if args.transfer:
        policy_net.load_dynamics(args.moddir + filename.replace("transfer", "single"))
#         policy_net.load_dynamics(args.moddir + args.loadfile)
        print("loaded from", args.moddir + filename.replace("transfer", "single"))

    
    memory = Memory()
    op_memory = OPMemory()
    
    all_rewards = []
    timestep = 0
    
    for episode in range(start_episode, max_episodes):
        state = env.reset()
        rewards = []
        for steps in range(max_steps):
            timestep += 1
            
            if render:
                env.render()
                
            state_tensor, action_tensor, log_prob_tensor = policy_net.act(state)
            
            if isinstance(env.action_space, Discrete):
                action = action_tensor.item()
            else:
                action = action_tensor.cpu().data.numpy().flatten()
            new_state, reward, done, _ = env.step(action)
            
            rewards.append(reward)
            
            memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
            new_state_tensor = torch.from_numpy(new_state).float().to(device) 
            op_memory.add(state_tensor, action_tensor, new_state_tensor, reward, done)

            if done or steps == max_steps-1:
                policy_loss, model_loss = policy_net.update_policy(memory, op_memory)
                logger.info("ploss: {}, mloss: {}\n".format(policy_loss, model_loss))
                memory.clear_memory()
                timestep = 0

            state = new_state
            
            if done:
                all_rewards.append(np.sum(rewards))
                if episode % args.log_interval == 0:
                    print("episode: {}, total reward: {}".format(episode, np.round(np.sum(rewards), decimals = 3)))
                
                rew_file.write("episode: {}, total reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                break

        if (episode+1) % save_every == 0:
            path = args.moddir + filename
            policy_net.save_models(path)

    for name, param in policy_net.dynamic_model.named_parameters():
        print(name, param)

    rew_file.close()

    print("mean:", np.mean(all_rewards), "std", np.std(all_rewards))
           
    env.close()