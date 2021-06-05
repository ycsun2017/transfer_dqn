### Roll out trajectories and get the return
### gamma: discount factor
### num_t: number of trajectories
### max_steps: maximum steps to take in each trajectory
### sa_mdp: (default=False) If sa_mdp is set to be True, it will print its Mean/Max Perturbation Norm
import torch
from experiment.utils.param import Param
import numpy as np

def test_return(env, agent, gamma, num_t, max_steps, sa_mdp=True):
    avg_return = 0
    for i in range(num_t):
        obs, done = env.reset(), False
        r = 0
        a_norm = []
        for t in range (max_steps):
            if sa_mdp:
                obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype)
                a = agent.act(obs).cpu().numpy()
                a_norm.append(np.max(a))
            else:
                a = int(agent.step(obs))
            next_o, reward, done, _ = env.step(a)
            obs = next_o
            if sa_mdp:
                r -= gamma*(reward)
            else:
                r += gamma*(reward)
            if done:
                break
        avg_return += r/num_t
        # if sa_mdp:
        #     print("Average Perturbed Infinity norm:{}".format(sum(a_norm)/len(a_norm)))
        #     print("Maximum Perturbed Infinity norm:{}".format(max(a_norm)))
    return avg_return