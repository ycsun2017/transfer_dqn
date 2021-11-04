### An implementation of Soft-Actor Critic Algorithms adapted from openai spinup 
### https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
### This file containing the training process of the soft actor critic algorithms


import os
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from utils.sac_core import SACReplayBuffer
import utils.sac_core as core
from utils.param import Param
from utils.dynamic_model import ProbabilisticTransitionModel
from utils.schedule import *


def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=128, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=50, no_save=False, exp_name='sac', no_log=False, 
        feature_size=256, env_linear=False, transfer=False, no_reg=False, load_dir=None,
        coeff=None, pretrain=True, train_dynamics=False, source_dir=None, update_env_freq=0):
    """
    Soft Actor-Critic (SAC)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================
            Calling ``pi`` should return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        lr (float): Learning rate (used for both policy and value learning).
        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.
        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if not no_log:
        ### Set-up logger file
        logger_file = open(os.path.join(Param.data_dir, r"logger_{}.txt".format(exp_name)), "wt")
    
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(Param.device)
    ac_targ = deepcopy(ac)
    
    # Experience buffer
    replay_buffer = SACReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    
    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    if not no_log:
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    else:
        logger_file.write('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    
    ### Define the environment transition model
    transition_model = ProbabilisticTransitionModel(feature_size, env.action_space.shape,
                                                    env_linear=env_linear).to(Param.device)
    if source_dir is not None:
        print("Load Trained Action Encoder and Policy Net")
        ac.load(source_dir)
    
    if transfer:
        assert load_dir is not None, "No loading directory"
        print("Load Trained Dynamics Model")
        state_dict = torch.load(load_dir, map_location=Param.device)
        transition_model.load_state_dict(state_dict["model"])
        for p in transition_model.parameters():
            p.requires_grad = False
        for p in ac.a_encoder.parameters():
            p.requires_grad = False
        encoders = (ac.q1.s_encoder, ac.a_encoder)
        if pretrain:
            loss_info = core.pretrain_encoder(env, replay_buffer, encoders, transition_model, num_t=20, epochs=1000)
            ac.q2.s_encoder.load_state_dict(ac.q1.s_encoder.state_dict())
            ac.pi.s_encoder.load_state_dict(ac.q1.s_encoder.state_dict())
            
            logger_file.write("-------------------Pre-Train Encoder-------------------\n")
            logger_file.write('Loss P:{}\n'.format(loss_info['Loss P']))
            logger_file.write('Loss R:{}\n'.format(loss_info['Loss R']))
            print("----------------Pre-Train Encoder----------------")
            print('Loss P:{}\n'.format(loss_info['Loss P']))
            print('Loss R:{}\n'.format(loss_info['Loss R']))
    
    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)


    def update(data, coeff):
        
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        if not no_reg:
            if not transfer:
                ### Freeze Transition Dynamics while taking the derivative w.r.t encoder
                for p in transition_model.parameters():
                    p.requires_grad = False
            s_encoder = lambda x: (ac.q1.s_encoder(x) + ac.q2.s_encoder(x))/2
            encoders = (s_encoder, ac.a_encoder)
            loss_model, model_loss_info = core.compute_model_loss(replay_buffer, transition_model, encoders)
            ### Release the gradient flow if it's not the transfer setting
            if not transfer:
                for p in transition_model.parameters():
                    p.requires_grad = True
                    
            ### Add model loss into the final loss of the Q network update
            loss = loss_q + coeff*loss_model
        else:
            loss = loss_q
        loss.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        
        loss_info = dict(LossQ=loss_q.item(), LossPi=loss_pi.item())  
        
        if not no_reg:
            return {**loss_info, **pi_info, **q_info, **model_loss_info}
        else:
            return {**loss_info, **pi_info, **q_info}

    def get_action(o, deterministic=False):
        return ac.act(torch.from_numpy(o).to(Param.device).type(Param.dtype), 
                      deterministic)

    def test_agent():
        total_ep_ret, total_ep_len = [], []
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            total_ep_ret.append(ep_ret)
            total_ep_len.append(ep_len)
        return sum(total_ep_ret)/len(total_ep_ret), sum(total_ep_len)/len(total_ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    total_ep_ret, total_ep_len = [], []
    
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            total_ep_ret.append(ep_ret)
            total_ep_len.append(ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                logger_info = update(data=batch, coeff=coeff.value((t+1) // steps_per_epoch))
                
           
        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if ((epoch % save_freq == 0) or (epoch == epochs)):
                ac.save(model_name='./{}.pt'.format(exp_name))

            # Test the performance of the deterministic version of the agent.
            test_ep_ret, test_ep_len = test_agent()

            if not no_log:
                # Log info about epoch
                print('----------------------Epoch:{}---------------------'.format(epoch))
                print('Total Interaction with Environment:{}'.format(t))
                print('Mean Episode Return:{}'.format(sum(total_ep_ret)/len(total_ep_ret)))
                print('Mean Test Return:{}'.format(test_ep_ret))
                print('Mean Episode Len:{}'.format(sum(total_ep_len)/len(total_ep_len)))
                print('Mean Test Episode Len:{}'.format(test_ep_len))
                print('TotalEnvInteracts:{}'.format(t))
                #print('Q1Vals:{}\n'.format(logger_info['Q1Vals']))
                #print('Q2Vals:{}\n'.format(logger_info['Q2Vals']))
                #print('LogPi:{}\n'.format(logger_info['LogPi']))
                print('LossPi:{}'.format(logger_info['LossPi']))
                print('LossQ:{}'.format(logger_info['LossQ']))
                if not no_reg:
                    print('Loss P:{}'.format(logger_info['Loss P']))
                    print('Loss R:{}'.format(logger_info['Loss R']))
                print('Time:{}'.format(time.time()-start_time))

                # Log info about epoch
                logger_file.write('----------------------Epoch:{}---------------------\n'.format(epoch))
                logger_file.write('Total Interaction with Environment:{}\n'.format(t))
                logger_file.write('Mean Episode Return:{}\n'.format(sum(total_ep_ret)/len(total_ep_ret)))
                logger_file.write('Mean Test Return:{}\n'.format(test_ep_ret))
                logger_file.write('Mean Episode Len:{}\n'.format(sum(total_ep_len)/len(total_ep_len)))
                logger_file.write('Mean Test Episode Len:{}\n'.format(test_ep_len))
                logger_file.write('TotalEnvInteracts:{}\n'.format(t+1))
                #logger_file.write('Q1Vals:{}\n'.format(logger_info['Q1Vals']))
                #logger_file.write('Q2Vals:{}\n'.format(logger_info['Q2Vals']))
                #logger_file.write('LogPi:{}\n'.format(logger_info['LogPi']))
                logger_file.write('LossPi:{}\n'.format(logger_info['LossPi']))
                logger_file.write('LossQ:{}\n'.format(logger_info['LossQ']))
                if not no_reg:
                    logger_file.write('Loss P:{}\n'.format(logger_info['Loss P']))
                    logger_file.write('Loss R:{}\n'.format(logger_info['Loss R']))
                logger_file.write('Time:{}\n'.format(time.time()-start_time))
                logger_file.flush()
            
            ### Update environment model if necessary
            if update_env_freq > 0:
                if (epoch > 100) and ((epoch % update_env_freq) == 0):
                    s_encoder = lambda x: (ac.q1.s_encoder(x) + ac.q2.s_encoder(x))/2
                    encoders = (s_encoder, ac.a_encoder)
                    for p in ac.q1.s_encoder.parameters():
                        p.requires_grad = False
                    for p in ac.q2.s_encoder.parameters():
                        p.requires_grad = False
                    if not transfer:
                        for p in ac.a_encoder.parameters():
                            p.requires_grad = False
                    core.update_model(replay_buffer, transition_model, encoders, batch_size=256, epochs=300)
                    for p in ac.q1.s_encoder.parameters():
                        p.requires_grad = True
                    for p in ac.q2.s_encoder.parameters():
                        p.requires_grad = True
                    if not transfer:
                        for p in ac.a_encoder.parameters():
                            p.requires_grad = True
    
    if train_dynamics:
        transition_model = ProbabilisticTransitionModel(feature_size, env.action_space.shape, 
                                                        env_linear=env_linear).to(Param.device)
        encoders = (ac.q1.s_encoder, ac.a_encoder)
        loss_info = core.update_model(replay_buffer, transition_model, encoders, batch_size=256, epochs=1000)
        logger_file.write("----------------Train Dynamic Model----------------\n")
        logger_file.write('Loss P:{}\n'.format(loss_info['Loss P']))
        logger_file.write('Loss R:{}\n'.format(loss_info['Loss R']))
        print("----------------Train Dynamic Model----------------")
        print('Loss P:{}\n'.format(loss_info['Loss P']))
        print('Loss R:{}\n'.format(loss_info['Loss R']))
        state_dict = {
            "model": transition_model.state_dict(),
            "a_encoder": ac.a_encoder.state_dict()
        }
        torch.save(state_dict, './data/{}.pt'.format(exp_name))
    
    if not no_log:   
        logger_file.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_save', action="store_true")
    parser.add_argument('--no_log', action="store_true")
    parser.add_argument('--no_cuda', action="store_true")
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default=None)
    
    parser.add_argument('--train-dynamics', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--env-linear', action='store_true')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--no-reg', action='store_true')
    parser.add_argument('--load-dir', type=str, default=None)
    parser.add_argument('--source-dir', type=str, default=None)
    parser.add_argument('--update-env-freq', type=int, default=0)
    parser.add_argument('--coeff', type=float, default=1.0)
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.cuda)))
    
    if args.transfer:
        coeff_schedule = PiecewiseSchedule([(0,args.coeff), (args.epochs, 0.)], outside_value=0.)
    elif args.update_env_freq>0:
        schedule = PiecewiseConstantSchedule([(0,0.),(100,args.coeff), (args.epochs,args.coeff)])
    else:
        coeff_schedule = ConstantSchedule(args.coeff)
    
    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, 
        no_save=args.no_save, exp_name=args.exp_name if args.exp_name is not None else 'SAC_{}'.format(args.env), 
        no_log=args.no_log, feature_size=args.hid, 
        env_linear=args.env_linear, transfer=args.transfer,
        no_reg=args.no_reg, load_dir=args.load_dir, coeff=coeff_schedule, 
        pretrain=args.pretrain, train_dynamics=args.train_dynamics, 
        source_dir=args.source_dir, update_env_freq=args.update_env_freq)