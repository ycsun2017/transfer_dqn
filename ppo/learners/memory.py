import random
from collections import namedtuple, deque
import numpy as np

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
    def add (self, state, action, logprob, reward, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)


class OPMemory:
    def __init__(self, size=10000):
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        # self.dones = deque(maxlen=size)

        
    def add(self, state, action, next_state, reward, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        if is_terminal:
            self.rewards.append(0.0)
        else:
            self.rewards.append(reward)
        # self.dones.append(is_terminal)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.rewards), size=batch_size)
        states, actions, nexts, rewards = [], [], [], []
        for i in idx:
            states.append(self.states[i])
            actions.append(self.actions[i])
            nexts.append(self.next_states[i])
            rewards.append(self.rewards[i])
            # dones.append(self.dones[i])
        return states, actions, nexts, rewards