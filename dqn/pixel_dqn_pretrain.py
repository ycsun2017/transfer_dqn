'''
Pre-train an encoder based on time-alignment
'''

import os
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--env-name', type=str, default="cart_pretrain")
parser.add_argument('--name', type=str, default="pretrain")
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--feature-size', type=int, default=16)
parser.add_argument('--hiddens', type=int, default=32)
parser.add_argument('--head-layers', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--coeff', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('-transfer', action="store_true")
parser.add_argument('-no-reg', action="store_true")
parser.add_argument('-detach-next', action="store_true")
parser.add_argument('-decay-coeff', action="store_true")
parser.add_argument('--load-from', type=str, default="")
args = parser.parse_args()

env = gym.make(args.env).unwrapped

save_path = "data/{}/".format(args.env_name)
os.makedirs(save_path, exist_ok=True)
os.makedirs("learned_models/{}/".format(args.env_name), exist_ok=True)

env.seed(args.seed)
# random.seed(args.seed)
# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

# if gpu is to be used
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

RepPair = namedtuple('RepPair',
                        ('state', 'repre'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a pair of state and old representation"""
        self.memory.append(RepPair(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def build_mlp(
        input_size,
        output_size,
        n_layers,
        size,
        activation = nn.ReLU(),
        output_activation = nn.Identity(),
        init_method=None,
        norm=False
):
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)
    if norm:
        layers.append(L2Norm())
        
    return nn.Sequential(*layers)

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)

class ActionDynamicModel(nn.Module):
    '''Only for discrete action space'''
    def __init__(self, feature_size, num_actions, hiddens):
        super().__init__()

        self.h_size = hiddens
        self.num_actions = num_actions
        self.enc_size = feature_size
        
        self.transitions = build_mlp(self.enc_size, self.enc_size*self.num_actions, 0, self.h_size, norm=False)
        self.rewards = build_mlp(self.enc_size, self.num_actions, 0, self.h_size, norm=False)
    
    def forward(self, encoding, actions):
        dist_actions = actions.flatten() #torch.LongTensor(actions).to(device)
        predict_next = self.transitions(encoding)
        predict_reward = self.rewards(encoding)
        
        ind_starts = dist_actions * self.enc_size
        ind_ends = ind_starts + self.enc_size
        indices = torch.stack([torch.arange(ind_starts[i], ind_ends[i]) for i in range(ind_starts.size()[0])]).to(device)
        predict_next = predict_next.gather(1, indices)
        predict_reward = predict_reward.gather(1, dist_actions.view(-1,1))
        
        return predict_next, predict_reward

class Encoder(nn.Module):
    def __init__(self, h, w, feature_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.output = nn.Linear(linear_input_size, feature_size)
        self.norm = L2Norm()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.norm(self.output(x.view(x.size(0), -1)))

class DQN(nn.Module):

    def __init__(self, h, w, outputs, feature_size, hiddens, head_layers=2):
        super(DQN, self).__init__()
        self.encoder = Encoder(h, w, feature_size)
        self.head = build_mlp(feature_size, outputs, head_layers, hiddens)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = self.encoder(x)
        return self.head(x)

class SDQN(nn.Module):

    def __init__(self, inputs, outputs, hiddens, feature_size, head_layers=2):
        super(SDQN, self).__init__()

        self.encoder = build_mlp(inputs, feature_size, 2, hiddens, norm=True)
        self.head = build_mlp(feature_size, outputs, head_layers, hiddens)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = self.encoder(x)
        x = self.head(x)
        return x

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    if args.env == "CartPole-v0":
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions, feature_size=args.feature_size, 
                    hiddens=args.hiddens, head_layers=args.head_layers).to(device)
target_net = DQN(screen_height, screen_width, n_actions, feature_size=args.feature_size, 
                    hiddens=args.hiddens, head_layers=args.head_layers).to(device)
print("policy", policy_net)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

dynamic_model = ActionDynamicModel(feature_size=args.feature_size, num_actions=n_actions, hiddens=64).to(device)
print("dynamics", dynamic_model)

optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)


n_actions = env.action_space.n
state_size = env.observation_space.shape[0]
source_policy = SDQN(state_size, n_actions, hiddens=args.hiddens, feature_size=args.feature_size, 
                head_layers=args.head_layers).to(device)
if args.load_from:
    source_policy.encoder.load_state_dict(torch.load(args.load_from, map_location=device)["encoder"])
    source_policy.head.load_state_dict(torch.load(args.load_from, map_location=device)["head"])

memory = ReplayMemory(10000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def model_loss(state_batch, next_batch, action_batch, reward_batch, target=False):
    criterion = nn.SmoothL1Loss()
    if target:
        encodings = target_net.encoder(state_batch).detach()
        next_encodings = target_net.encoder(next_batch).detach()
    else:
        encodings = policy_net.encoder(state_batch)
        next_encodings = policy_net.encoder(next_batch)
        if args.detach_next:
            next_encodings = next_encodings.detach()
    predict_next, predict_reward = dynamic_model(encodings, action_batch)
    model_loss = criterion(predict_next, next_encodings) + criterion(predict_reward.flatten(), reward_batch)

    return model_loss

def optimize_model(epi):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = RepPair(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    repre_batch = torch.cat(batch.repre)
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    encode_batch = policy_net.encoder(state_batch)
    # print("encode", encode_batch)
    loss = criterion(encode_batch, repre_batch.detach())
    # print("loss", loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.encoder.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()


num_episodes = args.episodes
total_rewards = []
mean_loss = []

with open('trajs/source.pkl', 'rb') as f:
    trajs = pickle.load(f)
print("loaded", len(trajs), "trajectories")

for i_episode in range(len(trajs)):
    traj = trajs[i_episode]
    print("fitting", i_episode, "with", len(traj["states"]), len(traj["actions"]))
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    eps_reward = 0
    for t in range(len(traj["actions"])):
        # store the corresponding state-repre pair
        repre = source_policy.encoder(traj["states"][t].to(device))
        memory.push(state, repre)

        # take the same actions as the old trajectory does
        _, reward, done, _ = env.step(traj["actions"][t])
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        
        eps_reward += reward.item()

        # Move to the next state
        state = next_state

        # Perform one step of the optimization for the encoder (time-based alignment)
        mloss = optimize_model(i_episode) 
        print(t, end=",")
        if done or t > 200:
            episode_durations.append(t + 1)
            # plot_durations()
            break

    print("episode", i_episode, "reward", eps_reward, "loss", mloss)
    total_rewards.append(eps_reward)
    mean_loss.append(mloss)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.plot(total_rewards)
plt.savefig("data/{}/{}.png".format(args.env_name, args.name), format="png")
plt.close()
with open("data/{}/{}.txt".format(args.env_name, args.name), "w") as f:
    for i, reward in enumerate(total_rewards):
        f.write("Episode: {}, Reward: {}\n".format(i, reward))

torch.save({
        # "dynamics": dynamic_model.state_dict(), 
        "encoder": policy_net.encoder.state_dict(),
        "head": source_policy.head.state_dict()
    },
    "learned_models/{}/{}.pt".format(args.env_name, args.name)
)

env.close()
# plt.ioff()
# plt.show()
