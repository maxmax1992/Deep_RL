# from pong import Pong
import matplotlib

matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from random import randint
# import pickle
import numpy as np
# from simple_ai import PongAi, MyAi
# # import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
# from PIL import Image
# from skimage.transform import resize

np.set_printoptions(threshold=np.nan)
import collections
import gym

torch.set_default_tensor_type('torch.cuda.DoubleTensor')
# CUDA
use_cuda = torch.cuda.is_available()
print("Using cuda:", use_cuda)


class CriticNN(nn.Module):
    def __init__(self, lr):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        print('self params', self.parameters)


    def forward(self, x):
        x = torch.from_numpy(x).cuda()
        # x = F.layer_norm(x, x.size())
        x = F.leaky_relu(self.fc1(x))
        # x = F.layer_norm(x, x.size())
        x = self.fc2(x)
        # print('x', x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            # print('HERE ---')lr
            # torch.nn.init.xavier_uniform(m.weight)
            m.weight.data.fill_(1e-3)
            m.bias.data.fill_(1e-3)

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ActorNN(nn.Module):
    def __init__(self, lr):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        # print(x.size())
        # print('x', x)
        # x = F.layer_norm(x, x.size())
        x = F.leaky_relu(self.fc1(x))
        # x = F.layer_norm(x, x.size())
        x = torch.sigmoid(self.fc2(x))
        return x

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



# class ReplayMemory(object):
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#
#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)


actor = ActorNN(lr=1e-4)
critic = CriticNN(lr=1e-4)
critic.apply(critic.init_weights)
env = gym.make('CartPole-v0')
gamma = 0.99

# Compute discounted reward for reward rollouts
def discount_rewards(r):
    len = len(r)
    return_arr = np.zeros(l)
    i = len - 1
    while i > 0:

  return discounted_r


def finish_episode():
    pass


def select_action(obs):
    left_prob = actor.forward(torch.from_numpy(obs).cuda())
    action = 0 if np.random.uniform() < left_prob else 1
    lprob = torch.log(left_prob) if action == 0 else torch.log(1 - left_prob)
    return lprob, action

log_freq = 100
running_rewards = collections.deque(maxlen=log_freq)

batch_size = 20
for i_episode in range(1, 10000):
    done = False
    observation_prev = env.reset()
    ep_reward = 0
    frame = 1
    while not done:
        # env.render()
        lprob, action = select_action(observation_prev)
        observation, reward, done, _ = env.step(action)
        # Append the V(s)s
        advantage = critic.forward(torch.from_numpy(observation_prev).cuda())
        loss = torch.pow(ep_reward - advantage, 2)
        critic.train(loss)
        actor.train( - log_freq * (ep_reward - advantage).item())
        ep_reward += reward

        if done or batch_size % frame == 0:



        frame += 1




    if i_episode % log_freq == 0:
        print(f"Episode: {i_episode}, last {log_freq} episodes mean reward: { np.mean(running_rewards)}")
    running_rewards.append(ep_reward)

