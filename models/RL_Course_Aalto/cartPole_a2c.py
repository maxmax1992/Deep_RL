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

    def train(self, loss, C):
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.parameters():
            p.grad *= C  # or whatever other operation
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

    def train(self, loss, C):
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.parameters():
            p.grad *= C  # or whatever other operation
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


actor = ActorNN(lr=1e-3)
critic = CriticNN(lr=1e-3)
critic.apply(critic.init_weights)

# props to karpathy
def discount_rewards(r, gamma=0.99):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


render = False
env = gym.make('CartPole-v0')
actor_update_freq = 10
critic_update_freq = 1
gamma = 0.99
log_freq = 100
running_rewards = collections.deque(maxlen=log_freq)
rewards, probs, actions, value_approx = [], [], [], []

for i_episode in range(1, 10000):
    done = False
    losses_actor, losses_critic = [], []
    observation_prev = env.reset()
    v_prev = critic.forward(observation_prev)
    ep_reward = 0
    while not done:
        # env.render()
        left_prob = actor.forward(torch.from_numpy(observation_prev).cuda())
        action = 0 if np.random.uniform() < left_prob else 1
        lprob = torch.log(left_prob) if action == 0 else torch.log(1 - left_prob)
        observation, reward, done, info = env.step(action)
        ep_reward += reward
        if done:
            reward = -2*ep_reward

        """
         For critic:
         compute td error:
         t_e = R + y*v(S, w) - v(S, w)
         w = w + lr* o * grad(v(S, w))

         in the code: 
         loss for critic = 
         - td_error * v(S, w)
        """
        v_curr = critic.forward(observation)

        torch_obs = None
        if done:
            torch_obs = 0
        else:
            # print(v_curr)
            torch_obs = gamma * v_curr.item()

        td_error = reward + torch_obs - v_prev.item()

        critic.train(v_prev, -td_error)

        actor.train(lprob, -td_error)

        """
            For critic:

            0 = 0 + alpha*td_error* grad(log_prob(action))

            in the code:
             loss for actor = 
             - td_error * logprob_action(a)
            """
        v_prev = v_curr
        observation_prev = observation

    if i_episode % log_freq == 0:
        print(f"Episode: {i_episode}, last {log_freq} episodes mean reward: { np.mean(running_rewards)}")
    running_rewards.append(ep_reward)

