import matplotlib.pyplot as plt
import pickle
import numpy as np
np.set_printoptions(threshold=np.nan)
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from collections import deque
import matplotlib.pyplot as plt
import collections
import gym
import random
print("asddsads")
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Simple policy network (for selecting the action)
class Q_network(nn.Module):
    def __init__(self, observation_space, n_actions, lr=0.002):
        super(Q_network, self).__init__()
        self.onehotmask = torch.zeros(observation_space).to(device)
        self.seq = nn.Sequential(
            nn.Linear(observation_space, 50),
            nn.ReLU(),
            nn.Linear(50, n_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    #recieves 1 integer, one hot encode it to 
    def forward(self, state):
        state = torch.tensor(state).float()
        x = self.seq(state)
        return x
    
    def train(self, inputs, targets):
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(inputs, targets)
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def set_weights(self, other_nn):
        self.load_state_dict(other_nn.state_dict())

class LinearSchedule():
    def __init__(self, decay_steps=1000, ep_start=1.0, ep_end=0.05):
        self.decay_steps = decay_steps
        self.ep_start = ep_start
        self.ep_end = ep_end

    def value(self, timestep):
        return max(self.ep_end, self.ep_start - (self.ep_start - self.ep_end)*timestep/self.decay_steps)


class Replay_Memory():
    def __init__(self, maxlen=5000):
        self.array = deque([], maxlen=maxlen)
    
    def sample(self, n_batch):
        results = random.sample(self.array, min(len(self.array), n_batch))
        return results
    
    def append(self, value):
        self.array.append(value)
    
def process_batch(batch, dqn, dqn_target):
    outputs_ = []
    targets_ = []

    for (outputs, state0, action, reward, state, done) in batch:


        outputs_.append(outputs)
        y = reward
        temp = outputs.detach().clone()
        if not done:
            out = state.reshape((1, 4))
            dqn_target_term = torch.max(dqn_target(out)[0]).item()
            y += 0.99 * dqn_target_term
        temp[0, action] = y
        targets_.append(temp)
    return torch.stack(outputs_).to(device), torch.stack(targets_).to(device)
        
env = gym.make('CartPole-v0')
e_start = 1.00
e_end = 0.05
n_step_decays = 5000
gamma  = 0.99
g_t = 0
target_update_I = 50
T_max = 100000
N_episodes = 1500

env._max_episode_steps = T_max
replay_memory = Replay_Memory(1000)
total_rewards = []
running_reward = deque([0], maxlen=10)
for ep in range(N_episodes):
    done = False
    state0 = env.reset()
    t = 0
    total_r = 0
    while True:
#             env.render()
        curr_epsilon = max(end, start - (start-end)*g_t/n_step_decays)
        # print('curr ', curr_epsilon)
        if np.random.uniform() < curr_epsilon:
            action = env.action_space.sample()
            DQN_output = DQN(state0.reshape((1, 4)))
        else:
            DQN_output = DQN(state0.reshape((1, 4)))
            action = torch.argmax(DQN_output[0]).item()

        state, reward, done, info = env.step(action)
        reward = 1
        total_r += reward
        replay_memory.append([DQN_output, state0, action, -reward, state, done])
        
        if g_t % 30 == 0:
            batch = replay_memory.sample(30)
            outputs, targets = process_batch(batch, DQN, DQN)
            DQN.train(outputs, targets)

        state0 = state
        # update target DQN
        if g_t % target_update_I == 0:
            DQN.set_weights(DQN)
        g_t += 1
        t += 1
        if done:
#                 print('ep ', ep, ' done')
#                 if ep % 10 == 0:
            running_reward.append(total_r)
            print('last 10 mean', np.mean(running_reward))
            total_rewards.append(total_r)
            break
#         print(curr_epsilon)
    if ep % 250 == 0:
        print('finished ep ', ep)
        print(curr_epsilon)
plt.plot(np.arange(len(total_rewards)), total_rewards)
plt.show()
env.close()
