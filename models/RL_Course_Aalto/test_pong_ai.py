from pong import Pong
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint
import pickle
import numpy as np
from simple_ai import PongAi, MyAi
import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from PIL import Image
from skimage.transform import resize
np.set_printoptions(threshold=np.nan)
import collections
torch.set_default_tensor_type('torch.cuda.DoubleTensor')
# CUDA
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Tensor = FloatTensor
#
# if use_cuda:
#     lgr.info ("Using the GPU")
#     X = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch
#     Y = Variable(torch.from_numpy(y_data_np).cuda())
# else:
#     lgr.info ("Using the CPU")
#     X = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch
#     Y = Variable(torch.from_numpy(y_data_np))

class ConvNetPG(nn.Module):
    def __init__(self, in_channels=3, load_model=False):
        super(ConvNetPG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(5 * 4 * 64, 512)
        self.fc5 = nn.Linear(512, 10)
        self.fc6 = nn.Linear(10, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.savePATH = 'model_pong.pt'
        if load_model:
            print('loaded model')
            self.load_model()


    def forward(self, x):
        # print('startforward_', x)
        x = torch.from_numpy(x).cuda()
        # print(x.size())
        x = x.view((1, 3, 70, 67))
        # print(x.size())
        x = F.leaky_relu(self.conv1(x))
        # print(x.size())
        x = F.leaky_relu(self.conv2(x))
        # print(x.size())
        x = F.leaky_relu(self.conv3(x))
        # print(x.size())
        x = F.leaky_relu(self.fc4(x.view(x.size(0), -1)))
        # print(x.size())
        x = F.leaky_relu(self.fc5(x))
        # print(x.size())
        x = torch.sigmoid(self.fc6(x))
        # print(x.size())
        # print(x.item())
        # print('x', x)
        return x

    def train(self, losses):
        # # losses = torch.from_numpy(losses)losses
        # for loss in losses:
        self.optimizer.zero_grad()
        policy_loss = losses.sum()
        # print('policy loss', policy_loss.item())
        # print('losses', losses)
        # # policy_loss.backward()
        # print(policy_loss)
        policy_loss.backward()
        self.optimizer.step()


    def save_model(self):
        torch.save(self.state_dict(), self.savePATH)

    def load_model(self):
        self.load_state_dict(torch.load(self.savePATH))
        # self.eval()

class ConvNetCritic(nn.Module):
    def __init__(self, in_channels=3, load_model=False):
        super(ConvNetCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(5 * 4 * 64, 512)
        self.fc5 = nn.Linear(512, 10)
        self.fc6 = nn.Linear(10, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.savePATH = 'model_pong_critic.pt'
        if load_model:
            print('loaded model')
            self.load_model()


    def forward(self, x):
        # print('startforward_', x)
        x = torch.from_numpy(x).cuda()
        # print(x.size())
        x = x.view((1, 3, 70, 67))
        # print(x.size())
        x = F.leaky_relu(self.conv1(x))
        # print(x.size())
        x = F.leaky_relu(self.conv2(x))
        # print(x.size())
        x = F.leaky_relu(self.conv3(x))
        # print(x.size())
        x = F.leaky_relu(self.fc4(x.view(x.size(0), -1)))
        # print(x.size())
        x = F.leaky_relu(self.fc5(x))
        # print(x.size())
        x = self.fc6(x)
        return x

    def train(self, critic_losses):
        # # losses = torch.from_numpy(losses)losses
        # for loss in losses:
        self.optimizer.zero_grad()
        critic_loss = critic_losses.sum()
        # print('policy loss', policy_loss.item())
        # print('losses', losses)
        # # policy_loss.backward()
        # print(policy_loss)
        critic_loss.backward()
        self.optimizer.step()


    def save_model(self):
        torch.save(self.state_dict(), self.savePATH)

    def load_model(self):
        self.load_state_dict(torch.load(self.savePATH))
        # self.eval()


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--resume", action="store_true", help="Load_model from previous checkpoint")
# parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor (default: 0.99)')
# parser.add_argument('--seed', type=int, default=543, metavar='N',
#                     help='random seed (default: 543)')
# parser.add_argument('--render', action='store_true',
#                     help='render the environment')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='interval between training status logs (default: 10)')
# args = parser.parse_args()
#
#
#
resume = False # resume from previous checkpoint?
args = parser.parse_args()
# print('aqqqqq', args)


def prepro(I):
    # print('I', I.shape)
    """ input image of 200x210 -> to 80x80 1d image"""
    """ prepro 200x210x3 uint8 frame into 67, 70 2d vector """
    I = I[::3,::3, :] # downsample by factor of 2
    I = I[:, :, 0] + I[:, :, 1]
    I[I != 0] = 1
    return I.astype(np.float)


# prepro = np.vectorize(prepro)

env = Pong(headless=args.headless)
episodes = 1000000
gamma = 0.99
player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = PongAi(env, player_id)
env.set_names(player.get_name(), opponent.get_name())

policy_network = ConvNetPG(load_model=args.resume)
policy_critic = ConvNetCritic(load_model=args.resume)

MOVE_UP, MOVE_DOWN, STAY = 1, 2, 0

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros(len(r))
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

rewards, observations, actions, probs, critic_forwards = [], [], [], [], []
last_100_ep_rewards = collections.deque(maxlen=100)
for i in range(0, episodes):
    done = False
    env.reset()
    last_3_frames = np.zeros((3, 200, 210, 3))
    last_3_frames = np.array([prepro(frame) for frame in last_3_frames])
    ep = 1
    batch_size = 1
    # iterate unless the episode is complete
    while not done:
        # get action and logprob
        prob_up = policy_network.forward(last_3_frames)
        action1 = MOVE_UP if np.random.uniform() < prob_up.item() else MOVE_DOWN
        logProb = torch.log(prob_up) if action1 == 1 else torch.log(1 - prob_up)
        # update buffers
        probs.append(logProb)
        actions.append(action1)
        observations.append(last_3_frames)
        # save to update later
        critic_forwards.append(policy_critic.forward(last_3_frames))

        action2 = opponent.get_action()
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        rewards.append(rew1)

        # update frames with new ones
        last_3_frames[0] = last_3_frames[1]
        last_3_frames[1] = last_3_frames[2]
        last_3_frames[2] = prepro(ob1)

        if not args.headless:
            env.render()
        # update Policy Network
        if done:
            # print('i', i)
            # add last reward for book keeping 10 or -10
            last_100_ep_rewards.append(rew1)
            if i % batch_size == 0:
                # print('preforming update of on network')
                # rewards here,
                ers = np.vstack(rewards)
                # make discounted rewards [G1, G2, G3 ... ]
                # print('ers1, ', ers)
                ers = discount_rewards(ers)
                # print(ers)
                # print('rewards', ers.max())
                # print('rewards min', ers.min())

                eos = np.vstack(observations)
                eas = np.vstack(actions)
                # make the logs negative since we are using gradient ascent
                eps = torch.stack(probs)
                # print(ers.max(), ers.min())
                ers = discount_rewards(ers)
                # print(max(rewards), min(rewards))
                # print('rewards', rewards)
                ers = torch.tensor(ers).cuda()
                ers = (ers - ers.mean()) / ers.std()
                cfs = torch.stack(critic_forwards)
                # print(eps)
                # print('losses', eps)
                # print('ers2', ers)

                losses = torch.zeros(len(rewards)).cuda()
                losses_critic = torch.zeros(len(rewards)).cuda()
                for k in range(0, len(rewards)):
                    # print(f"epsi = {eps[i]}, ers_i = {# ers[i]}")
                    losses[k] = -eps[k] * (ers[k] - cfs[k].item())
                    losses_critic[k] = - torch.pow(ers[k] - cfs[k], 2)
                    # print('losses_i', losses[i])

                # print('losses_inner', losses)
                policy_critic.train(losses_critic)
                policy_network.train(losses)

                rewards, observations, actions, probs, critic_forwards = [], [], [], [], []
    # ep += 1
    # print('ssss')
    # print(type(i))
    if i % 100 == 0:
        policy_network.save_model()
        policy_critic.save_model()
        print("Mean reward from 100 last episodes", np.mean(last_100_ep_rewards), "on episode: ", i)


    # if len(rollout_p1) > 0:
    # player.process_rollout(rollout_p1)
    # print("Finished episode")
# Needs to be called in the end to shut down pygame
env.end()



