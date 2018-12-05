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
from torch import nn
from torch.nn import functional as F
from PIL import Image

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
    def __init__(self, in_channels=3):
        super(ConvNetPG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(6 * 6 * 64, 512)
        self.fc5 = nn.Linear(512, 10)
        self.fc6 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.from_numpy(x).cuda()
        x = x.view((1, 3, 80, 80))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.fc4(x.view(x.size(0), -1)))
        x = F.leaky_relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x

    def train(batch):
        # TODO
        pass


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

def prepro(I):
    print('I', I.shape)
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    print(I.shape)
    return I.astype(np.float).ravel()

# prepro = np.vectorize(prepro)

env = Pong(headless=args.headless)
episodes = 1000
player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = PongAi(env, player_id)

env.set_names(player.get_name(), opponent.get_name())
policy_network = ConvNetPG()
MOVE_UP, MOVE_DOWN, STAY = 1, 2, 0
for i in range(0, episodes):
    done = False
    env.reset()
    rollout_p1 = []
    rew1_total = 0
    rew2_total = 0
    print('new_episode')
    last_3_frames = np.zeros((3, 210, 160, 3))
    print(last_3_frames.shape)
    # print(processed.shape)
    set_frame_pos = 2
    n_frames = 3
    repeat_action = None

    while not done:
        # frame skipping
        if n_frames % 3 == 0:
            # last_3_frames = np.array([prepro(frame) for frame in last_3_frames])
            # prob_up = policy_network.forward(last_3_frames)
            # action1 = MOVE_UP if np.random.uniform() < prob_up else MOVE_DOWN
            # repeat_action = action1
            action1 = 1
        else:
            action1 = repeat_action

        action2 = opponent.get_action()
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        print(ob1.shape)
        t = ob1[:, 25:]
        image = Image.fromarray(t)
        image.show()
        # plt.imshow(ob1[:, 25:])


        # display it        # update last 3 frames
        # last_3_frames[0] = last_3_frames[1]
        # last_3_frames[1] = last_3_frames[2]
        # last_3_frames[2] = prepro(ob1)

        # add rewards
        rew2_total += rew2
        rew1_total += rew1

        if not args.headless:
            env.render()

        # if n_frames % 60 == 0:
        #     preprop_imgs(ob1)

        n_frames += 1
    # if len(rollout_p1) > 0:
    # player.process_rollout(rollout_p1)
    print("Finished episode")
# Needs to be called in the end to shut down pygame
env.end()



