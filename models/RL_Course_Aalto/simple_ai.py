from pong import Pong
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from skimage.transform import resize
from cv2 import resize
# from PolicyNetwork import PolicyNetwork
import torch
from functools import reduce


class PongAi(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.player_id = player_id
        self.bpe = 4
        self.name = "SimpleAI"

    def get_name(self):
        return self.name

    def get_action(self, ob=None):
        player = self.env.player1 if self.player_id == 1 else self.env.player2
        my_y = player.y
        ball_y = self.env.ball.y + (random.random()*self.bpe-self.bpe/2)
        y_diff = my_y - ball_y

        if abs(y_diff) < 2:
            action = 0  # Stay
        else:
            if y_diff > 0:
                action = self.env.MOVE_UP  # Up
            else:
                action = self.env.MOVE_DOWN  # Down

        return action

    def reset(self):
        # Nothing to done for now...
        return


MOVE_UP, MOVE_DOWN, STAY = 1, 2, 0
class MyAi(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.player_id = player_id
        self.bpe = 4
        self.name = "SimpleAI"
        self.printed_ob = False
        self.i = 0
        self.ob_shape = 40*40
        # self.policy_nn = PolicyNetwork(input_dim=self.ob_shape, learning_rate=0.00001)
        self.df = 0.99
        # self.env.__get_observation(player_id)
        # print(self.env)


    def get_name(self):
        return self.name

    def process_rollout(self, rollout):
        total_r = 0
        pow = 0
        Xs, ys = [], []
        for prev_ob1, action1, rew1, ob1 in rollout:
            total_r += rew1*self.df ** pow
            pow += 1

            Xs.append(self.preprocess_frames(prev_ob1, ob1))
            ys.append(action1)
        # print(Xs)
        X = torch.cat(Xs, 0)
        y = torch.tensor(ys).view(-1, 1).float()
        rewards = [i[2] for i in rollout]
        total_r -= np.mean(rewards)
        total_r /= np.std(rewards)
        # print(X.size())
        # print('X_size: ', X.size())
        print(total_r)
        self.policy_nn.train(X.view(-1, 1600), y, len(y)//32, 3, total_reward=total_r)



    def preprocess_frame(self, ob, smaller=0):
        img_arr = ob[:, :, 0] + ob[:, :, 1]
        if smaller == 0:
            return resize(img_arr, dsize=(40, 40))
        else:
            return resize(img_arr, dsize=(40, 40))//2

    def preprocess_frames(self, ob_prev=None, ob_curr=None):
        ob_prev_ = self.preprocess_frame(ob_prev, 1)
        ob_curr_ = self.preprocess_frame(ob_curr)
        # imtest = Image.fromarray(ob_curr_ - ob_prev_)
        # imtest.save('observation_diff.jpeg')
        # combined_numpy = (ob_curr - ob_prev).reshape((self.ob_shape, ))
        combined = torch.from_numpy(ob_curr_ - ob_prev_).view((self.ob_shape, )).float()
        # print(combined.size())
        return combined

    def saveImage(self, ob):
        pass

    def get_action(self, ob_prev=None, ob_curr=None):
        combined = self.preprocess_frames(ob_prev, ob_curr)
        result = self.policy_nn.forward(combined)
        # result > 0.5 take action up
        rand = np.random.uniform()
        # print(result)
        if rand < result:
            return MOVE_UP
        else:
            return MOVE_DOWN

    def reset(self):
        # Nothing to done for now...
        return



