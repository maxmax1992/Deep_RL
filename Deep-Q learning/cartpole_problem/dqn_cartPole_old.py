from collections import deque

import numpy as np
import keras
import random
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras import backend as K
from keras.optimizers import Adam


env = gym.make('CartPole-v0')
env._max_episode_steps = 500

class QNetwork:
    def __init__(self, learning_rate=0.0025, state_space=4, action_space=2, model=None):
        if model is not None:
            self.model = model
            return

        # state inputs to the Q-network
        self.model = Sequential()
        self.model.add(Dense(8, activation='relu', input_dim=state_space))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_space, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

    def copyModel(self):
        copy_model = keras.models.clone_model(self.model)
        copy_model.set_weights(self.model.get_weights())
        return QNetwork(model=copy_model)

    def saveModel(self):
        self.model.save_weights('breakout-v0-weights-v0.h5')

    def loadModel(self):
        self.model.load_weights('breakout-v0-weights-v0.h5')

    def predict(self, state):
        return self.model.predict(np.array([state]))[0]

    def fit(self, x, y):
        self.model.fit(x=x, y=y, shuffle=False, verbose=False)

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def getWeights(self):
        return self.model.get_weights()

def processBatch(data_arr, df, DQN, DQN_target):
    X, Y = [], []

    for elem in data_arr:
        s1, action, reward, done, s2 = elem.getValues()
        y = DQN.predict(s1)
        y[action] = reward
        if not done:
            y[action] = reward + df * max(DQN_target.predict(s2))
        X.append(s1)
        Y.append(y)
    return np.array(X), np.array(Y)


class Memory:
    def __init__(self, s1, action, reward, done, s2):
        self.values = s1, action, reward, done, s2

    def getValues(self):
        return self.values

colors = ['red', 'blue', 'green']
patches = []

def plot_rewards(rewards, saveName, label, plotnum):
    # plot all rewards
    plt.plot(rewards, color=colors[plotnum])
    plt.ylabel('reward')
    plt.xlabel('episode')
    color_patch = mpatches.Patch(color=colors[plotnum], label=label)
    patches.append(color_patch)
    plt.legend(handles=patches)
    plt.savefig(saveName)


def train(eps=100, use_replay=True, use_target=True, rand_agent=False):
    last10_avg = deque([], maxlen=10)
    e_start = float(1.00)
    e_end = float(0.10)
    decay_frames = 1000
    change = float(e_start - e_end) / float(decay_frames)
    epsilon = e_start
    test_e = epsilon

    df = 0.99
    rewards = []

    DQN = QNetwork()
    DQN_target = None
    if use_target:
        DQN_target = DQN.copyModel()


    replay_memory = None
    if use_replay:
        replay_memory = deque([], maxlen=1000)

    frame = 0
    rewards = []
    ep = 0
    for i in range(0, eps):

        frame += 1
        done = False
        state = env.reset()
        action = env.action_space.sample()
        totalReward = 0
        ep += 1

        while not done:

            # env.render()
            randaction_p = random.uniform(0, 1)

            if rand_agent or randaction_p < epsilon:
                action = env.action_space.sample()
                # print('random action', action)
            else:
                action = np.argmax(DQN.predict(state))
                # print('action', action)

            state1, reward, done, info = env.step(action)
            totalReward += 1

            batch = None
            if use_replay:
                replay_memory.append(Memory(state, action, reward, done, state1))
                batch = np.random.choice(replay_memory, min(32, len(replay_memory)), False)
            else:
                batch = np.array([Memory(state, action, reward, done, state1)])

            X, Y = None, None
            if use_target:
                X, Y = processBatch(batch, df, DQN, DQN_target)
            else:
                X, Y = processBatch(batch, df, DQN, DQN)
            DQN.fit(X, Y)

            if frame % 50 == 0 and use_target:
                DQN_target.setWeights(DQN.getWeights())

            state = state1
            frame += 1
            epsilon = max(epsilon - change, e_end)
        last10_avg.append(totalReward)
        rewards.append(sum(last10_avg)/float(len(last10_avg)))
        print('episode: {}, epsilon: {}, frames: {}, totalSteps: {}'.format(ep, epsilon, frame, totalReward))

    return rewards

if __name__ == "__main__":
    # random agent
    plot_rewards(train(300, use_replay=True, use_target=False, rand_agent=False), './plots/learning_result_1.png', label='RM', plotnum=0)
    plot_rewards(train(300, use_replay=False, use_target=True, rand_agent=False), './plots/learning_result_2.png', label='TN', plotnum=1)
    plot_rewards(train(300, use_replay=True, use_target=True, rand_agent=False), './plots/learning_result_3.png', label='TM + RN', plotnum=2)
    K.clear_session()

