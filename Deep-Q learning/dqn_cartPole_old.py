from collections import deque

import numpy as np
import keras
import random
import gym
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam


env = gym.make('CartPole-v0')
env.max_episode_steps = 500

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


def plot_rewards(rewards):
    # plot all rewards
    plt.plot(rewards)
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()


def train():
    e_start = float(1.00)
    e_end = float(0.05)
    decay_frames = 1000
    change = float(e_start - e_end) / float(decay_frames)
    epsilon = e_start
    sum = 0
    test_e = epsilon

    df = 0.99
    rewards = []

    DQN = QNetwork()
    DQN_target = DQN.copyModel()

    replay_memory = deque([], maxlen=1000)

    frame = 0
    showedFirst = False
    rewards = []

    ep = 0
    for i in range(0, 1000):

        frame += 1
        done = False
        state = env.reset()
        action = env.action_space.sample()
        totalReward = 0
        ep += 1

        while not done:

            # env.render()
            randaction_p = random.uniform(0, 1)

            if randaction_p < epsilon:
                action = env.action_space.sample()
                # print('random action', action)
            else:
                action = np.argmax(DQN.predict(state))
                # print('action', action)

            state1, reward, done, info = env.step(action)
            totalReward += 1

            replay_memory.append(Memory(state, action, reward, done, state1))

            batch = np.random.choice(replay_memory, min(32, len(replay_memory)), False)

            X, Y = processBatch(batch, df, DQN, DQN_target)
            DQN.fit(X, Y)

            if frame % 50 == 0:
                DQN_target.setWeights(DQN.getWeights())

            state = state1

            frame += 1
            epsilon = max(epsilon - change, e_end)

        rewards.append(totalReward)
        print('episode: {}, epsilon: {}, frames: {}, totalSteps: {}'.format(ep, epsilon, frame, totalReward))

    plot_rewards(rewards)

if __name__ == "__main__":
    train()

