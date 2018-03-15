from collections import deque

import numpy as np
import keras
import random
import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam


env = gym.make('CartPole-v0')

#baselines deepq hyperparameters:
    # env -> enviroment
    # q_func = model, -> CNN
    # lr = 1e-4,    ->  learning rate
    # max_timesteps = args.num_timesteps,   ->
    # buffer_size = 10000,
    # exploration_fraction = 0.1,
    # exploration_final_eps = 0.01,
    # train_freq = 4,
    # learning_starts = 10000,
    # target_network_update_freq = 1000,
    # gamma = 0.99,
    # prioritized_replay = bool(args.prioritized)


class QNetwork:
    def __init__(self, learning_rate=0.001, state_space=4, action_space=2):
        # state inputs to the Q-network
        self.model = Sequential()
        self.model.add(Dense(state_space, activation='relu', input_dim=4))
        self.model.add(Dense(6, activation='relu'))
        self.model.add(Dense(action_space, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])
    def predict(self, state):
        return self.model.predict(np.array([state]))[0]

    def fit(self, x, y):
        self.model.fit(x=x, y=y, shuffle=False, verbose=False)


def processBatch(data_arr, df, DQN, DQN_target):
    X, Y = [], []
    for elem in data_arr:
        s1, action, reward, done, s2 = elem.getValues()
        y = DQN.predict(s1)
        y[action] = reward
        if not done:
            y[action] = df * max(DQN_target.predict(s2))
        X.append(s1)
        Y.append(y)
    return np.array(X), np.array(Y)


class Memory:
    def __init__(self, s1, action, reward, done, s2):
        self.values = s1, action, reward, done, s2

    def getValues(self):
        return self.values



df = 0.99
rewards = []

DQN = QNetwork()

replay_memory = deque([], maxlen=1000)
e_start = float(1.0)
e_end = float(0.001)
decay_frames = float(100000)
change = (e_start - e_end)/decay_frames



rewards = []

ep = 0

print(env.spec)

for i in range(0, 500):
    done = False
    state = env.reset()
    totalReward = 0
    ep += 1
    action = None

    while not done:
        r_act = random.uniform(0, 1)

        if r_act <= e_start:
            action = env.action_space.sample()
        else:
            action = np.argmax(DQN.predict(np.array(state)))


        state1, reward, done, info = env.step()
        replay_memory.append((state, action, reward, done, state1))

        if len(replay_memory) >= 100:
            batch = np.random.choice(replay_memory, 32, False)

            X, Y = [], []

            for s1, a, r, d, s2 in batch:

                y = r
                if d:
                    y = r + 0.99 * max(DQN.predict(np.array(s2)))

                s1pred = DQN.predict(np.array(s1))
                s1pred[a] = y

                X.append(s1)
                Y.append(s1pred)

            X = np.array(X)
            Y = np.array(Y)

            DQN.fit(X, Y)

        state = state1
        totalReward += reward
        env.render()

    rewards.append(totalReward)
    print('episode: {}, totReward: {}' .format(ep, totalReward))



#plot all rewards
# plt.plot(rewards)
# plt.ylabel('reward')
# plt.xlabel('episode')
# plt.show()


