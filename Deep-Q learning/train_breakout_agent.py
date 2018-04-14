from collections import deque
import numpy as np
import random
from baselines import deepq
from baselines.common.atari_wrappers import make_atari
from Deep_QNet import QNetwork


# wrap atari to nicer preprocessed environment
env = make_atari('PongNoFrameskip-v4')

env = deepq.wrap_atari_dqn(env)

# learn on batch of transitions
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

e_start = float(1.00)
e_end = float(0.10)
decay_frames = 500000
change = float(e_start - e_end) / float(1000000)
epsilon = e_start

df = 0.99
rewards = []

DQN = QNetwork(lr = 0.0025, weightsName='pong-weights-1.h5')

DQN_target = DQN.copyModel()

replay_memory = deque([], maxlen=100000)

frame = 0
learnStep = 0
showedFirst = False
last100_ep_rewards = deque([], maxlen=100)

ep = 0
randPolicy = True
for i in range(0, 5000000):

    frame += 1
    done = False
    state = env.reset()
    action = env.action_space.sample()
    totalReward = 0
    ep += 1


    if randPolicy and frame > 50000:
        randPolicy = False

    while not done:
        epsilon = max(epsilon - change, e_end)
        env.render()
        if frame % 5000 == 0:
            DQN_target.setWeights(DQN.getWeights())

        randaction_p = random.uniform(0, 1)

        if randaction_p < epsilon or randPolicy:
            action = env.action_space.sample()
        else:
            action = np.argmax(DQN.predict(state))

        state1, reward, done, info = env.step(action)
        totalReward += reward

        replay_memory.append(Memory(state, action, reward, done, state1))

        if not randPolicy and frame % 16 is 0:
            batch = random.sample(replay_memory, min(32, len(replay_memory)))
            X, Y = processBatch(batch, df, DQN, DQN_target)
            DQN.fit(X, Y)
            #plotting
            # if frame % 500 == 0:
            #     plot_rewards(rewards)
        state = state1
        # env.render()
        frame += 1

    last100_ep_rewards.append(totalReward)
    if ep % 100 is 0:
        print("Episodes: {}, frames: {}, last100 rewards mean: {}, epsilon: {}" .format(ep, frame, sum(last100_ep_rewards)/len(last100_ep_rewards), epsilon) )
        DQN.saveModel()



