from collections import deque
import numpy as np
import random
from baselines import deepq
from baselines.common.atari_wrappers import make_atari
from Deep_QNet import QNetwork


# This validation file is made for testing
# learned architecture of the DQN on atari game 'Breakout'
# where almost each action is executed with the learned weights that are loaded into the QNetwork

env = make_atari('BreakoutNoFrameskip-v4')
env = deepq.wrap_atari_dqn(env)

DQN = QNetwork()
DQN.loadModel()

last100R = deque([], maxlen=1000)

ep = 0
for i in range(0, 5000000):

    done = False
    state = env.reset()
    action = env.action_space.sample()
    totalReward = 0
    ep += 1

    while not done:

        randaction_p = random.uniform(0, 1)

        if randaction_p < 0.05:
            action = env.action_space.sample()
        else:
            action = np.argmax(DQN.predict(state))

        state1, reward, done, info = env.step(action)
        totalReward += reward

        state = state1
        # env.render()

    last100R.append(totalReward)
    if ep % 1000 == 0:
        print("LAST 1000 reward mean: ", sum(last100R)/len(last100R))

