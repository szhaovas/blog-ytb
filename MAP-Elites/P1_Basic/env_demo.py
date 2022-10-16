import gym
import QDgym
import numpy as np
from math import pi

env = gym.make('QDAntBulletEnv-v0')
env.seed(0)
for i in range(5):
    state = env.reset()
    env.seed(0)
    done = False
    while not done:
        t = env.T
        action = np.sin([pi*t / 10]*8)
        state, reward, done, info = env.step(action)
    print(f'fitness: {env.tot_reward}')
    print(f'bd: {env.desc}')
