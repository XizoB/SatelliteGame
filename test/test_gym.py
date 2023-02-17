import datetime
import time
import os

import numpy as np

from gym import spaces

# ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
# path = os.path.dirname(__file__)
# log_dir = os.path.join(path,"outputs", ts_str)

# print(path)
# print(log_dir)

# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

high = np.array([6,6])
low = np.array([0,0])

action_space = spaces.Discrete(5)
observation_space = spaces.Box(-low, high, dtype=np.int0)

print(action_space)

print("spaces.Box:", spaces.Box)
print("env.action_space.dim:", action_space.n)
print("env.observation_space.dim:", observation_space.shape[0])

for _ in range(1):
    action = action_space.sample()  # 卫星策略
    obs = observation_space.sample()
    print("action:", action)
    print("obs:", obs)
