import random
import numpy as np
import gym
import torch
#随机从列表中选取多个不重复的元素

# np.random.seed(0)

# a = np.random.randint(0,7,size=(2))
# b = np.random.randint(0,7,size=(2))
# print(a,b)

# c = np.array([0,1])
# d = [0,4]
# e = [0,1]
# ddd = np.array([c==e,c==e,c==e])
# print(ddd)
# print(ddd.all())



#--- 设置随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#--- 环境初始化
env_name = "SatelliteAvoidance-v0"
env = gym.make(env_name) # 构建环境对象
nfs = env.observation_space
nfa = env.action_space
print("nfs:%s; nfa:%s"%(nfs,nfa))

env.seed(0)
env.action_space.seed(0)  # env_action_space 随机采样种子

for _ in range(2):
    a = env.action_space.sample()
    print(a)

print(np.inf)


probs = torch.tensor([0.1,0.3,0.6])
action_dist = torch.distributions.Categorical(probs)
action = action_dist.sample()
print(probs)
print(action_dist)
print(action)
