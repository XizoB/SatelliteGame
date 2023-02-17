import gym
import time

from gym import spaces

#--- 环境初始化
env_name = "SatelliteAvoidance-v0"
env = gym.make(env_name) # 构建环境对象
env.reset()
nfs = env.observation_space
nfa = env.action_space
print("nfs:%s; nfa:%s"%(nfs,nfa))
print(env.state)
print(env.observation_space)
print(env.action_space.n)
info = {}
info["x"], info["y"] = env.start[0], env.start[1]


for _ in range(1000):
    env.render()
    #--- 敌对卫星与环境交互
    enemy_action = env.action_space.sample()  # 敌对卫星策略
    env.enemy_step(enemy_action)  # 敌对卫星运动学

    #--- 卫星与环境交互
    action = env.action_space.sample()  # 卫星策略
    state, reward, isdone, info = env.step(action)  # 卫星运动学
    print("{0}, {1}, {2}, {3}".format(action, reward, isdone, info))

    time.sleep(1)