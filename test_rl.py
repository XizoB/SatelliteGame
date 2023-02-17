import gym
import torch
import random

import numpy as np

from sac import SAC
from gym import wrappers
from itertools import count
from time import time, sleep # just to have timestamps in the files



def main():
    #--- 设置随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


    #--- 构建格子世界环境
    env_name = "SatelliteAvoidance-v0"
    env = gym.make(env_name) # 构建环境对象
    env.action_space.seed(0+100)  # env_action_space 随机采样种子
    env.seed(0 + 100)

    # env = wrappers.Monitor(env, f'./videos/{env_name}/' + str(time()) + '/')  # 录制视频


    #--- 构建卫星智能体
    actor_lr = 1e-3
    critic_lr = 1e-2
    alpha = 0.2
    alpha_lr = 1e-2
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数

    target_entropy = -1
    HORIZON = 200  # 每一个episode中有多少决策步
    device = torch.device("cpu")

    state_dim = 4  # 输入两个维度,卫星的位置与敌对卫星的位置
    action_dim = env.action_space.n  # 输出卫星的动作
    agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha, alpha_lr, target_entropy, tau, gamma, device)


    #--- 导入智能体
    policy_file = "/home/xizobu/NutstoreFiles/RCIR/卫星对抗博弈/Satellite/Test/train_model"
    agent.load(policy_file)
    print(f'Loading policy from: {policy_file}')


    for epoch in count():
        state = env.reset()
        print("start:", state)
        episode_reward = 0
        episode_step = 0
        for _ in range(HORIZON):
            env.render()
            #--- 卫星与环境进行交互
            action = agent.take_action(state)  # 卫星策略
            next_state, reward, done, _ = env.step(action)  # 卫星运动学
            episode_reward += reward

            #--- 敌对卫星与环境进行交互
            enemy_action = env.action_space.sample()  # 敌对卫星策略
            enemy_next_state = env.enemy_step(enemy_action)  # 敌对卫星运动学

            if done:
                break
            next_state = np.hstack([next_state, enemy_next_state])  # 合并状态
            state = next_state

            episode_step += 1

            sleep(0.1)
        print("end:", state)
        print('Ep {}\tMoving average score: {:.2f}\tEpisode_step:{}'.format(epoch, episode_reward, episode_step))


if __name__ == "__main__":
    main()
