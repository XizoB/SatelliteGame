import os
import gym
import time
import torch
import rl_utils
import random
import datetime

import numpy as np

from sac import SAC
from tensorboardX import SummaryWriter


def main():
    #--- 设置随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)



    #--- 环境初始化
    env_name = "SatelliteAvoidance-v0"
    env = gym.make(env_name) # 构建环境对象
    eval_env = gym.make(env_name) # 构建评估环境对象
    env.action_space.seed(0+5)  # env_action_space 随机采样种子
    env.seed(0)
    eval_env.seed(10)
    eval_env.action_space.seed(10+5)  # env_action_space 随机采样种子
    print("nfs:%s; nfa:%s"%(env.observation_space,env.action_space))


    #--- 智能体设置
    actor_lr = 1e-3
    critic_lr = 1e-3
    alpha = 0.2
    alpha_lr = 3e-4

    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000  # 记忆池大小
    minimal_size = 5000
    batch_size = 256  # batch训练
    target_entropy = -5  # 目标熵
    eval_interval = 5000  # 智能体评估周期
    HORIZON = 500  # 每一个episode中有多少决策步
    LEARN_STEPS = 1000000  # 智能体训练步骤
    device = torch.device("cpu")


    #--- 设置智能体
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = 4  # 输入两个维度,卫星的位置与敌对卫星的位置
    action_dim = env.action_space.n  # 输出卫星的动作维度
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")  # 当前时期与时间
    path_dir = os.path.dirname(__file__)  # 当前文件的目录
    log_dir = os.path.join(path_dir,"outputs_rl", ts_str) # 当前时刻保存的路径
    summary_dir = os.path.join(log_dir, 'summary')  # 保存/logs/Ant-v2/cail/seed0-20220802-165144/summary的路径
    writer = SummaryWriter(log_dir=summary_dir)

    agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha, alpha_lr, target_entropy, tau, gamma, device)


    #--- 训练智能体
    return_list = rl_utils.train_off_policy_agent(env, eval_env, agent, HORIZON, replay_buffer, minimal_size, batch_size, eval_interval, log_dir, writer, LEARN_STEPS)


if __name__ == "__main__":
    main()