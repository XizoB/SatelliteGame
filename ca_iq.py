"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch
import torch.nn.functional as F
import itertools
import time

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import parameters_to_vector

from collections import Counter
from utils.utils import average_dicts, get_concat_samples


def update_critic_grad(self, batch, expert_conf):
    obs, next_obs, action = batch[0:3]

    # 获得当前观测状态的 V 价值与下一观测状态的 V‘ 价值，比较耗费时间
    # 判断是否使用target网络
    current_V = self.getV(obs)  # 根据当前观测状态获得V_current值
    with torch.no_grad():
        next_V = self.get_targetV(next_obs)

    current_Q = self.critic_1(obs, action)
    critic_loss, loss_dict = ca_iq_loss(self, current_Q, current_V, next_V, batch, expert_conf)

    return critic_loss, loss_dict


def ca_iq_update(self, policy_buffer, expert_buffer, logger, step):
    # 智能体学员与专家MDP元组采样指定Batch大小的数据样本
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_exp_samples(self.batch_size, self.conf, self.device)  # 把conf传入Memory类中，以在里面全局使用

    # 定义智能体类的参数，以求在全局计算
    loss_dict = {}
    self.conf = expert_batch[-1] # 把Memory中的专家样本置信度放在智能体的类中，以在下面各函数中使用
    expert_conf = expert_batch[-2] # 把Memory中的Batch专家样本置信度提取出来
    
    # # 仅使用观察值对 IL 使用策略操作而不是专家操作
    # if self.args.only_expert_states:
    #     policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    #     expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch[0:-2] # 一个batch的专家样本
    #     # Use policy actions instead of experts actions for IL with only observations
    #     expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    # 智能体与专家的MDP元组数据合并与解压
    batch = get_concat_samples(policy_batch, expert_batch[0:5], self.args)

    ########################################################
    # IQ_CAIL 算法体现所在
    # ------------------- 第二步，更新 Q值函数 的theta
    critic_1_loss, critic_loss_dict = update_critic_grad(self, batch, Variable(expert_conf))
    loss_dict.update(critic_loss_dict)


    self.critic_1_optimizer.zero_grad()
    critic_1_loss.backward()
    self.critic_1_optimizer.step()
    loss_dict['train/critic_1_loss'] = critic_1_loss


    return loss_dict, batch, expert_conf



def ca_iq_loss(self, current_Q, current_v, next_v, batch, expert_conf):
    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    args = self.args
    gamma = self.gamma

    loss_dict = {}
    obs, next_obs, action, env_reward, done, is_expert, is_agent = batch

    # y = (1 - done) * self.gamma * self.getV(next_obs)
    y = (1 - done) * gamma * next_v

    reward = (current_Q - y)[is_expert].mul(expert_conf)
    loss = -(reward).mean()
    loss_dict['iq_loss/softq_loss'] = loss.item()  # loss_dict集合中存入softq_loss值


    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = ((current_v - y)[is_expert].mul(expert_conf)+(current_v - y)[is_agent]).mean()/2
    loss += value_loss
    loss_dict['iq_loss/value_loss'] = value_loss.item()

    # Use χ2 divergence (adds a extra term to the loss)
    reward = current_Q - y
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    loss_dict['iq_loss/chi2_loss'] = chi2_loss.item()
    loss_dict['iq_loss/total_loss'] = loss.item()

    return loss, loss_dict

