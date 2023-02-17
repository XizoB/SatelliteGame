import numpy as np
import torch
import os
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def evaluate(actor, env, args, logger, epoch, learn_steps, vis=True):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    num_episodes = args.eval.eps
    total_returns = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        
        with eval_mode(actor):
            for _ in range(args.env.horizon):
                #--- 卫星与环境进行交互
                action = actor.choose_action(state)  # 卫星策略
                next_state, reward, done, _ = env.step(action)  # 卫星运动学

                #--- 敌对卫星与环境进行交互
                enemy_action = env.action_space.sample()  # 敌对卫星策略
                enemy_next_state = env.enemy_step(enemy_action)  # 敌对卫星运动学

                #--- 存储transition片段
                next_state = np.hstack([next_state, enemy_next_state])  # 合并状态
                state = next_state
                total_rewards += reward
                if done:
                    break
            total_returns.append(total_rewards)
            
    total_returns = np.array(total_returns)
    logger.log('eval/episode_reward', total_returns.mean(), learn_steps)
    logger.log('eval/episode_reward_std', total_returns.std(), learn_steps)
    logger.log('eval/episode', epoch, learn_steps)
    logger.log('eval/max_returns', total_returns.max(), learn_steps)
    logger.log('eval/min_returns', total_returns.min(), learn_steps)
    logger.dump(learn_steps, ty="eval")
    # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

    return np.around(total_returns.mean(),3)


def weighted_softmax(x, weights):
    x = x - torch.max(x, dim=0)[0]
    return weights * torch.exp(x) / torch.sum(
        weights * torch.exp(x), dim=0, keepdim=True)


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    """
    构建多层感知机MLP
    hidden_depth: 表示构建几层MLP
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]  # 第一层+激活函数ReLu
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]  # 第N层+激活函数ReLu
        mods.append(nn.Linear(hidden_dim, output_dim))  # 最后一层
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def get_concat_samples(policy_batch, expert_batch, args):
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch

    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch

    if args.method.type == "sqil":
        # convert policy reward to 0
        online_batch_reward = torch.zeros_like(online_batch_reward)
        # convert expert reward to 1
        expert_batch_reward = torch.ones_like(expert_batch_reward)

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat([online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
    is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool), torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)
    is_agent = torch.cat([torch.ones_like(online_batch_reward, dtype=torch.bool), torch.zeros_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert, is_agent


def save_state(tensor, path, num_states=5):
    """Show stack framed of images consisting the state"""

    tensor = tensor[:num_states]
    B, C, H, W = tensor.shape
    images = tensor.reshape(-1, 1, H, W).cpu()
    save_image(images, path, nrow=num_states)
    # make_grid(images)


def average_dicts(dict1, dict2):
    return {key: 1/2 * (dict1.get(key, 0) + dict2.get(key, 0))
                     for key in set(dict1) | set(dict2)}


def build_param_list(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    weights = []
    biases = []
    units = input_dim
    re_grad = True
    hidden_units = [] # 隐藏层的结构(64x64)

    for _ in range(hidden_depth):
        hidden_units.append(hidden_dim)

    for next_units in hidden_units:
        weights.append(torch.zeros(next_units, units, requires_grad=re_grad))
        biases.append(torch.zeros(next_units, requires_grad=re_grad))
        units = next_units
    weights.append(torch.zeros(output_dim, units, requires_grad=re_grad))
    biases.append(torch.zeros(output_dim, requires_grad=re_grad))

    return weights, biases


def save(agent, args, eval_returns, learn_steps, output_dir='results'):
    """
    保存智能体的网络结构
    """
    name = f'{args.method.type}_{args.env.name}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = f'{output_dir}/{learn_steps}_returns_{eval_returns}'
    if not os.path.exists(output_dir):  # 如果该路径下文件夹不存在，则创建该路径文件夹
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/{args.agent.name}_{name}')

def save_trajs(info_use, info_label, output_dir='trajs'):
    """
    保存导入专家数据的信息
    """
    if not os.path.exists(output_dir):  # 如果该路径下文件夹不存在，则创建该路径文件夹
        os.mkdir(output_dir)
    file = f'{output_dir}/info_trajs.txt'
    with open(file, 'w') as f:
        f.write(f'info_use: {info_use}\n')
        f.write(f'info_use.shape: {info_use.shape}\n')
        f.write(f'use_mean: {info_use.mean().item()}\n')
        f.write(f'use_max: {info_use.max().item()}\n')
        f.write(f'use_min: {info_label.min().item()}\n')
        f.write(f'info_label: {info_label}\n')
        f.write(f'info_label.shape: {info_label.shape}\n')
        f.write(f'label_mean: {info_label.mean().item()}\n')
        f.write(f'label_max: {info_label.max().item()}\n')
        f.write(f'label_min: {info_label.min().item()}\n')

def save_config(config, learn_steps, output_dir):
    """
    保存智能体的置信度
    """
    if not os.path.exists(output_dir):  # 如果该路径下文件夹不存在，则创建该路径文件夹
        os.mkdir(output_dir)
    file = f'{output_dir}/{learn_steps}_conf.csv'
    with open(file, 'w') as f:
        for i in range(config.shape[0]):
            f.write(f'{config[i].item()}\n')
