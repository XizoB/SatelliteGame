from itertools import count

import hydra
import torch
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from make_envs import make_env
from agent import make_agent
from utils.utils import evaluate

def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf/iq", config_name="config")
def main(cfg: DictConfig):
    # 获取参数
    args = get_args(cfg)

    # 设置环境与智能体
    env = make_env(args)
    agent = make_agent(env, args)

    # 设置算法类型
    if args.method.type == "sqil":
        name = f'sqil'
    else:
        name = f'iq'

    policy_file = f'results/{args.method.type}.para'
    if args.eval.policy:
        policy_file = f'{args.eval.policy}'
    print(f'Loading policy from: {policy_file}')

    # 导入训练好的智能体网络结构参数
    if args.eval.transfer:
        agent.load(hydra.utils.to_absolute_path(policy_file), f'_{name}_{args.eval.expert_env}')
    else:
        agent.load(hydra.utils.to_absolute_path(policy_file), f'_{name}_{args.env.name}')
    
    # 评估智能体 -> 回报 更新步数
    eval_returns, eval_timesteps = evaluate(agent, env, num_episodes=args.eval.eps)
    print(f'Avg. eval returns: {np.mean(eval_returns)}, timesteps: {np.mean(eval_timesteps)}')
    if args.eval_only:
        exit()

    # 测量相关性
    measure_correlations(agent, env, args, log=True)


def measure_correlations(agent, env, args, log=False, use_wandb=False):
    GAMMA = args.gamma

    env_rewards = []
    learnt_rewards = []

    for epoch in range(100):

        part_env_rewards = []
        part_learnt_rewards = []

        state = env.reset()
        episode_reward = 0
        episode_irl_reward = 0

        for time_steps in count():
            # 智能体与环境交互得到MDP元组(s,a,s',r,done)直至done
            # env.render()
            action = agent.choose_action(state, sample=False)
            next_state, reward, done, _ = env.step(action)

            ######
            # Get sqil reward
            with torch.no_grad():
                q = agent.infer_q(state, action)
                next_v = agent.infer_v(next_state)
                y = (1 - done) * GAMMA * next_v
                irl_reward = (q - y)  # 恢复奖励函数

            episode_irl_reward += irl_reward.item()
            episode_reward += reward
            part_learnt_rewards.append(irl_reward.item())  # 学习恢复到的每一步奖励
            part_env_rewards.append(reward)  # 环境真实奖励
            ######

            if done:
                break
            state = next_state

        if log:
            print('Ep {}\tEpisode env rewards: {:.2f}\t'.format(epoch, episode_reward))  # 画出一个episode的环境奖励
            print('Ep {}\tEpisode learnt rewards {:.2f}\t'.format(epoch, episode_irl_reward))  # 画出一个episode的恢复奖励

        learnt_rewards.append(part_learnt_rewards)  # 存入每一个epoch的恢复奖励
        env_rewards.append(part_env_rewards)  # 存入每一个epoch的环境奖励

    # mask = [sum(x) < -5 for x in env_rewards]  # skip outliers
    # env_rewards = [env_rewards[i] for i in range(len(env_rewards)) if mask[i]]
    # learnt_rewards = [learnt_rewards[i] for i in range(len(learnt_rewards)) if mask[i]]

    # 相关性测量
    print(f'Spearman correlation {spearmanr(eps(learnt_rewards), eps(env_rewards))}')
    print(f'Pearson correlation: {pearsonr(eps(learnt_rewards), eps(env_rewards))}')

    # 奖励函数推理结果保存路径 vis/{args.env.name}/correlation
    # plt.show()
    savedir = hydra.utils.to_absolute_path(f'vis/{args.env.name}/correlation')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # 画出并保存每一个episode轨迹的环境真实奖励的回报与算法恢复的奖励的回报
    sns.set()
    plt.figure(dpi=150)
    plt.scatter(eps(env_rewards), eps(learnt_rewards), s=10, alpha=0.8)
    plt.xlabel('Env rewards')
    plt.ylabel('Recovered rewards')
    if use_wandb:
        wandb.log({f"Episode rewards": wandb.Image(plt)})
    plt.savefig(savedir + '/%s.png' % 'Episode rewards')
    plt.close()

    # 画出并保存每个episode轨迹的累积环境真实奖励与算法恢复的奖励的每一步骤奖励 前20个轨迹
    sns.set()
    plt.figure(dpi=150)
    for i in range(20):
        plt.scatter(part_eps(env_rewards)[i], part_eps(learnt_rewards)[i], s=5, alpha=0.6)
    plt.xlabel('Env rewards')
    plt.ylabel('Recovered rewards')
    if use_wandb:
        wandb.log({f"Partial rewards": wandb.Image(plt)})
    plt.savefig(savedir + '/%s.png' % 'Partial rewards')
    plt.close()

    # 画出并保存每个episode轨迹的累积环境真实奖励与算法恢复的奖励的每一步骤奖励 前20个轨迹
    sns.set()
    plt.figure(dpi=150)
    for i in range(20):
        plt.plot(part_eps(env_rewards)[i], part_eps(learnt_rewards)[i], markersize=1, alpha=0.8)
    plt.xlabel('Env rewards')
    plt.ylabel('Recovered rewards')
    if use_wandb:
        wandb.log({f"Partial rewards - Interplolate": wandb.Image(plt)})
    plt.savefig(savedir + '/%s.png' % 'Partial rewards - Interplolate')
    plt.close()

    # 画出并保存每一个episode轨迹的环境真实奖励与算法恢复的奖励的每一步骤奖励 前5个epoch
    sns.set()
    plt.figure(dpi=150)
    for i in range(5):
        plt.scatter(env_rewards[i], learnt_rewards[i], s=5, alpha=0.5)
    plt.xlabel('Env rewards')
    plt.ylabel('Recovered rewards')
    if use_wandb:
        wandb.log({f"Step rewards": wandb.Image(plt)})
    plt.savefig(savedir + '/%s.png' % 'Step rewards')
    plt.close()

    # 画出并保存每一个episode轨迹的环境真实奖励的回报与算法恢复的奖励的回报
    sns.set()
    fig = plt.figure(dpi=150)
    fig1 = fig.add_subplot(211)
    fig1.plot(range(1, len(env_rewards)+1), eps(env_rewards), markersize=1, alpha=0.8)
    plt.xlabel('Episodes steps')
    plt.ylabel('Env rewards')
    fig2 = fig.add_subplot(212)
    fig2.plot(range(1, len(learnt_rewards)+1), eps(learnt_rewards), markersize=1, alpha=0.8)
    plt.xlabel('Episodes steps')
    plt.ylabel('Recovered rewards')
    plt.savefig(savedir + '/%s.png' % 'Episode Return')
    plt.close()

def eps(rewards):
    """
    求出每一个epoch中的回报(奖励总和)
    -> [sum(epoch_1), sum(epoch_2), ...]
    """
    return [sum(x) for x in rewards]


def part_eps(rewards):
    """
    沿列表列出累积的的回报(奖励总和)
    -> [epoch[0,0+1,0+1+2,...], epoch[...], ...]
    """
    return [np.cumsum(x) for x in rewards]


if __name__ == '__main__':
    main()
