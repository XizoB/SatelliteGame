import datetime
import os
import random
import time
import hydra
import numpy as np
import torch
import torch.nn.functional as F
# import wandb

from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from collections import deque
from itertools import count
from utils.logger import Logger
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import evaluate, eval_mode, save

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() and cfg.device is None else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg

# main函数里面的存储路径全在 /outputs/当前日期/此刻时间/ 目录下
@hydra.main(config_path="conf/rl", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    # wandb.init(project=args.env.name + '_rl', entity='iq-learn', sync_tensorboard=True, reinit=True, config=args)

    #--- 设置种子类别
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #--- 设置GPU
    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    #--- 设置训练环境
    env_args = args.env
    env = make_env(args)  # 初始化训练环境
    eval_env = make_env(args)  # 初始化评估环境

    env.seed(args.seed)
    eval_env.seed(args.seed + 10)
    env.action_space.seed(args.seed+5)  # env_action_space 随机采样种子
    eval_env.action_space.seed(args.seed+5)  # env_action_space 随机采样种子

    #--- 初始化实验参数
    REPLAY_MEMORY = int(env_args.replay_mem)            # Replay_Buffer 的大小
    INITIAL_MEMORY = int(env_args.initial_mem)          # 第一次更新前，Replay_Buffer 中需要提前存入的 Transition 片段数目
    EPISODE_WINDOW = int(env_args.eps_window)           # 智能体训练过程中的奖励窗口大小
    LEARN_STEPS = int(env_args.learn_steps)             # 智能体更新次数
    HORIZON = int(env_args.horizon)                     # 智能体与环境交互的最长步数，限制每一个 EPOCH 的交互数
    ROUllOUT_LENGTH = int(env_args.roullout_length)     # 每一次更新循环前，智能体需要与环境交互的数目
    TRAIN_STEPS = int(env_args.train_steps)             # 每一次更新循环内，智能体的策略与价值网络的更新数目

    #--- 初始化智能体
    agent = make_agent(env, args)

    #--- 初始化 Replay_Buffer
    memory_replay = Memory(REPLAY_MEMORY, args.seed)

    #--- 设置训练日志目录 Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir, log_frequency=args.log_interval, writer=writer, save_tb=True, agent=args.agent.name)
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # 跟踪训练过程中的奖励

    ####################################################################
    steps = 0  # 智能体与环境交互步数
    learn_steps = 0  # 智能体学习更新步数
    begin_learn = False
    # 创建无限迭代, 开始训练智能体
    for epoch in count():
        # 初始化一个epoch的训练设置
        state = env.reset()
        episode_reward = 0
        done = False

        start_time = time.time()  # 记录一个epoch的开始时间

        # 每一个epoch/episode/trajectory 中 episode 的最大步数，避免让智能体一直探索或陷入死循环
        for episode_step in range(HORIZON):
            # env.render()

            # Step-1 --- 智能体交互闭环： 与环境进行交互得到(s,a,r,s',done)
            #--- 卫星与环境进行交互
            with eval_mode(agent):
                action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            #--- 敌对卫星与环境进行交互
            enemy_action = env.action_space.sample()  # 敌对卫星策略
            enemy_next_state = env.enemy_step(enemy_action)  # 敌对卫星运动学
            
            # 仅在一个 epoch/episode/horizon 结束的时候 (done)才为1 到时间限制时为0
            # （允许无限引导）在时间限制的环境中直接截断在固定交互步数中（1000）
            next_state = np.hstack([next_state, enemy_next_state])  # 合并状态
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            memory_replay.add((state, next_state, action, reward, done_no_lim))

            if done:
                break
            state = next_state

            # Step-2 --- 智能体训练闭环
            if memory_replay.size() >= INITIAL_MEMORY and (steps-INITIAL_MEMORY) % ROUllOUT_LENGTH == 0:
                # --- 开始训练
                if begin_learn is False:
                    print(f'Learn begins and memory_replay_size is {memory_replay.size()}!')
                    begin_learn = True

                for _ in range(TRAIN_STEPS):
                    # --- 评估智能体， 按照评估间隙eval_interval
                    if learn_steps % args.env.eval_interval == 0:
                        eval_returns = evaluate(agent, eval_env, args, logger, epoch, learn_steps)
                        learn_steps += 1  # To prevent repeated eval at timestep 0
                        save(agent, args, eval_returns, learn_steps, output_dir='results')

                    # --- 结束训练
                    if learn_steps == LEARN_STEPS:
                        print(f'Finished and memory_replay_size is {memory_replay.size()}!')
                        # wandb.finish()
                        return  # 结束主函数

                    # --- 训练智能体
                    losses = agent.update(memory_replay, logger, learn_steps)
                    learn_steps += 1

                    if learn_steps % args.log_interval == 0:
                        for key, loss in losses.items():
                            writer.add_scalar(key, loss, global_step=learn_steps)


        # tensorboard 可视化
        rewards_window.append(episode_reward)
        logger.log('train/episode', epoch, steps)
        logger.log('train/episode_reward', episode_reward, steps)
        logger.log('train/duration', time.time() - start_time, steps)
        # logger.dump(steps, save=begin_learn)
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))

    ####################################################################


if __name__ == "__main__":
    main()
