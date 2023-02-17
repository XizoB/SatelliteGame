import numpy as np
import torch
import collections
import random
import time
import os
import sys

from itertools import count


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)


class Logger(object):
    def __init__(self, dir, stream=sys.stdout):
        self.terminal = stream
        filename = f"{dir}/default.log"
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def train_off_policy_agent(env, eval_env, agent, HORIZON, replay_buffer, minimal_size, batch_size, eval_interval, log_dir, writer, LEARN_STEPS):
    """训练off_policy智能体
    """
    return_list = []
    learn_step = 0  # 智能体学习更新步数
    old_retuns = -np.inf
    begin_learn = False
    sys.stdout = Logger(log_dir,stream=sys.stdout)

    for epoch in count():
        episode_return = 0
        episode_step = 0
        state = env.reset()
        # print("start_state::", state)
        done = False
        
        epoch_time = time.time()
        # 每一个epoch/episode/trajectory 中 episode 的最大步数，避免让智能体一直探索或陷入死循环
        for _ in range(HORIZON): # 一共训练100个episodes
            # env.render()
            ########### Step-1 --- 智能体交互闭环：与环境进行交互得到(s,a,r,s',done)
            #--- 卫星与环境进行交互
            action = agent.take_action(state)  # 卫星策略
            next_state, reward, done, info = env.step(action)  # 卫星运动学
            # print("action:", action)

            #--- 敌对卫星与环境进行交互
            enemy_action = env.action_space.sample()  # 敌对卫星策略
            enemy_next_state = env.enemy_step(enemy_action)  # 敌对卫星运动学

            #--- 存储transition片段
            next_state = np.hstack([next_state, enemy_next_state])  # 合并状态
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            replay_buffer.add(state, action, reward, next_state, done_no_lim)  # 存入记忆池
            episode_return += reward

            ########### Step-2 --- 智能体训练闭环
            if replay_buffer.size() >= minimal_size:  #--- SAC智能体更新的最小minimal_size
                # --- 开始训练
                if begin_learn is False:
                    print(f'Learn begins and memory_replay_size is {replay_buffer.size()}!')
                    begin_learn = True

                # --- 评估智能体， 按照评估间隙eval_interval
                if learn_step % eval_interval == 0:
                    eval_returns = evaluate(agent, eval_env, HORIZON, writer, learn_step)
                    learn_step += 1  # To prevent repeated eval at timestep 0
                    if eval_returns >= old_retuns:
                        save(agent, eval_returns, log_dir, "best")
                        old_retuns = eval_returns
                    save(agent, eval_returns, log_dir, "train")


                # --- 训练智能体
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict, writer, learn_step) # SAC智能体更新
                learn_step += 1
                return_list.append(episode_return)
                writer.add_scalar('train/reward', episode_return, learn_step)


                # --- 结束训练
                if learn_step == LEARN_STEPS:
                    print(f'Finished and memory_replay_size is {replay_buffer.size()}!')
                    return  # 结束主函数


            if done:
                # print("end_state:[{},{}]".format(info["x"],info["y"]) )
                break
            state = next_state
            episode_step += 1

        # print("end_state::", state)
        writer.add_scalar('train/episode_step', episode_step, learn_step)
        writer.add_scalar('train/epoch_time', time.time()-epoch_time, learn_step)


    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def evaluate(agent, env, HORIZON, writer, learn_step):
    """评估智能体
    """
    total_returns = []
    for _ in range(10):
        state = env.reset()
        done = False
        total_rewards = 0
        
        for _ in range(HORIZON):
            #--- 卫星与环境进行交互
            action = agent.take_action(state)  # 卫星策略
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
    mean = int(total_returns.mean())
    std = int(total_returns.std())
    max = int(total_returns.max())
    min = int(total_returns.min())

    writer.add_scalar('eval/episode_reward', mean, learn_step)
    writer.add_scalar('eval/episode_reward_std', std, learn_step)
    writer.add_scalar('eval/max_returns', max, learn_step)
    writer.add_scalar('eval/min_returns', min, learn_step)
    print("Learn_Steps:{}---Mean:{}---Max:{}---Min:{}---Std:{}".format(learn_step, mean, max, min, std()))

    return np.around(total_returns.mean(),3)

def save(agent, eval_returns, log_dir, type):
    """保存智能体的网络结构
    """
    model_dir = os.path.join(log_dir, f'{type}_model')  # 保存/logs/Ant-v2/cail/seed0-20220802-165144/model的路径
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    actor_path = f"{model_dir}/actor"
    critic_1_path = f"{model_dir}/critic_1"
    critic_2_path = f"{model_dir}/critic_2"

    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic_1.state_dict(), critic_1_path)
    torch.save(agent.critic_2.state_dict(), critic_2_path)
    # print('Saving models to {}'.format(model_dir))

