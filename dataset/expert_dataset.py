from random import sample
from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pickle
import torch
import random
from torch.utils.data import Dataset
import os


class ExpertDataset(Dataset):
    """
    专家轨迹的数据集
    假设专家数据集是一个带有键 {states, actions,rewards,lengths} 的字典，其值包含一个列表
    下面给定形状的专家属性。 每个轨迹可以具有不同的长度

    专家奖励不是必需的，但可用于评估

    Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values containing a list of
    expert attributes of given shapes below. Each trajectory can be of different length.

    Expert rewards are not required but can be useful for evaluation.

        shapes:
            expert["states"]  =  [num_experts, traj_length, state_space]
            expert["actions"] =  [num_experts, traj_length, action_space]
            expert["rewards"] =  [num_experts, traj_length]
            expert["lengths"] =  [num_experts]
    """

    def __init__(self,
                 expert_location: str,
                 device,
                 subopt_class_num: int,
                 num_trajectories: int = 4,
                 subsample_frequency: int = 20,
                 seed: int = 0,
                 label_ratio: float = 0.0,
                 sparse_sample: bool = True
                 ):
        """Subsamples an expert dataset from saved expert trajectories.

        Args:
            专家位置: 保存的专家轨迹的位置
            um_trajectories: 要采样的专家轨迹数（随机）
            subsample_frequency: 以指定的步长频率对每个轨迹进行二次采样
            确定性: 如果为真，则对确定性专家轨迹进行采样
            expert_location:          Location of saved expert trajectories.
            num_trajectories:         Number of expert trajectories to sample (randomized).
            subsample_frequency:      Subsamples each trajectory at specified frequency of steps.
            deterministic:            If true, sample determinstic expert trajectories.
        """
        # 导入专家样本
        all_trajectories = load_trajectories(expert_location, subopt_class_num, num_trajectories, seed)
        self.trajectories = {}

        # Randomize start index of each trajectory for subsampling
        # start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        # 每个 `subsample_frequency` 步骤的子样本专家轨迹, 即从每一整条轨迹中采样片段
        # Subsample expert trajectories with every `subsample_frequency` step.
        for k, v in all_trajectories.items():
            data = v

            if k != "lengths":
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i][0::subsample_frequency]) # 相间隔 subsample_frequency 进行采样
                self.trajectories[k] = samples
            else:
                # Adjust the length of trajectory after subsampling
                self.trajectories[k] = np.array(data) // subsample_frequency

        self.i2traj_idx = {}
        self.length = self.trajectories["lengths"].sum().item()

        del all_trajectories  # Not needed anymore 这个存着专家样本中所有的元素，非常占空间资源
        traj_idx = 0
        i = 0

        # 将trajectories中“lengths”轨迹，按照每条轨迹先后顺序提取出轨迹中元素的索引元组（traj_idx,i）
        # Convert flattened index i to trajectory indx and offset within trajectory
        self.get_idx = []

        for _j in range(self.length):
            while self.trajectories["lengths"][traj_idx].item() <= i:
                i -= self.trajectories["lengths"][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))
            i += 1

        n_traj = len(self.trajectories["states"]) # 专家样本的总轨迹数目
        i_traj = random.sample(range(n_traj), int(label_ratio * n_traj)) # 算法需要用到的轨迹数目(随机抽样)
        self.n_labeled_traj = int(label_ratio * n_traj)
        if sparse_sample: # (间隔抽样)
            i_traj = [i * (int(n_traj/(self.n_labeled_traj-1))-1) for i in range(self.n_labeled_traj)]

        # 采样专家标签轨迹,实际操作过程中，指定轨迹
        self.label_states_traj = []
        self.label_next_states_traj = []
        self.label_actions_traj = []
        self.label_rewards_traj = []
        self.label_done_traj =  []

        for i in i_traj:
            self.label_states_traj.append(self.trajectories["states"][i])
            self.label_next_states_traj.append(self.trajectories["next_states"][i])
            self.label_actions_traj.append(self.trajectories["actions"][i])
            self.label_rewards_traj.append(np.array(self.trajectories["rewards"][i]).sum())
            self.label_done_traj.append(self.trajectories["dones"][i])
        
        self.label_states_traj = torch.as_tensor(self.label_states_traj, dtype=torch.float, device=device)
        self.label_next_states_traj = torch.as_tensor(self.label_next_states_traj, dtype=torch.float, device=device)
        self.label_actions_traj = torch.as_tensor(self.label_actions_traj, dtype=torch.float, device=device)
        self.label_rewards_traj = torch.as_tensor(self.label_rewards_traj, dtype=torch.float, device=device)
        self.label_done_traj = torch.as_tensor(self.label_done_traj, dtype=torch.float, device=device)

        # 打印使用的轨迹
        self.use_rewards_traj = []
        for i in range(n_traj):
            self.use_rewards_traj.append(np.array(self.trajectories["rewards"][i]).sum())
        self.use_rewards_traj = torch.as_tensor(self.use_rewards_traj, dtype=torch.float, device=device)


    def __len__(self) -> int:
        """
        返回数据集的总长度
        Return the length of the dataset
        """
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        states = self.trajectories["states"][traj_idx][i]
        next_states = self.trajectories["next_states"][traj_idx][i]

        # Rescale states and next_states to [0, 1] if are images
        if isinstance(states, np.ndarray) and states.ndim == 3:
            states = np.array(states) / 255.0
        if isinstance(states, np.ndarray) and next_states.ndim == 3:
            next_states = np.array(next_states) / 255.0

        return (states,
                next_states,
                self.trajectories["actions"][traj_idx][i],
                self.trajectories["rewards"][traj_idx][i],
                self.trajectories["dones"][traj_idx][i])
    
    def sample_traj(self):
        return self.label_states_traj, self.label_next_states_traj, self.label_actions_traj, self.label_rewards_traj, self.label_done_traj, self.n_labeled_traj

    def use_label_traj(self):
        return self.use_rewards_traj, self.label_rewards_traj

def load_trajectories(expert_location: str,
                      subopt_class_num: int,
                      num_trajectories: int = 10,
                      seed: int = 0) -> Dict[str, Any]:
    """
    加载专家轨迹
    Load expert trajectories

    Args:
        专家位置：保存的专家轨迹的位置
        num_trajectories: 要采样的专家轨迹数（随机）
        确定性：如果为真，则关闭随机行为
        expert_location:          Location of saved expert trajectories.
        num_trajectories:         Number of expert trajectories to sample (randomized).
        deterministic:            If true, random behavior is switched off.

    Returns:
        包含键 {"states", "lengths"} 和可选的 {"actions", "rewards"} 和值的字典
        包含相应的专家数据属性
        Dict containing keys {"states", "lengths"} and optionally {"actions", "rewards"} with values
        containing corresponding expert data attributes.
    """
    if os.path.isfile(expert_location):
        # 从单个文件加载数据
        # Load data from single file.
        with open(expert_location, 'rb') as f:
            trajs = read_file(expert_location, f)

        rng = np.random.RandomState(seed)
        # Sample random `num_trajectories` experts.
        # 随机抽样 `num_trajectories` 专家
        perm = []
        idx = []
        space_len = int(len(trajs["states"])/subopt_class_num)
        for i in range(subopt_class_num):
            perm = rng.permutation(np.arange(space_len*i,space_len*(i+1)))
            idx.append(perm[:int(num_trajectories/subopt_class_num)])
        idx = np.array(idx).reshape(1,-1).squeeze(0)

        for k, v in trajs.items():  # 转换为字典并从相应key中抽取idx相应的轨迹
            # if not torch.is_tensor(v):
            #     v = np.array(v)  # convert to numpy array
            trajs[k] = [v[i] for i in idx]

    else:
        raise ValueError(f"{expert_location} is not a valid path")
    return trajs


def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
    """
    从输入路径读取文件 假设文件存储字典数据
    Read file from the input path. Assumes the file stores dictionary data.

    Args:
        路径: 本地或 S3 文件路径
        file_handle: 文件的文件句柄
        path:               Local or S3 file path.
        file_handle:        File handle for file.

    Returns:
        文件的字典表示
        The dictionary representation of the file.
    """
    if path.endswith("pt"):
        data = torch.load(file_handle)
    elif path.endswith("pkl"):
        data = pickle.load(file_handle)
    elif path.endswith("npy"):
        data = np.load(file_handle, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    else:
        raise NotImplementedError
    return data
