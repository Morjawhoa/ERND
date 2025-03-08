import queue
import time
from collections import deque
import yaml
import threading
import random
import torch
import numpy as np
from DuelingDQN import DuelingNet, DoubleDuelingDQN
import json


def convert_item(item):
    """递归地将item中的张量转换为列表，其它支持的容器也做相应处理"""
    if isinstance(item, torch.Tensor):
        # 转换张量为列表
        return item.tolist()
    elif isinstance(item, dict):
        # 对字典的每个值递归转换
        return {key: convert_item(value) for key, value in item.items()}
    elif isinstance(item, (list, tuple, deque)):
        # 对列表、元组或deque中每个元素递归转换
        converted = [convert_item(elem) for elem in item]
        # 如果原数据是元组则返回元组，否则返回列表
        return tuple(converted) if isinstance(item, tuple) else converted
    else:
        # 其它类型直接返回
        return item


def tuple_to_dict(item):
    sarsd = {'state': item[0],
             'action': item[1],
             'reward': item[2],
             'next_state': item[3],
             'done': item[4]}
    return sarsd


class ReplayBuffer(object):
    def __init__(self, capacity: int, important_scale=3):
        # important_scale: 代表重要列表的训练占比多少
        self.common_buffer = deque(maxlen=capacity)
        self.import_buffer = deque(maxlen=int(capacity / important_scale))
        self.now_experience = None
        self.important_scale = 3

    def add_common(self, state, action, reward, next_state, done):
        self.common_buffer.append((state, action, reward, next_state, done))
        self.now_experience = (state, action, reward, next_state, done)

    def add_important(self, state, action, reward, next_state, done):
        self.import_buffer.append((state, action, reward, next_state, done))
        self.now_experience = (state, action, reward, next_state, done)

    def sample(self, batch_size):
        # 从普通的经验池和重要的经验池抽出经验训练，以及加上最新的一个经验
        important_size = min(int(batch_size / self.important_scale), self.import_buffer.__len__())
        common_size = batch_size - important_size - 1
        common_size = min(int(common_size), self.common_buffer.__len__())

        transitions = random.sample(self.import_buffer, important_size)
        transitions.extend(random.sample(self.common_buffer, common_size))
        transitions.append(self.now_experience)

        state, action, reward, next_state, done = zip(*transitions)
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        return state, action, reward, next_state, done

    def to_dict(self):
        """将 ReplayBuffer 转换为字典，包含所有的 deque 和元组结构"""
        return {
            "common_buffer": [self._convert_tuple_to_serializable(exp) for exp in self.common_buffer],
            "import_buffer": [self._convert_tuple_to_serializable(exp) for exp in self.import_buffer]
        }

    def _convert_tuple_to_serializable(self, exp):
        """将元组中的 Tensor 转换为可读的格式"""
        return tuple(x.tolist() if isinstance(x, torch.Tensor) else x for x in exp)

    def save_to_yaml(self, file_name):
        """保存为 YAML 文件"""
        with open(file_name, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=True)

    def __len__(self):
        return self.common_buffer.__len__() + self.import_buffer.__len__()


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, input_size, hidden_size, output_size, channel_size, gamma, update_steps, device, learning_rate, batch_size,
                 min_size, important_scale=3, alpha=0.6, beta=0.4):
        # important_scale: 代表重要列表的训练占比多少
        self.buffer = []
        self.priorities = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.min_size = min_size

        self.important_scale = important_scale
        self.alpha = alpha  # 控制优先级的程度
        self.beta = beta  # 用于控制重要性采样权重的程度
        self.now_experience = None

        self.queues = {'memory': queue.Queue(),
                       'priorities': queue.Queue(),
                       'samples': queue.Queue(),
                       'indices': queue.Queue(),
                       'state_dict': queue.Queue()}

        self.dqn = DoubleDuelingDQN(input_size, hidden_size, output_size, channel_size, gamma, update_steps, device, learning_rate)

        self.device = device
        self.flag = True
        self.save_now = False
        self.memorizer = threading.Thread(target=self.auto_memory)
        self.memorizer.start()

    def auto_memory(self):
        while self.flag:
            time.sleep(0.001)
            if not self.queues['state_dict'].empty():
                state_dict = self.queues['state_dict'].get()
                self.dqn.load_state_dict(state_dict)

            if (not self.queues['priorities'].empty()) and (not self.queues['indices'].empty()):
                priorities = self.queues['priorities'].get()
                indices = self.queues['indices'].get()
                self.update_priorities(indices, priorities)

            memories = []
            while not self.queues['memory'].empty():
                s, a, r, ns, d = self.queues['memory'].get()
                memories.append((s, a, r, ns, d))
            if len(memories) > 0:
                state, action, reward, next_state, done = zip(*memories)
                state = torch.stack(state)
                next_state = torch.stack(next_state)
                reward = torch.stack(reward)
                # 构造训练集
                transition_dict = {
                    'states': state,
                    'actions': action,
                    'next_states': next_state,
                    'rewards': reward,
                    'done': done,
                }
                td_errors = self.dqn.calculate_td_error(transition_dict)
                for memory, td_error in zip(memories, td_errors):
                    state, action, reward, next_state, done = memory
                    self._add_memory(state, action, reward, next_state, done, priority=td_error)

            if self.queues['samples'].empty() and len(self) > self.min_size:
                transition_dict, indices = self._sample()
                self.queues['samples'].put(transition_dict)
                self.queues['indices'].put(indices)

            if self.save_now:
                self._save_to_yaml()
                self.save_now = False

    def _get_priority_probs(self, priorities):
        scaled_priorities = np.array(priorities) ** self.alpha
        return scaled_priorities / scaled_priorities.sum()

    def _get_importance_weights(self, probs, total_size):
        # 计算重要性采样的权重
        weights = (1.0 / (total_size * probs)) ** self.beta
        # 正则化权重以避免数值不稳定
        weights /= weights.max()
        return weights

    def _add_memory(self, state, action, reward, next_state, done, priority=1.0):
        if len(self.buffer) >= self.capacity:
            # 基于最小优先级进行遗忘
            min_priority_idx = np.argmin(self.priorities)
            del self.buffer[min_priority_idx]
            del self.priorities[min_priority_idx]
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        self.now_experience = (state, action, reward, next_state, done)

    def add_memory(self, state, action, reward, next_state, done):
        self.queues['memory'].put((state, action, reward, next_state, done))

    def add_item(self, key: str, item):
        if self.queues[key].empty():
            self.queues[key].put(item)

    def _sample(self):
        # 获取缓冲区大小
        buffer_size = len(self.buffer)
        sample_size = min(buffer_size, self.batch_size)

        # 确保缓冲区不为空
        if buffer_size == 0:
            raise ValueError("Buffer is empty!")

        # 按照优先级进行加权随机抽样
        priority_probs = self._get_priority_probs(self.priorities)
        priority_probs = np.clip(priority_probs, 1e-5, 1.0)  # 将零值替换为非常小的正值
        priority_probs /= np.sum(priority_probs)  # 归一化概率
        indices = np.random.choice(buffer_size, sample_size, p=priority_probs, replace=False)

        # 根据采样到的索引获取经验
        sampled_experiences = [self.buffer[i] for i in indices]

        # 获取采样的经验（state, action, reward, next_state, done）的各个部分
        state, action, reward, next_state, done = zip(*sampled_experiences)

        # 获取采样的概率，用于计算重要性采样权重
        sampled_probs = priority_probs[indices]
        weights = self._get_importance_weights(sampled_probs, buffer_size)

        # 将状态和下一个状态转换为张量
        states = torch.stack(state)
        next_states = torch.stack(next_state)
        rewards = torch.stack(reward)

        transition_dict = {
            'states': states,
            'actions': action,
            'next_states': next_states,
            'rewards': rewards,
            'done': done,
            'weights': weights,
        }
        return transition_dict, indices

    def sample(self) -> dict:
        """return: transition_dict"""
        transition_dict = self.queues['samples'].get()
        return transition_dict

    def update_priorities(self, indices, new_priorities):
        # 更新优先级
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def _save_to_yaml(self):
        with open(self.file_name, 'w') as f:
            # buffer = convert_item(self.buffer)
            converted_structure = [tuple_to_dict(convert_item(item)) for item in self.buffer]
            yaml.dump(converted_structure, f, default_flow_style=None, sort_keys=False)
        print("Saving Finished!")

    def save_to_yaml(self, file_name):
        self.file_name = file_name
        self.save_now = True

    def __len__(self):
        return len(self.buffer)
