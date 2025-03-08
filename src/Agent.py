import random
import time
from DuelingDQN import DoubleDuelingDQN
import torch
from Record import PrioritizedReplayBuffer
import threading
import queue
from RND import RNDQNModule, LossStranger
from config import config_default
import numpy as np


def normalize_rewards(rewards):
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / std


class Agent:
    def __init__(self, name, config=config_default):
        self.name = name
        self.env = None  # 棋子持有棋盘引用
        self.start_x = None
        self.start_y = None
        self.x = None  # 棋子的x坐标，初始为空
        self.y = None  # 棋子的y坐标，初始为空
        self.dest_x = None
        self.dest_y = None
        # 探索参数
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.min_epsilon = config.min_epsilon
        self.max_epsilon = config.max_epsilon
        # 网络参数
        self.sight_size = config.sight_size
        input_size = config.input_size
        hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.channel_size = config.channel_size
        # 训练参数
        self.device = config.device
        learning_rate = config.learning_rate
        gamma = torch.tensor([0.95, 0.95, 0.0], dtype=torch.float64, device=self.device)
        capacity = config.capacity
        self.min_size = config.min_size
        self.batch_size = config.batch_size
        self.reward_explore_size = config.reward_explore_size
        self.update_steps = config.update_steps
        # DQN
        self.dqn = DoubleDuelingDQN(input_size, hidden_size, self.output_size, self.channel_size, gamma,
                                    self.update_steps, self.device,
                                    learning_rate)
        self.learner_dqn = DoubleDuelingDQN(input_size, hidden_size, self.output_size, self.channel_size, gamma,
                                            self.update_steps,
                                            self.device, learning_rate)
        self.learner_dqn.load_state_dict(self.dqn.state_dict())
        # RND
        self.rndqn = RNDQNModule(input_size, hidden_size, self.output_size, gamma, self.update_steps, learning_rate,
                                 self.device)
        self.learner_rndqn = RNDQNModule(input_size, hidden_size, self.output_size, gamma, self.update_steps,
                                         learning_rate, self.device)
        self.learner_rndqn.load_state_dict(self.rndqn.state_dict())

        self.memory = PrioritizedReplayBuffer(capacity, input_size, hidden_size, self.output_size, self.channel_size,
                                              gamma, self.update_steps, self.device,
                                              learning_rate, self.batch_size, self.min_size)
        # 记忆参数
        self.sight = None
        self.last_state = None
        self.state = None
        self.action = 0
        self.alive = True
        self.win = False
        self.reward = 0
        # 传输信道
        self.queues = {'dqn': queue.Queue(),
                       'rnd': queue.Queue()}
        # 统计参数
        self.q_value = None
        self.familiar = None
        self.strange_prob = None
        self.value_prob = None
        self.action_mode = 'exploit'
        self.accu_reward = 0
        self.learner_count = 0
        self.track = []
        self.learned_num = 0
        self.learned_total = 0
        self.losses = torch.zeros((5, 1))
        self.satisfy = 0.0
        # 运行参数
        self.flag = True
        self.save_now = False
        self.train = config.train
        # 学习器
        self.leaner = threading.Thread(target=self.auto_learn)
        if self.train:
            self.leaner.start()

    def auto_learn(self):
        dqn_state_dict = self.learner_dqn.state_dict()
        rnd_state_dict = self.learner_rndqn.state_dict()
        while self.flag and self.train:
            time.sleep(0.005)
            if len(self.memory) > self.min_size:
                transition_dict = self.memory.sample()
                # 训练
                td_error, losses = self.learner_dqn.update(transition_dict)
                self.learner_rndqn.update(transition_dict)
                # 拷贝
                dqn_state_dict = self.learner_dqn.state_dict()
                rnd_state_dict = self.learner_rndqn.state_dict()
                # 装填
                self.memory.add_item('priorities', td_error)
                self.memory.add_item('state_dict', dqn_state_dict)
                self.add_item('dqn', dqn_state_dict)
                self.add_item('rnd', rnd_state_dict)
                # 同步
                self.losses = losses
                self.learner_count += 1

            # 保存
            if self.save_now:
                torch.save(dqn_state_dict, f"../models/{self.name}.pth")
                torch.save(rnd_state_dict, f"../models/rnd{self.name}.pth")
                self.save_now = False

    def teach(self):
        try:
            dqn_state_dict = self.queues['dqn'].get_nowait()
            rndqn_state_dict = self.queues['rnd'].get_nowait()
            self.dqn.load_state_dict(dqn_state_dict)
            self.rndqn.load_state_dict(rndqn_state_dict)
            self.learned_num += 1
            self.learned_total += 1
        except queue.Empty:
            pass

    def formulate_state(self):
        self.state = torch.cat((self.sight,
                                torch.tensor([self.x, self.y], dtype=torch.float, device=self.device),
                                torch.tensor([self.dest_x, self.dest_y], dtype=torch.float, device=self.device)),
                               dim=-1).detach()
        return self.state

    def choose_action(self):
        return self._choose_action_dodge()

    def _choose_action_default(self):
        state = self.state.reshape(1, -1).to(self.device)
        q_value = self.dqn.get_q_value(state)
        q_integrate = q_value.sum(dim=-1)

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(5))
            self.action_mode = 'random'
        elif torch.all(torch.abs(q_integrate - q_integrate[0][0]) < 1e-5).item():
            action = random.choice(range(5))
            self.action_mode = 'random'
        else:
            action = q_integrate.max(1)[1].item()
            self.action_mode = 'exploit'

        self.q_value = q_integrate
        self.familiar = q_integrate
        self.value_prob = q_integrate
        self.strange_prob = q_integrate

        return action, action, action

    def _choose_action_dodge(self):
        state = self.state.reshape(1, -1).to(self.device)

        q_value = self.dqn.get_q_value(state)
        familiar, q_familiar = self.rndqn.get_familiar(state)

        q_sum = q_value.sum(dim=-1)
        q_danger = q_value[..., 2]
        q_integrate = q_danger * familiar

        if random.uniform(0, 1) < self.epsilon:
            # 找出大于 -1000 的元素的索引
            indices = torch.nonzero(q_integrate[0] > -800).squeeze(-1)
            # 随机选择一个索引
            action = random.choice(indices.tolist())
            self.action_mode = 'random'
        elif torch.all(torch.abs(q_sum - q_sum[0][0]) < 1e-5).item():
            action = random.choice(range(5))
            self.action_mode = 'random'
        else:
            action = q_sum.max(1)[1].item()
            self.action_mode = 'exploit'

        self.q_value = q_sum
        self.familiar = familiar
        self.value_prob = q_value[0, :, 0]
        self.strange_prob = q_familiar

        return action, action, action

    def _choose_action_new(self):
        state = self.state.reshape(1, -1).to(self.device)

        q_value = self.dqn.get_q_value(state)
        q_sum = q_value.sum(dim=-1)
        familiar, q_familiar = self.rndqn.get_familiar(state)

        satisfied_q = self.satisfy
        total = (q_sum - satisfied_q) * familiar
        total = (total // 0.1) * 0.1

        max_value = total.max(dim=1)[0]
        max_indices = (total == max_value).nonzero()
        action = max_indices[torch.randint(len(max_indices), (1,))][0][1].item()

        action_values = q_sum.max(1)[1].item()
        action_strange = familiar.max(1)[1].item()

        self.q_value = q_sum
        self.familiar = familiar
        self.value_prob = q_value[0, :, 0]
        self.strange_prob = q_familiar
        return action, action_values, action_strange

    def _choose_action_exception(self):
        state = self.state.reshape(1, -1).to(self.device)

        q_value = self.dqn.get_q_value(state)
        familiar, q_familiar = self.rndqn.get_familiar(state)

        q_sum = q_value.sum(dim=-1)
        q_danger = q_value[..., 2]
        q_integrate = q_danger * familiar

        if random.uniform(0, 1) < self.epsilon:
            # 找出大于 -1000 的元素的索引
            indices = torch.nonzero(q_integrate[0] > -800).squeeze(-1)
            # 随机选择一个索引
            action = random.choice(indices.tolist())
            self.action_mode = 'random'
        elif torch.all(torch.abs(q_sum - q_sum[0][0]) < 1e-5).item():
            action = random.choice(range(5))
            self.action_mode = 'random'
        else:
            satisfied_q = self.satisfy
            total = (q_sum - satisfied_q) * familiar
            total = (total // 0.1) * 0.1

            max_value = total.max(dim=1)[0]
            max_indices = (total == max_value).nonzero()
            action = max_indices[torch.randint(len(max_indices), (1,))][0][1].item()
            self.action_mode = 'exploit'

        self.q_value = q_sum
        self.familiar = familiar
        self.value_prob = q_value[0, :, 0]
        self.strange_prob = q_familiar

        return action, action, action

    def update_state(self, x, y, sight):
        self.x = x
        self.y = y
        self.sight = sight

        self.last_state = self.state
        self.state = self.formulate_state()
        self.track.append((x, y))

        reward = [2000.0 if self.win else 0.0,  # 胜利奖励
                  (-100.0 if self.action == 0 else 0.0)  # 静止惩罚
                  - (((x - self.dest_x) ** 2 + (y - self.dest_y) ** 2) ** 0.5) * 0.4,  # 距离惩罚
                  -1000.0 if not self.alive else 0.0]  # 死亡惩罚

        self.reward = torch.tensor(reward)
        self.accu_reward += self.reward.sum().detach().item()

    def move(self, action):
        # 通过棋盘来处理移动
        if self.env:
            self.env.move_agent(self, action)
        else:
            print("棋子未放置在棋盘上，无法移动。")

    def combo(self, delay=0):
        if self.alive and not self.win:
            action, action_values, action_strange = self.choose_action()
            self.action = action

            self.move(action)
            self.memorize()

        if self.train:
            self.teach()

        time.sleep(delay)

    def memorize(self):
        if all(x is not None for x in [self.last_state, self.action, self.reward, self.state, self.alive, self.win]):
            done = (not self.alive) or self.win
            self.memory.add_memory(self.last_state, self.action, self.reward, self.state, done)

    def add_item(self, key, item):
        if self.queues[key].empty():
            self.queues[key].put(item)

    def place_env(self, env):
        self.env = env

    def set_position(self, x, y):
        """更新棋子自身的坐标"""
        self.x, self.y = x, y
        self.track.append((x, y))

    def set_start_position(self, start_x, start_y):
        self.start_x, self.start_y = start_x, start_y

    def set_dest_position(self, dest_x, dest_y):
        self.dest_x, self.dest_y = dest_x, dest_y

    def __repr__(self):
        return f"Agent({self.name}, x={self.x}, y={self.y})"

    def stop(self):
        self.flag = False

    def reset(self):
        self.last_state = None
        self.action = None
        self.reward = None
        self.state = None
        self.alive = True
        self.win = False

        self.accu_reward = 0
        self.track = []
        self.learned_num = 0

        self.set_position(self.start_x, self.start_y)

        self.learner_rndqn.reset()

    def test(self):
        self.epsilon = 0
        self.dqn.q_net.eval()
        self.learner_dqn.eval()
        self.memory.dqn.eval()
        return self

    def load(self):
        self.dqn.load_state_dict(torch.load(f"../models/{self.name}.pth"))
        self.learner_dqn.load_state_dict(torch.load(f"../models/{self.name}.pth"))
        self.memory.dqn.load_state_dict(torch.load(f"../models/{self.name}.pth"))
        return self

    def save(self):
        self.save_now = True

    def save_memory(self):
        self.memory.save_to_yaml(f"../memory/{self.name}.yaml")

    def explore_old(self, delta=0.10):
        if self.win:
            self.epsilon -= delta
            self.satisfy = 2000.0
        else:
            self.epsilon += delta
        self.epsilon = np.clip(self.epsilon, self.min_epsilon, self.max_epsilon)

    def start(self):
        self.explore_old()
        self.reset()


class Agents:
    def __init__(self, *args: Agent):
        self.agents = {}
        for agent in args:
            self.agents[agent.name] = agent

    def batch_combo(self, delay=0):
        for agent in self.agents.values():
            agent.combo(delay=delay)
        return self

    def batch_load(self):
        for agent in self.agents.values():
            agent.load()
        return self

    def batch_save(self):
        for agent in self.agents.values():
            agent.save()
        return self

    def batch_save_memory(self):
        for agent in self.agents.values():
            agent.save_memory()
        return self

    def batch_stop(self):
        for agent in self.agents.values():
            agent.stop()
        return self

    def batch_test(self):
        for agent in self.agents.values():
            agent.test()
        return self

    def batch_start(self):
        for agent in self.agents.values():
            agent.start()
        return self

    def rewards(self):
        return [int(agent.accu_reward) for agent in self.agents.values()]

    def epsilons(self):
        return [agent.epsilon for agent in self.agents.values()]
