import torch.nn as nn
import torch.nn.functional as F
import torch
import random


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layer):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for idx, net in enumerate(self.layers):
            if idx == len(self.layers) - 1:
                return net(x)
            else:
                x = F.leaky_relu(net(x))
        return x


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gamma, device, learning_rate):
        super().__init__()
        self.q_net = Net(input_size, hidden_size, output_size).to(device)
        self.target_q_net = Net(input_size, hidden_size, output_size).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.gamma = gamma
        self.device = device

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def update(self, transitions):
        states = transitions["states"].to(self.device)
        actions = torch.tensor(transitions["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = transitions["next_states"].to(self.device)
        alive = torch.tensor(transitions["done"], dtype=torch.int64).view(-1, 1).to(self.device)

        q_value = self.q_net(states).gather(1, actions)
        next_q_value = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # next_q_value = self.target_q_net(next_states).mean(1)[0].view(-1, 1)
        target_q_value = next_q_value * self.gamma * alive + rewards

        loss = F.mse_loss(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_q_value(self, state):
        return self.q_net(state)


class DoubleDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gamma, device, learning_rate):
        super().__init__()
        self.q_net = Net(input_size, hidden_size, output_size).to(device)
        self.target_q_net = Net(input_size, hidden_size, output_size).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.gamma = gamma
        self.device = device

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def update(self, transitions):
        states = transitions["states"].to(self.device)
        actions = torch.tensor(transitions["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = transitions["next_states"].to(self.device)
        alive = torch.tensor(transitions["done"], dtype=torch.int64).view(-1, 1).to(self.device)

        # 计算当前 Q 值
        q_value = self.q_net(states).gather(1, actions)

        # 使用主网络选择下一步动作
        next_actions = self.q_net(next_states).max(1)[1].view(-1, 1)  # 主网络选择动作

        # 使用目标网络计算下一个状态的 Q 值
        next_q_value = self.target_q_net(next_states).gather(1, next_actions).detach()  # 目标网络计算 Q 值

        # 计算目标 Q 值
        target_q_value = rewards + self.gamma * next_q_value * alive

        # 计算损失并更新网络
        loss = F.mse_loss(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_q_value(self, state):
        return self.q_net(state)

