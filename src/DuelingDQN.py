import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNet(nn.Module):
    def __init__(self, input_size, hidden_size, action_dim, channel_size, hidden_layer=1):
        super().__init__()
        self.channel_size = channel_size
        self.action_dim = action_dim
        self.feature = nn.ModuleList()  # 用于提取共享特征的部分
        self.feature.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layer):
            self.feature.append(nn.Linear(hidden_size, hidden_size))

        # 用于计算 V(s) 的分支
        self.value = nn.Linear(hidden_size, 1 * channel_size)  # 状态价值函数输出为标量

        # 用于计算 A(s, a) 的分支
        self.advantage = nn.Linear(hidden_size, action_dim * channel_size)  # 优势函数输出为每个动作的值

    def forward(self, x):
        for layer in self.feature:
            x = F.leaky_relu(layer(x))  # 特征提取层

        # 计算状态价值 V(s)
        value = self.value(x).view(-1, 1, self.channel_size)

        # 计算优势函数 A(s, a)
        advantage = self.advantage(x).view(-1, self.action_dim, self.channel_size)

        # 将 V(s) 和 A(s, a) 合并为 Q(s, a)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_value


class DoubleDuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, channel_size, gamma, update_steps, device, learning_rate):
        super().__init__()
        self.q_net = DuelingNet(input_size, hidden_size, output_size, channel_size).to(device)
        self.target_q_net = DuelingNet(input_size, hidden_size, output_size, channel_size).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 同步初始权重

        self.channel_size = channel_size
        self.gamma = gamma
        self.device = device

        self.steps = 0
        self.update_steps = update_steps

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def update(self, transitions):
        self.steps += 1
        if self.steps % self.update_steps == 0:
            self.update_target_network()

        return self._update(transitions)

    def _update(self, transitions):
        """return: td_error"""
        states = transitions["states"].to(self.device)
        actions = torch.tensor(transitions["actions"]).view(-1, 1, 1).repeat(1, 1, self.channel_size).to(self.device)
        rewards = transitions["rewards"].view(-1, 1, self.channel_size).to(self.device)
        next_states = transitions["next_states"].to(self.device)
        done = torch.tensor(transitions["done"], dtype=torch.int64).view(-1, 1, 1).repeat(1, 1, self.channel_size).to(
            self.device)
        weights = torch.tensor(transitions["weights"], dtype=torch.float).view(-1, 1, 1).repeat(1, 1,
                                                                                                self.channel_size).to(
            self.device)
        gamma = self.gamma.view(1, 1, -1).to(self.device)

        # 计算当前 Q 值
        q_value = self.q_net(states).gather(1, actions)

        # 使用主网络选择下一步动作
        next_actions = self.q_net(next_states).sum(dim=-1).max(1)[1].view(-1, 1, 1).repeat(1, 1,
                                                                                           self.channel_size)  # 主网络选择动作

        # 使用目标网络计算下一个状态的 Q 值
        next_q_value = self.target_q_net(next_states).gather(1, next_actions).detach()  # 目标网络计算 Q 值

        # 计算目标 Q 值
        target_q_value = rewards + gamma * next_q_value * (1 - done)

        # 计算 TD 误差，并用 weights 加权
        td_error = target_q_value - q_value
        losses = td_error.pow(2).sum(dim=-1)  # 加权的均方误差损失
        loss = (weights * losses).mean()

        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return torch.abs(td_error).sum(dim=-1).detach().reshape(-1).cpu().tolist(), losses

    def calculate_td_error(self, transitions):
        """return: td_error"""
        states = transitions["states"].to(self.device)
        actions = torch.tensor(transitions["actions"]).view(-1, 1, 1).repeat(1, 1, self.channel_size).to(self.device)
        rewards = transitions["rewards"].view(-1, 1, self.channel_size).to(self.device)
        next_states = transitions["next_states"].to(self.device)
        done = torch.tensor(transitions["done"], dtype=torch.int64).view(-1, 1).to(self.device)

        # 计算当前 Q 值
        q_value = self.q_net(states)
        # print(states.shape, actions.shape, rewards.shape, next_states.shape, done.shape, q_value.shape, '\n')

        q_value = q_value.gather(1, actions)

        # 使用主网络选择下一步动作
        next_actions = self.q_net(next_states).sum(dim=-1).max(1)[1].view(-1, 1, 1).repeat(1, 1,
                                                                                           self.channel_size)  # 主网络选择动作

        # 使用目标网络计算下一个状态的 Q 值
        next_q_value = self.target_q_net(next_states).gather(1, next_actions).detach()  # 目标网络计算 Q 值

        # 计算目标 Q 值
        target_q_value = rewards + self.gamma * next_q_value * (1 - done)

        # 计算 TD 误差，并用 weights 加权
        td_errors = torch.abs((q_value - target_q_value).sum(dim=-1)).detach().reshape(-1).cpu().tolist()
        return td_errors

    def get_q_value(self, state):
        return self.q_net(state)

    def update_target_network(self):
        # 定期将 q_net 的参数更新到 target_q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())


if __name__ == '__main__':
    input_size = 29
    hidden_size = 256
    action_dim = 5
    channel_size = 3

    dqn = DuelingNet(input_size, hidden_size, action_dim, channel_size)
    state = torch.zeros((10, input_size))

    output = dqn(state)

    print(output.shape)
