import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from RunningStats import RunningStats


def bonus_to_strange(bonus: torch.Tensor, boarder=0.001):
    bonus_scaled = torch.exp(-(bonus / boarder) ** 2)
    bonus_scaled[bonus_scaled < 0.01] = 0.0
    bonus_scaled[bonus_scaled > 0.99] = 1.0
    strange = 1 - bonus_scaled
    return strange.detach()


def q_value_to_familiar(bonus: torch.Tensor, boarder=0.001):
    bonus_scaled = torch.exp(-(bonus / boarder) ** 2)
    strange = 1 - bonus_scaled
    return strange.detach()


def strange_to_familiar(strange: torch.Tensor, boarder=3.0):
    familiar = 1.0 - torch.clamp(strange/boarder, min=0.0, max=1.0)
    return familiar


class RNDNetwork(nn.Module):
    """ RND 网络: 输入 state, 输出所有动作的特征向量 """

    def __init__(self, state_dim, action_dim, feature_dim):
        super(RNDNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim * feature_dim)  # 每个动作输出 feature_dim 维度的特征
        self.action_dim = action_dim
        self.feature_dim = feature_dim

    def forward(self, state):
        """ 输入 state, 输出 [batch_size, action_dim, feature_dim] 形状的特征 """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.action_dim, self.feature_dim)  # reshape 成 (batch, action_dim, feature_dim)


class RNDModule(nn.Module):
    """ RND 计算内在激励，并训练 predictor """
    def __init__(self, state_dim, action_dim, device, feature_dim=2, lr=1e-4):
        super(RNDModule, self).__init__()
        self.predictor = RNDNetwork(state_dim, action_dim, feature_dim).to(device)  # 预测网络
        self.target = RNDNetwork(state_dim, action_dim, feature_dim).to(device)  # 目标网络 (不训练)

        self.stats = RunningStats()

        # 目标网络参数固定
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        self.device = device

    def compute_rnd_reward(self, state):
        """ 计算某个状态下所有动作的 RND 内在奖励 """
        with torch.no_grad():
            target_features = self.target(state)  # (batch, action_dim, feature_dim)

        predicted_features = self.predictor(state)  # (batch, action_dim, feature_dim)

        # 计算所有动作的 L2 误差 (不再需要索引 action)
        bonus = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=2)  # shape: (batch, action_dim)

        self.stats.update_multi(bonus[0].tolist())

        bonus_scaled = 1 - bonus_to_strange(bonus)

        return bonus_scaled  # 直接返回所有动作的 RND 奖励

    def update(self, transition_dict):
        """ 训练 RND 预测网络 """
        self.optimizer.zero_grad()

        states = transition_dict["states"].to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        weights = torch.tensor(transition_dict["weights"], dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            target_features = self.target(states)  # (batch, action_dim, feature_dim)

        predicted_features = self.predictor(states)  # (batch, action_dim, feature_dim)

        # 确保 actions 形状正确: (batch, 1, 1)
        actions = actions.long().view(-1, 1, 1)  # 先转换成 LongTensor，并 reshape

        # 取出当前动作对应的特征向量
        target_action_features = target_features.gather(1, actions.expand(-1, -1, target_features.size(
            2)))  # (batch, 1, feature_dim)
        predicted_action_features = predicted_features.gather(1, actions.expand(-1, -1, predicted_features.size(
            2)))  # (batch, 1, feature_dim)

        # 计算 MSE 损失
        losses = F.mse_loss(predicted_action_features, target_action_features, reduction='none').mean(dim=2)
        loss = (weights * losses).mean()
        loss.backward()
        self.optimizer.step()

        return losses

    def reset(self):
        pass


class RNDDuelingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer=1):
        super().__init__()
        self.feature = nn.ModuleList()  # 用于提取共享特征的部分
        self.feature.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layer):
            self.feature.append(nn.Linear(hidden_size, hidden_size))

        # 用于计算 V(s) 的分支
        self.value = nn.Linear(hidden_size, 1)  # 状态价值函数输出为标量

        # 用于计算 A(s, a) 的分支
        self.advantage = nn.Linear(hidden_size, output_size)  # 优势函数输出为每个动作的值

    def forward(self, x):
        for layer in self.feature:
            x = F.leaky_relu(layer(x))  # 特征提取层

        # 计算状态价值 V(s)
        value = self.value(x)

        # 计算优势函数 A(s, a)
        advantage = self.advantage(x)

        # 将 V(s) 和 A(s, a) 合并为 Q(s, a)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # return q_value.pow(2)
        return F.sigmoid(q_value) * 20


class RNDoubleDuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gamma, update_steps, learning_rate, device):
        super().__init__()
        self.q_net = RNDDuelingNet(input_size, hidden_size, output_size).to(device)
        self.target_q_net = RNDDuelingNet(input_size, hidden_size, output_size).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 同步初始权重

        self.gamma = gamma
        self.device = device

        self.count = 0
        self.update_steps = update_steps

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def update(self, transitions):
        self._update(transitions)

        self.count += 1
        if self.count % self.update_steps == 0:
            self.update_target_network()

    def _update(self, transitions):
        """return: td_error"""
        states = transitions["states"].to(self.device)
        actions = torch.tensor(transitions["actions"]).view(-1, 1).to(self.device)
        rewards = transitions["rewards"].view(-1, 1).to(self.device)
        next_states = transitions["next_states"].to(self.device)
        done = torch.tensor(transitions["done"], dtype=torch.int64).view(-1, 1).to(self.device)
        weights = torch.tensor(transitions["weights"], dtype=torch.float).view(-1, 1).to(self.device)

        # 计算当前 Q 值
        q_value = self.q_net(states).gather(1, actions)

        # 使用主网络选择下一步动作
        next_actions = self.q_net(next_states).max(1)[1].view(-1, 1)  # 主网络选择动作

        # 使用目标网络计算下一个状态的 Q 值
        next_q_value = self.target_q_net(next_states).gather(1, next_actions).detach()  # 目标网络计算 Q 值

        # 计算目标 Q 值
        target_q_value = rewards + self.gamma * next_q_value * (1-done)

        # 计算 TD 误差，并用 weights 加权
        td_error = target_q_value - q_value
        losses = weights * td_error.pow(2)  # 加权的均方误差损失
        loss = losses.mean()

        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_q_value(self, state):
        return self.q_net(state)

    def update_target_network(self):
        # 定期将 q_net 的参数更新到 target_q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())


class RNDQNModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gamma, update_steps, learning_rate, device):
        super(RNDQNModule, self).__init__()
        self.gamma = 0.75

        self.rnd = RNDModule(input_size, output_size, device)
        self.dqn = RNDoubleDuelingDQN(input_size, hidden_size, output_size, self.gamma, update_steps, learning_rate, device)

    def update(self, transition):
        losses = self.rnd.update(transition)

        transition['rewards'] = bonus_to_strange(losses)

        self.dqn.update(transition)

    def get_familiar(self, state):
        q_value = self.dqn.get_q_value(state)
        multiplier = 1 / (1-self.gamma)
        # familiar = strange_to_familiar(q_value, boarder=3.0)
        familiar = 1 - q_value_to_familiar(q_value, boarder=0.4)
        return familiar, q_value

    def reset(self):
        self.rnd.reset()


class LossStranger(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gamma, update_steps, learning_rate, device):
        super(LossStranger, self).__init__()
        self.gamma = 0.5
        self.dqn = RNDoubleDuelingDQN(input_size, hidden_size, output_size, self.gamma, update_steps, learning_rate, device)

    def update(self, transition, losses):
        # print(losses[:4])
        transition['rewards'] = bonus_to_strange(losses)
        self._update(transition)

    def _update(self, transition):
        self.dqn.update(transition)

    def get_familiar(self, state):
        q_value = self.dqn.get_q_value(state)
        strange = bonus_to_strange(q_value, boarder=0.1)
        familiar = 1 - strange
        return familiar

    def reset(self):
        pass


