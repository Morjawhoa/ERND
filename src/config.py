from dataclasses import dataclass
import torch


@dataclass
class Config:
    sight_size: int = 2
    input_size: int = (2 * sight_size + 1) ** 2 + 4
    hidden_size: int = 512
    output_size: int = 5
    channel_size: int = 3

    learning_rate: float = 5e-5  # 5e-5
    gamma: list = 0.95
    update_steps: int = 30
    epsilon_decay: float = 0.995
    epsilon: float = 0.3
    min_epsilon: float = 0.0
    max_epsilon: float = 0.3

    capacity: int = 4000
    min_size: int = 4
    batch_size: int = 1024
    reward_explore_size: int = 10

    train: bool = True
    device: torch.device = torch.device("cuda")


# 创建Config实例
config_default = Config()

