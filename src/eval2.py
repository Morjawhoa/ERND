import torch

from Environmnet import Env, create_agent
from RewardPlotter import RewardPlotter
from Agent import Agents
import csv
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

episodes = 10000
save_interval = 50
max_step = 200
device = torch.device('cuda')
start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

maze = np.loadtxt('../assets/global.txt', delimiter=',')
print(f"地图的尺寸为{maze.shape}")

if __name__ == '__main__':
    env = Env(maze, max_step)
    plotter = RewardPlotter(env)
    agent = create_agent('1', (11, 11), (40, 16), env)
    # agent.rndqn.load_state_dict(torch.load(f"../models/rnd1.pth"))

    familiar = np.zeros(maze.shape)
    q_values = np.zeros(maze.shape)

    # agent.load()
    agent.test()
    for row in range(maze.shape[0]):
        for column in range(maze.shape[1]):
            env.transport_agent(agent, row, column)
            env.refresh_chess(agent)
            state = agent.state.reshape(1, -1).to(device)
            q_values[row, column] = agent.dqn.get_q_value(state)[0][0].max().item()

    q_values = np.array(q_values)
    q_max = np.max(q_values)
    q_min = np.min(q_values)

    q_values /= (q_max - q_min)
    q_values += q_min

    # 创建深浅度地图
    plt.imshow(q_values, cmap='winter', interpolation='nearest')  # 使用灰度色图，值越大颜色越深
    plt.colorbar()  # 添加色条

    plt.show()
