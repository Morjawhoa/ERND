from Environmnet import Env, create_agent
from RewardPlotter import RewardPlotter
from Agent import Agents
import csv
import numpy as np
from datetime import datetime
from tqdm import tqdm

episodes = 10000
save_interval = 50
max_step = 200

start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

maze = np.loadtxt('../assets/global.txt', delimiter=',')
print(f"地图的尺寸为{maze.shape}")

if __name__ == '__main__':
    env = Env(maze, max_step)
    plotter = RewardPlotter(env)
    agents = Agents(
                    create_agent('1', (11, 11), (40, 16), env),
                    create_agent('2', (30, 30), (11, 11), env),
                    create_agent('3', (8, 32), (30, 30), env),
                    create_agent('4', (40, 16), (8, 32), env),
                    )
    # agents.batch_load()
    # agents.batch_test()

    rewards_csv = []
    steps_csv = []

    win_streak = 0  # 连胜次数

    for episode in range(episodes):
        if env.is_successful():
            win_streak += 1
        else:
            win_streak = 0
        if win_streak > 0:
            break

        rewards_csv.append(agents.rewards())
        steps_csv.append([env.step_counter])

        plotter.add_rewards(agents.rewards())
        plotter.render()

        agents.batch_start()
        env.reset()

        pbar = tqdm(total=max_step, desc=str(episode))
        while not env.is_terminate():
            agents.batch_combo()

            env.screen.update()
            env.refresh()

            pbar.update(1)
            # 更新进度条
            pbar.set_postfix({
                # 'rewards': agents.rewards(),
                # 'epsilon': ["{:.2f}".format(agent.epsilon) for agent in agents.agents.values()],
                # 'alive': [agent.alive for agent in agents.agents.values()],
                'q_value': np.round(agents.agents['1'].q_value.cpu().detach().numpy(), 2),
                'familiar': np.round(agents.agents['1'].familiar.cpu().detach().numpy(), 2),
                'q_strange': np.round(agents.agents['1'].strange_prob.cpu().detach().numpy(), 2),
                'steps': env.steps_total,
                'learned': agents.agents['1'].learned_total,
                'satisfy': agents.agents['1'].satisfy,
                })
        del pbar

        if (episode + 1) % save_interval == 0:
            agents.batch_save()
            # agents.batch_save_memory()

            # 将列表保存为 CSV 文件
            with open(f'../log/rewards/{start_time}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                # 写入列表的每一行
                writer.writerows(rewards_csv)
                rewards_csv = []
            with open(f'../log/steps/{start_time}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                # 写入列表的每一行
                writer.writerows(steps_csv)
                steps_csv = []

    env.refresh()
    env.batch_draw_path()

    env.screen.mainloop()
