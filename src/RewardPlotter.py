import time
import tkinter as tk


class RewardPlotter:
    def __init__(self, parent, max_reward=5000, width=500, height=400):
        # 使用 Toplevel 而不是 Tk 来创建新窗口
        self.root = tk.Toplevel(parent.screen)
        self.root.title("Reward Plot")

        self.width = width
        self.height = height
        self.max_reward = max_reward  # 奖励的最大值，用于归一化

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='white')
        self.canvas.pack()

        # 初始化数据
        self.rewards_list = [[] for _ in range(4)]  # 存储四个不同奖励列表
        self.plot_padding = 20  # 边距
        self.max_points = self.width - 2 * self.plot_padding  # x 轴最大可画的点数

        # 颜色列表，用于四条曲线
        self.colors = ['blue', 'green', 'red', 'purple']

    def add_rewards(self, rewards):
        """添加新的四个 reward 列表"""
        for i in range(len(rewards)):
            self.rewards_list[i].append(rewards[i])
            if len(self.rewards_list[i]) > self.max_points / 2:
                self.rewards_list[i].pop(0)  # 保持每个列表只显示最近的 max_points 个奖励

    def update_plot(self):
        """更新奖励折线图"""
        self.canvas.delete("all")  # 清空画布

        # 画 x 和 y 轴
        self.canvas.create_line(self.plot_padding, self.height - self.plot_padding,
                                self.width - self.plot_padding, self.height - self.plot_padding, arrow=tk.LAST)  # x 轴
        self.canvas.create_line(self.plot_padding, self.height - self.plot_padding,
                                self.plot_padding, self.plot_padding, arrow=tk.LAST)  # y 轴

        for idx, rewards in enumerate(self.rewards_list):
            if len(rewards) < 2:
                continue  # 少于两个点无法画折线

            # 假设 reward 的范围是 [-self.max_reward, self.max_reward]
            y_mid = self.height / 2  # 中间点，表示 reward = 0 的位置
            y_range = self.max_reward * 2  # y 轴的范围是从 -max_reward 到 max_reward

            # 计算每个 reward 的位置 (归一化 reward 以适应画布)
            x_step = (self.width - 2 * self.plot_padding) / (len(rewards) - 1)
            points = []

            for i, reward in enumerate(rewards):
                x = self.plot_padding + i * x_step
                y = y_mid - (reward / y_range) * (self.height - 2 * self.plot_padding)  # 根据 -max_reward 到 max_reward 范围归一化
                points.append((x, y))

            # 绘制每条曲线
            for i in range(1, len(points)):
                x1, y1 = points[i - 1]
                x2, y2 = points[i]
                self.canvas.create_line(x1, y1, x2, y2, fill=self.colors[idx], width=2)

    def render(self):
        """手动刷新图表"""
        self.update_plot()
        self.root.update()  # 更新窗口内容，保持 Tkinter 主事件循环响应
