import tkinter as tk
import copy
from Agent import Agent
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageTk


class Barrier:
    def __repr__(self):
        return "Barrier()"  # 表示 Barrier 对象


class Destination:
    def __init__(self, name, x=None, y=None, env=None):
        self.name = name
        self.env = env  # 棋子持有棋盘引用
        self.x = x  # 棋子的x坐标，初始为空
        self.y = y  # 棋子的y坐标，初始为空

    def __repr__(self):
        return "Destination()"  # 表示 Barrier 对象

    def set_position(self, x, y):
        """更新棋子自身的坐标"""
        self.x, self.y = x, y


action_dict = {0: [0, 0],  # 停
               1: [1, 0],  # 下
               2: [-1, 0],  # 上
               3: [0, -1],  # 左
               4: [0, 1]}  # 右

# 定义颜色映射
color_map = {
    Barrier: 'black',  # 障碍物使用黑色
    Agent: 'cyan',  # 棋子使用蓝色
    Destination: 'yellow',  # 终点使用黄色
    list: 'white'  # 空白区域使用白色
}
# 定义形状映射，类似于 color_map
shape_map = {
    Agent: 'oval',  # Agent 用圆形
    Barrier: 'rectangle',  # 障碍物用方形
    Destination: 'rectangle',  # 终点用方形
    list: 'rectangle'  # 空白区域用方形
}

idx_map = {
    'boundary': -1,
    Agent: 2,  # Agent 用圆形
    Barrier: 1,  # 障碍物用方形
    Destination: 3,  # 终点用方形
    list: 0  # 空白区域用方形
}

colors = ['red', 'green', 'blue', 'yellow', 'black', 'white',
          'cyan', 'magenta', 'orange', 'purple', 'pink', 'gray', 'brown']


def convert_board_to_array(board):
    result = []
    for row in board:
        new_row = []
        for cell in row:
            if not cell:  # 如果是空列表
                new_row.append(0)
            elif isinstance(cell[0], Barrier):
                new_row.append(1)
            elif isinstance(cell[0], Agent):
                new_row.append(2)
            elif isinstance(cell[0], Destination):
                new_row.append(-1)
            else:
                new_row.append(0)  # 其他情况默认为空白
        result.append(new_row)
    return torch.Tensor(result)


class Screen(tk.Tk):
    def __init__(self, size, maze, grid):
        super().__init__()
        self.size = size
        self.row, self.column = maze.shape
        self.row_sep = int(min(960 / size[1], 960 / size[0]))
        self.column_sep = int(min(960 / size[1], 960 / size[0]))

        self.boat = Image.open("../assets/boat.ico")
        self.counter = 0

        self.widget_dict = {}

        self.focus_pos()
        self.create_widget(grid)

    def render(self, env_info):
        grid = env_info['grid']
        step_counter = env_info['step_counter']
        # 用于渲染环境
        self.counter += 1
        if self.counter % 2 == 0:
            self.generate_maze(grid)

        self.widget_dict[2].config(text=f"steps: {step_counter}", fg='black')
        self.update()

    def generate_maze(self, grid):
        board = self.widget_dict[1]  # 获取 Canvas 画布
        board.delete("all")  # 清除画布内容

        # 计算整张迷宫的像素宽高
        # 假设：self.grid 是一个 2D 列表，外层长度为行数，内层长度为列数
        total_cols = len(grid[0])  # 每一行的列数
        total_rows = len(grid)  # 总行数

        # 每个格子的宽高
        cell_w = self.row_sep
        cell_h = self.column_sep

        img_width = total_cols * cell_w
        img_height = total_rows * cell_h

        # 创建一张空白图 (RGB 模式，背景色 white)
        img = Image.new("RGB", (img_width, img_height), (223, 247, 247))
        draw = ImageDraw.Draw(img)

        # 可以根据自己需要设置字体，这里简单示例用一个默认字体
        # 如果要用 TTF 字体，可用 ImageFont.truetype("路径/字体.ttf", 字号)
        # font = ImageFont.load_default()

        # 遍历网格并在 PIL 图上绘制
        for row_idx, maze_row in enumerate(grid):
            for col_idx, items in enumerate(maze_row):
                # 计算当前格子在图像中的像素坐标
                start_x = col_idx * cell_w
                start_y = row_idx * cell_h
                end_x = start_x + cell_w
                end_y = start_y + cell_h

                # 先绘制格子背景（白底+灰色边框）
                # 你也可以不绘制“方块边框”，直接当成空白
                draw.rectangle([(start_x, start_y), (end_x, end_y)],
                               'white', outline='gray')

                # 每个格子里有若干 items，需要逐个决定它们的形状和颜色
                for item in items:
                    cell_type = type(item)
                    color = color_map.get(cell_type, 'white')  # 默认白色
                    shape = shape_map.get(cell_type, 'rectangle')  # 默认方形

                    # 如果有 alive 属性且为 False，变成紫色
                    if hasattr(item, 'alive') and not item.alive:
                        color = 'purple'

                    # 根据 shape 绘制图形
                    if shape == 'oval':
                        # PIL 的椭圆画法：draw.ellipse(左上角, 右下角)
                        draw.ellipse([(start_x, start_y), (end_x, end_y)],
                                     fill=color, outline='gray')
                    elif shape == 'boat':
                        boat_w, boat_h = 40, 40
                        boat = self.boat.resize((boat_w, boat_h))  # 缩放logo到100x100

                        # 将logo粘贴到图像的指定位置
                        logo_x = int((start_x + end_x - boat_w) // 2)
                        logo_y = int((start_y + end_y - boat_h) // 2)
                        img.paste(boat, (logo_x, logo_y), boat)  # 使用第三个参数以支持透明背景

                    else:
                        draw.rectangle([(start_x, start_y), (end_x, end_y)],
                                       fill=color, outline='gray')

                    # 如果是棋子(Chess)之类，带有 name 属性，就在中心写文字
                    if hasattr(item, 'name'):
                        text = str(item.name)
                        # 计算文字宽高
                        w, h = 20, 20
                        # 让文字居中
                        text_x = (start_x + end_x - w / 2) // 2
                        text_y = (start_y + end_y - h) // 2
                        font = ImageFont.truetype("times.ttf", size=w)  # 这里可以替换为你需要的字体路径和大小

                        draw.text((text_x, text_y), text, fill='magenta', font=font)

        # 所有格子都画完后，把 PIL 图转换成 Tkinter 可用的 PhotoImage
        self.tk_image = ImageTk.PhotoImage(img)
        # 在 Canvas 上一次性贴出整张图
        board.create_image(0, 0, anchor="nw", image=self.tk_image)
        # 注意：如果不保存 self.tk_image 引用，图片可能被垃圾回收导致不显示

    def focus_pos(self):
        # 设置窗口大小
        window_width = int((self.column + 2.4) * self.column_sep)
        window_height = int((self.row + 1.5) * self.row_sep)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def create_widget(self, grid):
        state_show_label = tk.Label(self, text="", fg="black")
        state_show_label.pack()
        self.widget_dict[2] = state_show_label

        width = self.column * self.column_sep
        height = self.row * self.row_sep
        board = tk.Canvas(self, width=width, height=height, bg='white')
        board.pack()
        self.widget_dict[1] = board
        self.generate_maze(grid)

    def close(self):
        self.destroy()

    def draw_path(self, path, color="red"):
        """
        在 Tkinter 的迷宫上绘制路径
        :param path: 由点 (x, y) 组成的路径列表
        :param color: 路径的颜色，默认为红色
        """
        board = self.widget_dict[1]  # 获取画布

        # 假设每个单元格的大小
        cell_width = board.winfo_width() // self.size[1]
        cell_height = board.winfo_height() // self.size[0]

        # 遍历路径，绘制每个点
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]

            # 计算坐标转换，将迷宫的点转换为画布上的坐标
            start_x = x1 * cell_width + cell_width // 2
            start_y = y1 * cell_height + cell_height // 2
            end_x = x2 * cell_width + cell_width // 2
            end_y = y2 * cell_height + cell_height // 2

            # 在 canvas 上绘制线条，表示路径
            board.create_line(start_x, start_y, end_x, end_y, fill=color, width=int(self.row_sep * 0.13))

        # 刷新画布
        self.update()


class Env:
    def __init__(self, maze, max_step):
        self.size = maze.shape
        self.row_sep = int(min(960 / self.size[1], 960 / self.size[0]))
        self.column_sep = int(min(960 / self.size[1], 960 / self.size[0]))
        # 初始化棋盘为空
        self.grid = [[[] if cell == 0 else [Barrier()] for cell in row] for row in maze]
        # 用于存储棋盘上的所有棋子及其位置信息
        self.agent_positions = {}
        self.dest_positions = {}
        self.maze = copy.copy(maze)
        self.row, self.column = maze.shape

        self.step_counter = 0
        self.max_step = max_step

        self.steps_total = 0

        self.boat = Image.open("../assets/boat.ico")

        self.screen = Screen(self.size, maze, self.grid[:])

    def place_agent(self, chess, x, y):
        # 将棋子放置在棋盘的指定位置
        if self.is_within_bounds(x, y):
            self.grid[x][y].append(chess)
            chess.env = self  # 让棋子知道它在哪个棋盘上
            self.agent_positions[chess] = (x, y)
            chess.set_position(x, y)  # 更新棋子自身的坐标
            chess.set_start_position(x, y)
        else:
            print(f"位置 ({x},{y}) 已有棋子或无效")

    def place_dest(self, dest, x, y):
        if self.is_within_bounds(x, y):
            self.grid[x][y].append(dest)
            dest.env = self  # 让棋子知道它在哪个棋盘上
            self.dest_positions[dest.name] = (x, y)
            dest.set_position(x, y)  # 更新棋子自身的坐标
        else:
            print(f"位置 ({x},{y}) 已有棋子或无效")

    def transport_agent(self, chess, x, y):
        current_x, current_y = self.agent_positions[chess]
        self.grid[current_x][current_y].remove(chess)  # 移除棋子原来的位置
        self.grid[x][y].append(chess)  # 更新棋子的位置到新坐标
        self.agent_positions[chess] = (x, y)

    def move_agent(self, chess, action):
        delta_x, delta_y = action_dict[action]
        current_x, current_y = self.agent_positions[chess]
        new_x, new_y = current_x + delta_x, current_y + delta_y
        # 在棋盘上更新棋子的位置
        if chess in self.agent_positions.keys() and self.is_within_bounds(current_x, current_y):
            if self.is_within_bounds(new_x, new_y):
                self.grid[current_x][current_y].remove(chess)  # 移除棋子原来的位置
                self.grid[new_x][new_y].append(chess)  # 更新棋子的位置到新坐标

                self.agent_positions[chess] = (new_x, new_y)

                self.refresh_chess(chess)  # 更新棋子的状态
            else:
                chess.alive = False
                self.refresh_chess(chess)
        else:
            print("棋子不在棋盘上，无法移动。")

    def refresh_chess(self, chess):
        x, y = self.agent_positions[chess]

        sight_size = chess.sight_size

        sight = convert_board_to_array(self.grid)
        sight = F.pad(sight, (sight_size, sight_size, sight_size, sight_size), value=3)
        sight = sight[x:x + 2 * sight_size + 1, y:y + 2 * sight_size + 1].reshape(-1).to(chess.device)

        for item in self.grid[x][y]:
            if isinstance(item, Barrier):
                chess.alive = False
            elif isinstance(item, Agent):
                if item.name != chess.name:
                    item.alive = False
                    chess.alive = False
            elif isinstance(item, Destination) and item.name == chess.name:
                chess.win = True
            else:
                chess.win = False

        chess.update_state(x, y, sight)

    def is_terminate(self):
        all_finished = all(chess.win for chess in self.agent_positions.keys())
        all_died = all((not chess.alive) for chess in self.agent_positions.keys())
        any_died = any((not chess.alive) for chess in self.agent_positions.keys())
        all_finished_or_died = all(((not chess.alive) or chess.win) for chess in self.agent_positions.keys())
        take_too_long = self.step_counter >= self.max_step
        return all_finished_or_died or take_too_long

    def is_successful(self):
        all_finished = all(chess.win for chess in self.agent_positions.keys())
        return all_finished

    def is_within_bounds(self, x, y):
        # 检查坐标是否在棋盘范围内
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]

    def reset(self):
        self.step_counter = 0

        for chess in self.agent_positions.keys():
            self.transport_agent(chess, chess.start_x, chess.start_y)
        for chess in self.agent_positions.keys():
            self.refresh_chess(chess)

    def refresh(self):
        env_info = {'grid': self.grid, 'step_counter': self.step_counter}
        self.screen.render(env_info)

        self.step_counter += 1
        self.steps_total += 1

    def batch_draw_path(self):
        path_info = {}
        for agent, color in zip(self.agent_positions.keys(), colors):
            self.screen.draw_path(agent.track, color)


def create_agent(name, xy, dest_xy, env):
    agent = Agent(name)
    destination = Destination(name)

    agent.place_env(env)
    agent.set_dest_position(dest_xy[0], dest_xy[1])
    env.place_agent(agent, xy[0], xy[1])
    env.place_dest(destination, dest_xy[0], dest_xy[1])

    env.refresh_chess(agent)
    return agent
