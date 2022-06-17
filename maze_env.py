import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 20  # pixels
MAZE_H = 20  # grid height
MAZE_W = 50  # grid width
action_MAZE_W = 45  # grid width
left = 7

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = [1]
        self.n_actions = len(self.action_space) * action_MAZE_W
        self.n_features = 1
        self.sumPoint = MAZE_H * MAZE_W  # 要挖掘的总点数
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT + UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.space = np.ones([MAZE_W, MAZE_H], dtype=int)
        self.space_cnn = np.ones([1, MAZE_W, MAZE_H], dtype=int)
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        self.fill()
        # pack all
        self.canvas.pack()

    def fill(self):
        for i in range(0, MAZE_W * UNIT, UNIT):
            for j in range(0, MAZE_H * UNIT, UNIT):
                hell1_center = np.array([i, j])
                if i <= 4 * UNIT or j > 16 * UNIT:
                    self.canvas.create_rectangle(
                        hell1_center[0], hell1_center[1],
                        hell1_center[0] + UNIT, hell1_center[1] + UNIT,
                        fill='yellow')
                else:
                    self.canvas.create_rectangle(
                        hell1_center[0], hell1_center[1],
                        hell1_center[0] + UNIT, hell1_center[1] + UNIT,
                        fill='green')

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.fill()

        # return observation
        self.space = np.ones([MAZE_W, MAZE_H], dtype=int)
        s_ = self.observation()
        return s_

    def step(self, action):

        h = 0
        # 当上面一层挖完后开始挖下面一排
        while np.sum(self.space[5:, h]) == 0:
            h = h + 1

        # w = MAZE_W - 1
        w = action % action_MAZE_W + 5
        action = action // action_MAZE_W + 1
        # while self.space[w, h] == 0:
        #     w = w - 1
        # w = np.random.randint(5, w+1)
        reward1 = 0
        if action == 1:
            reward1 = self.action_min(w, h)
        elif action == 2:
            reward1 = self.action_mid(w, h)
        elif action == 3:
            reward1 = self.action_mid1(w, h)
        elif action == 4:
            reward1 = self.action_max(w, h)

        reward1 = (reward1 - 0.5) * 10
        s_ = self.observation()
        done = False
        if np.sum(self.space[5:, :17]) == 0:
            done = True
            reward1 = 100
        return s_, reward1, done

    def action_min(self, w, h):
        """
        挖掘形状
        #######
         #####
          ###
        """
        point = [[w - 6, h], [w - 5, h], [w - 4, h], [w - 3, h], [w - 2, h], [w - 1, h], [w, h],
                 [w - 5, h + 1], [w - 4, h + 1], [w - 3, h + 1], [w - 2, h + 1], [w - 1, h + 1],
                 [w - 4, h + 2], [w - 3, h + 2], [w - 2, h + 2]]
        for i in point:
            self.canvas.create_rectangle(
                i[0] * UNIT, i[1] * UNIT, i[0] * UNIT + UNIT, i[1] * UNIT + UNIT,
                fill='red')
        reward_action = 0
        reward_action += np.sum(self.space[w - 6:w + 1, h], axis=0)
        self.space[w - 6:w + 1, h] = 0
        reward_action += np.sum(self.space[w - 5:w, h + 1])
        self.space[w - 5:w, h + 1] = 0
        reward_action += np.sum(self.space[w - 4:w - 1, h + 2])
        self.space[w - 4:w - 1, h + 2] = 0
        reward_action = round(reward_action / len(point), 2)
        return reward_action

    def action_mid(self, w, h):
        """
        挖掘形状
        ######
         ####
         ###
         ##
        """
        point = [[w - 5, h], [w - 4, h], [w - 3, h], [w - 2, h], [w - 1, h], [w, h],
                 [w - 4, h + 1], [w - 3, h + 1], [w - 2, h + 1], [w - 1, h + 1],
                 [w - 4, h + 2], [w - 3, h + 2], [w - 2, h + 2],
                 [w - 4, h + 3], [w - 3, h + 3]]
        for i in point:
            self.canvas.create_rectangle(
                i[0] * UNIT, i[1] * UNIT, i[0] * UNIT + UNIT, i[1] * UNIT + UNIT,
                fill='red')
        reward = 0
        reward += np.sum(self.space[w - 5:w + 1, h], axis=0)
        self.space[w - 5:w + 1, h] = 0
        reward += np.sum(self.space[w - 4:w, h + 1], axis=0)
        self.space[w - 4:w, h + 1] = 0
        reward += np.sum(self.space[w - 4:w - 1, h + 2], axis=0)
        self.space[w - 4:w - 1, h + 2] = 0
        reward += np.sum(self.space[w - 4:w - 2, h + 3], axis=0)
        self.space[w - 4:w - 2, h + 3] = 0
        return round(reward / len(point), 2)

    def action_mid1(self, w, h):
        """
        挖掘形状
        ###
        ##
        #
        """
        point = [[w - 2, h], [w - 1, h], [w, h], [w - 1, h + 1], [w - 2, h + 1], [w - 2, h + 2]]
        for i in point:
            self.canvas.create_rectangle(
                i[0] * UNIT, i[1] * UNIT, i[0] * UNIT + UNIT, i[1] * UNIT + UNIT,
                fill='red')
        reward = 0
        reward += np.sum(self.space[w - 2:w + 1, h], axis=0)
        self.space[w - 2:w + 1, h] = 0
        reward += np.sum(self.space[w - 2:w, h + 1], axis=0)
        self.space[w - 2:w, h + 1] = 0
        reward += np.sum(self.space[w - 2:w - 1, h + 2], axis=0)
        self.space[w - 2:w - 1, h + 2] = 0
        return round(reward / 6, 2)

    def action_max(self, w, h):
        """
        挖掘形状
        #####
         ###
          #
        """
        point = [[w - 4, h], [w - 3, h], [w - 2, h], [w - 1, h], [w, h], [w - 3, h + 1], [w - 2, h + 1], [w - 1, h + 1],
                 [w - 2, h + 2]]
        for i in point:
            self.canvas.create_rectangle(
                i[0] * UNIT, i[1] * UNIT, i[0] * UNIT + UNIT, i[1] * UNIT + UNIT,
                fill='red')
        reward = 0
        reward += np.sum(self.space[w - 4:w + 1, h], axis=0)
        self.space[w - 4:w + 1, h] = 0
        reward += np.sum(self.space[w - 3:w, h + 1], axis=0)
        self.space[w - 3:w, h + 1] = 0
        reward += np.sum(self.space[w - 2:w, h + 2], axis=0)
        self.space[w - 2:w - 1, h + 2] = 0
        return round(reward / 9, 2)

    def render(self):
        time.sleep(0.01)
        self.update()

    def observation(self):
        '''
        # 将矩阵拼接成字符串
        row = self.space.shape[1]
        a_str = ''
        for i in range(row):
            a_str = a_str + (''.join(str(x) for x in self.space[:, i]))
        # 将字符串转换成十进制
        s_ = np.array([(int(a_str, 2))])
        '''
        # s_ = np.hstack(self.space)
        s_ = np.expand_dims(self.space, axis=0)
        return s_


if __name__ == '__main__':
    maz = Maze()
    reward = maz.step(1)
    print(reward)
    maz.mainloop()
