import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 20  # pixels
MAZE_H = 12  # grid height
MAZE_W = 50  # grid width
action_MAZE_W = 45  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = [1, 2, 3, 4]
        self.n_actions = len(self.action_space) * action_MAZE_W
        self.n_features = 1

        self.title('maze')
        self.geometry('{0}x{1}'.format(1000, 1000))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=1000,
                                width=1000)
        # create origin
        self.fill()

        # pack all
        self.canvas.pack()

    def fill(self):
        a = self.canvas.create_rectangle(
            520, 20,
            550, 50,
            fill='yellow')
        self.canvas.coords(a, 520, 20,
                           550, 50, )
        self.canvas.create_rectangle(
            200, 200,
            230, 230,
            fill='red')
        self.canvas.create_rectangle(
            300, 300,
            350, 350,
            fill='red')
        self.canvas.create_rectangle(
            600, 300,
            650, 350,
            fill='red')
        self.canvas.create_oval(
            200, 800,
            250, 850,
            fill='yellow')

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.fill()

        # return observation
        s_ = self.observation()
        return s_

    def step(self, action):
        reward1 = 0
        s_ = self.observation()
        done = False

        return s_, reward1, done

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
        s_ = 0
        return s_


if __name__ == '__main__':
    maz = Maze()
    reward = maz.step(1)
    print(reward)
    maz.mainloop()
