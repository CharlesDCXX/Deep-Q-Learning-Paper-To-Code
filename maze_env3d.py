import numpy as np
import time
import sys


class Maze:
    def __init__(self, MAZE_L, MAZE_W, MAZE_H):
        # 动作空间
        self.action_space = [1]
        # 挖掘位置
        self.n_actions = MAZE_L * MAZE_W
        # 特征数量
        self.n_features = 1
        # 使用nd-array模拟作为点云数据
        self.space_cnn = np.ones([MAZE_L, MAZE_W, MAZE_H], dtype=int)
        self.MAZE_L = MAZE_L
        self.MAZE_W = MAZE_W
        self.MAZE_H = MAZE_H

    # 每一次epsilon之后，将模拟环境重置
    def reset(self):
        self.update()
        self.space_cnn = np.ones([self.MAZE_L, self.MAZE_W, self.MAZE_H], dtype=int)
        s_ = self.observation()
        return s_

    def step(self, action):
        h = 0
        """
        ##########
        ##########
        L#########         L：MAZE_L
        ##########         W：MAZE_W
        ##--WW--##         &：挖掘机
        &
        """
        # 当上面一层挖完后开始挖下面一排
        while np.sum(self.space[:, :, h]) == 0:
            h = h + 1
        action_l = action % self.MAZE_L - 1 if action % self.MAZE_L != 0 else self.MAZE_L - 1
        action_w = action // self.MAZE_L if action % self.MAZE_L != 0 else action // self.MAZE_L - 1
        reward = 0
        if action_w / action_l < 0.577:
            reward += self.excavate(action_l, action_w, h)
            reward += self.excavate(action_l - 1, action_w, h)
            reward += self.excavate(action_l - 2, action_w, h)
            reward += self.excavate(action_l - 1, action_w, h + 1)
            reward += self.excavate(action_l - 2, action_w, h + 1)
            reward += self.excavate(action_l - 2, action_w, h + 2)
        elif action_w / action_l < 1.732:
            reward += self.excavate(action_l, action_w, h)
            reward += self.excavate(action_l - 1, action_w - 1, h)
            reward += self.excavate(action_l - 2, action_w - 2, h)
            reward += self.excavate(action_l - 1, action_w - 1, h + 1)
            reward += self.excavate(action_l - 2, action_w - 2, h + 1)
            reward += self.excavate(action_l - 2, action_w - 2, h + 2)
        else:
            reward += self.excavate(action_l, action_w, h)
            reward += self.excavate(action_l, action_w - 1, h)
            reward += self.excavate(action_l, action_w - 2, h)
            reward += self.excavate(action_l, action_w - 1, h + 1)
            reward += self.excavate(action_l, action_w - 2, h + 1)
            reward += self.excavate(action_l, action_w - 2, h + 2)

        s_ = self.observation()
        done = False
        if np.sum(self.space[:, :, :]) == 0:
            done = True
            reward = 100
        return s_, reward, done

    # def action_common(self, action_w, action_l):

    def observation(self):
        # s_ = np.expand_dims(self.space, axis=0)
        s_ = self.space_cnn
        return s_

    def excavate(self, x, y, z):
        try:
            self.space_cnn[x, y, z] = 0
        except IndexError:
            return 0
        return 1
