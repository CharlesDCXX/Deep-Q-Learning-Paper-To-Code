# -*- coding: utf-8 -*

import numpy as np
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt
import time


def create_data():
    np_save = np.ones((train_size, point_size, point_size), dtype=int)
    for i in range(0, train_size):
        # 生成随机地图
        np_save[i, :, :] = np.random.randint(min_depth, max_depth, (point_size, point_size))

    # np.save("test_x",np_save)
    goal = [
        [0, 0], [0, 50], [50, 0], [50, 50],
        [4]
    ]
    goal1 = [
        [0, 0], [0, 2], [2, 0], [2, 2],
        [4]
    ]
    return np_save


def transition_data(np_save):
    # 任务深度
    task_depth = 6
    # 单元格 长和宽
    cell_x = 5
    cell_y = 5
    # tile 索引
    index_x = 0
    index_y = 0
    # 将实际地图信息转化为0，1二进制信息
    tile_space = np.where(np_save > task_depth, 1, 0)
    # np.sum(tile_space[0:cell_x,0:cell_y])
    tile_grid = np.zeros([train_size, int(point_size / cell_x), int(point_size / cell_y)], dtype=int)
    summ = 0
    # 将tile_grid按照一定规则转换
    for t in range(0, train_size):
        for i in range(0, len(tile_grid[0])):
            for j in range(0, len(tile_grid[1])):
                tile_grid[t, i, j] = 1 if np.sum(
                    tile_space[t, i * cell_x:i * cell_x + cell_x, j * cell_y:j * cell_y + cell_y]) > 0 else 0
                if tile_grid[t, i, j] == 0:
                    summ = summ + 1
    np.save("test_x", tile_grid)
    print(summ)


def create_test():
    action_space_exc = 0
    action_space_up = 1
    action_space_down = 2
    action_space_left = 3
    action_space_right = 4

    # 读取tile_grid
    tile_grid = np.load('test_x.npy')
    # 下一个动作标签
    action_list_space = []
    # 历史动作
    history_action_list = []
    chopped_map = tile_grid[0].copy()
    # 建立一个numpy shape(n,32,32)
    chopped_map1 = np.ones((20000, 32, 32), dtype=int)
    # TEM测试集
    index_chop = 0
    length_x = len(chopped_map)
    length_y = len(chopped_map[0])
    for i in range(0, train_size):
        index_x = 0
        index_y = 0
        action_list = []
        direct_bool = True
        while index_x < len(chopped_map) and index_y < len(chopped_map[0]):
            if chopped_map[index_x, index_y] != 0:
                # 追加历史动作
                history_action_list.append(action_list[:])
                # 将动作加入动作标签
                action_list_space.append(action_space_exc)
                # 将地图加入chopped
                chopped_map1[index_chop] = chopped_map
                index_chop = index_chop + 1
                # 将此坐标点化为 0
                chopped_map[index_x, index_y] = 0
                action_list.append(action_space_exc)

            if direct_bool:
                if index_x < length_y - 1:
                    # 追加历史动作
                    history_action_list.append(action_list[:])
                    # 将动作加入动作标签
                    action_list_space.append(action_space_right)
                    # 将地图加入chopped
                    chopped_map1[index_chop] = chopped_map
                    index_chop = index_chop + 1

                    action_list.append(action_space_right)
                    index_x = index_x + 1
                else:
                    # 追加历史动作
                    history_action_list.append(action_list[:])
                    # 将动作加入动作标签
                    action_list_space.append(action_space_down)
                    # 将地图加入chopped
                    chopped_map1[index_chop] = chopped_map
                    index_chop = index_chop + 1

                    action_list.append(action_space_down)
                    index_y = index_y + 1
                    direct_bool = bool(1 - direct_bool)
            else:
                if index_x > 0:
                    # 追加历史动作
                    history_action_list.append(action_list[:])
                    # 将动作加入动作标签
                    action_list_space.append(action_space_left)
                    # 将地图加入chopped
                    chopped_map1[index_chop] = chopped_map
                    index_chop = index_chop + 1

                    action_list.append(action_space_left)
                    index_x = index_x - 1
                else:
                    # 追加历史动作
                    history_action_list.append(action_list[:])
                    # 将动作加入动作标签
                    action_list_space.append(action_space_down)
                    # 将地图加入chopped
                    chopped_map1[index_chop] = chopped_map
                    index_chop = index_chop + 1

                    action_list.append(action_space_down)
                    index_y = index_y + 1
                    direct_bool = bool(1 - direct_bool)
        if i == 2:
            break
    print(len(history_action_list))
    print(chopped_map.shape)
    print(chopped_map1.shape)
    # print(history_action_list)
    for i in range(len(history_action_list)):
        history_action_list[i] = history_action_list[i][-8:]
        if len(history_action_list[i]) < 8:
            history_action_list[i] = (8 - len(history_action_list[i])) * [0] + history_action_list[i]

    # print(action_list_space)
    print(history_action_list[:5])
    # print(chopped_map[:5])
    return chopped_map1, history_action_list, action_list_space


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 32, 32)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 3, 1, 1),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # output shape (32, 7, 7)
        )
        self.out1 = nn.Linear(2056, 128)  # fully connected layer, output 10 classes
        self.out = nn.Linear(128, 128)
        self.test = nn.Linear(128, 5)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        # 拼接步骤

        # print(x.shape)
        # print(y.shape)
        x = torch.cat([x, y], 1)
        # print(x.shape)
        output = self.out1(x)
        output = self.out(output)
        output = self.test(output)

        return x


def train_test(chopped_map1, history_action_list, action_list_space):
    EPOCH = 1000
    LR = 0.001  # 学习率

    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    s_ = np.expand_dims(chopped_map1[:3000], axis=1)
    torch_data = torch.Tensor(s_)

    # print(cnn(torch_data))

    b_y = torch.LongTensor(history_action_list)

    test_y = torch.LongTensor(action_list_space)
    print(test_y.shape)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    bacht_size = 10
    for epoch in range(EPOCH):
        for step in range(0, 2040, bacht_size):  # 分配 batch data, normalize x when iterate train_loader
            output = cnn(torch_data[step:step + bacht_size], b_y[step:step + bacht_size])  # cnn output
            loss = loss_func(output, test_y[step:step + bacht_size])  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    from symbol import term

    index_re = 20
    test_output = cnn(torch_data[:index_re], b_y[:index_re])
    print(test_output)
    # 输出一维最大值 纵坐标
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:index_re].numpy())

    index_re = 60
    index_start = 2040
    test_output = cnn(torch_data[index_start:(index_start + index_re)], b_y[index_start:(index_start + index_re)])
    # 输出一维最大值 纵坐标
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[index_start:(index_start + index_re)].numpy())


if __name__ == "__main__":
    # 点云地图点数
    point_size = 160
    # 训练集大小 暂设 10000
    train_size = 10000
    # 最低深度
    min_depth = -5
    # 最高深度
    max_depth = 10
    chopped_map1, history_action_list, action_list_space = create_test()
    train_test(chopped_map1, history_action_list, action_list_space)
