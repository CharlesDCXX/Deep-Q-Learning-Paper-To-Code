import numpy as np

# 设定cell大小
m, n = 5, 5
train_size = 10000
# 最低深度
min_depth = -5
# 最高深度
max_depth = 10

np_save = np.random.randint(min_depth, max_depth, (train_size, m, n))
np.save("cell_x", np_save)
print(np_save.shape)

task_depth = -5
np_save = np.load('cell_x.npy')

cell_grid = np.ones([train_size * 10, m, n])

index = np.argmax(np_save[0])
x = int(index / n)
y = index % n

# 挖掘输出坐标
action_list_cem = []
index_chop = 0
for i in range(0, train_size):
    while np_save[i, x, y] > task_depth:
        cell_grid[index_chop] = np_save[i]
        index_chop = index_chop + 1
        action_list_cem.append(index)
        np_save[i, x, y] = task_depth
        index = np.argmax(np_save[i])
        x = int(index / n)
        y = index % n
    if i == 3000:
        break
print(len(action_list_cem))

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt


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
        self.out1 = nn.Linear(32, 128)  # fully connected layer, output 10 classes
        self.out = nn.Linear(128, 128)
        self.test = nn.Linear(128, 25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        # 拼接步骤

        # print(x.shape)
        # print(y.shape)
        # x = torch.cat([x,y],1)

        output = self.out1(x)
        output = self.out(output)
        output = self.test(output)
        return output


import time

EPOCH = 1000
LR = 0.001  # 学习率

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
sample_size = len(action_list_cem)
s_ = np.expand_dims(cell_grid[:sample_size], axis=1)
torch_data = torch.Tensor(s_)

# print(cnn(torch_data))

b_y = torch.LongTensor(action_list_cem)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
bacht_size = 40
for epoch in range(EPOCH):
    for step in range(0, sample_size - 1000, bacht_size):  # 分配 batch data, normalize x when iterate train_loader
        output = cnn(torch_data[step:step + bacht_size])  # cnn output
        loss = loss_func(output, b_y[step:step + bacht_size])  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

torch.save(cnn, 'cem_net.pkl')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

index_re = 60
index_start = sample_size - 1000
test_output = cnn(torch_data[index_start:(index_start + index_re)])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(b_y[index_start:(index_start + index_re)].numpy())