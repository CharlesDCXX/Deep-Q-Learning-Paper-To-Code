import sys
import numpy as np
# 传入x,y坐标
x = int(sys.argv[1])
y = int(sys.argv[2])
arm_length = 35.4

# 得到目标旋转角度
def get_up_angel():
    goal_x = x
    goal_y = y
    # 求出地面斜边
    # 斜边长度
    goal_xy = (goal_x * goal_x + goal_y * goal_y) ** 0.5
    return np.arccos(goal_xy / arm_length) * 180 / np.pi


# 得到目标底盘旋转角度
def get_base_angel():
    goal_x = x
    goal_y = y
    goal_xy = (goal_x * goal_x + goal_y * goal_y) ** 0.5
    return np.arccos(goal_y / goal_xy) * 180 / np.pi
