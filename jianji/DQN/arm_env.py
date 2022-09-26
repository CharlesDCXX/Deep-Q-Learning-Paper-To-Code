import numpy as np

'''
1.初始化，得到履带吊
    底座坐标
    臂长
    上扬角度
    底盘旋转角度
    目标点坐标
    障碍物坐标
2.根据目标点坐标及当前坐标形成目标轨迹
3.将轨迹传送给底层控制
'''


def math_a(angle):
    return angle * np.pi / 180


class ArmEnv(object):
    def __init__(self, space_x, space_y):
        # 第一个坐标右手边x，第二个坐标面朝方向y，第三个坐标高度z
        self.space = np.zeros([10, 10, 10], dtype=int)
        self.space_now = np.zeros([space_x, space_y])
        # 履带吊底座坐标
        self.base = [5, 0, 0]
        # 履带吊长度
        self.arm_length = 10
        # 履带吊上扬初始角度
        self.arm_angle = 80
        # 履带吊底盘旋转角度
        self.base_angle = 0
        # 是否显示
        self.is_render = False
        # 目标点
        self.goal = [3, 4, 0]
        # 障碍物
        self.obstacle = [3, 4, 5]
        # 履带吊顶端坐标
        self.arm_head = self.get_coordinate(self.base_angle, self.arm_angle)
        self.action_space = [[0, -1], [0, 1], [1, 0], [-1, 0], [1, 1], [-1, -1]]

    # 根据角度得到坐标
    def get_coordinate(self, base_angle, arm_angle):
        z = np.sin(math_a(arm_angle)) * self.arm_length
        z = round(z, 0)
        xy = np.cos(math_a(arm_angle)) * self.arm_length
        y = np.cos(math_a(base_angle)) * xy
        x = np.sin(math_a(arm_angle)) * xy
        x = round(x, 0)
        y = round(y, 0)
        return [x + self.base[0], y + self.base[1], z + self.base[2]]

    def step(self, action):
        self.arm_angle += self.action_space[action][0]
        self.base_angle += self.action_space[action][1]
        self.update_observation(self.base_angle, self.arm_angle)
        coordinate = self.get_coordinate(self.base_angle, self.arm_angle)
        distance = ((coordinate[0] - self.goal[0]) ** 2 + (coordinate[1] - self.goal[1]) ** 2) ** 0.5
        reward = 20 - distance
        done = True if distance < 5 else False
        info = ""
        return self.space_now, reward, done, info

    def reset(self):
        self.arm_angle = 80
        # 履带吊底盘旋转角度
        self.base_angle = 0
        return self.space_now

    def update_observation(self, base_angle, arm_angle):
        coordinate = self.get_coordinate(base_angle, arm_angle)
        self.space_now = self.space_now * 0
        self.space_now[coordinate[0], coordinate[1]] = 1


# 得到目标旋转角度
def get_up_angel(env):
    goal_x = env.goal[0]
    goal_y = env.goal[1]
    # 求出地面斜边
    # 斜边长度
    goal_xy = (goal_x * goal_x + goal_y * goal_y) ** 0.5
    return np.arccos(goal_xy / env.arm_length) * 180 / np.pi


# 得到目标底盘旋转角度
def get_base_angel(env):
    goal_x = env.goal[0]
    goal_y = env.goal[1]
    goal_xy = (goal_x * goal_x + goal_y * goal_y) ** 0.5
    return np.arccos(goal_y / goal_xy) * 180 / np.pi


# 得到障碍物旋转角度
def get_obstacle_angel(env):
    obstacle_x = env.obstacle[0]
    obstacle_y = env.obstacle[1]
    obstacle_z = env.obstacle[2]
    # 求出地面斜边
    obstacle_xy = (obstacle_x * obstacle_x + obstacle_y * obstacle_y) ** 0.5
    obstacle_length = (obstacle_x * obstacle_x + obstacle_y * obstacle_y + obstacle_z * obstacle_z) ** 0.5
    return np.arccos(obstacle_xy / obstacle_length) * 180 / np.pi


def get_obstacle_base_angel(env):
    obstacle_x = env.obstacle[0]
    obstacle_y = env.obstacle[1]
    obstacle_xy = (obstacle_x * obstacle_x + obstacle_y * obstacle_y) ** 0.5
    return np.arccos(obstacle_y / obstacle_xy) * 180 / np.pi


if __name__ == '__main__':
    env = ArmEnv(space_x=1, space_y=1)
    print(env.space_now)


def test():
    env = ArmEnv()
    array_get_up_angel = [round(i, 1) for i in np.arange(env.arm_angle, get_up_angel(env), -0.5)]
    array_get_base_angel = [round(i, 1) for i in np.arange(env.base_angle, get_base_angel(env), 0.5)]
    array_get_up_angel.append(get_up_angel(env))
    array_get_base_angel.append(get_base_angel(env))
    obstacle_angel = get_obstacle_angel(env)
    obstacle_base_angel = get_obstacle_base_angel(env)
    min_stop = min(len(array_get_base_angel), len(array_get_base_angel))

    for i in range(0, min_stop):
        if obstacle_base_angel + 1 > array_get_base_angel[i] > obstacle_base_angel - 1 \
                and obstacle_angel + 1 > array_get_up_angel[i] > obstacle_angel - 1:
            array_get_base_angel.insert(i, array_get_base_angel[i - 1])
