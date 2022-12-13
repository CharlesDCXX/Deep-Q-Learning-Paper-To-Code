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


def get_observation(s):
    s_ = np.expand_dims(s, axis=0)
    return s_

class ArmEnv(object):
    def __init__(self, space_x, space_y, space_z):
        # 第一个坐标右手边x，第二个坐标面朝方向y，第三个坐标高度z
        self.space_now = np.zeros([space_x, space_y, space_z], dtype=int)
        # 履带吊底座坐标
        self.base = [0, -7, 9.5]
        # 履带吊长度
        self.arm_length = 25.51
        # 履带吊吊绳长度
        self.line_length = 0
        # 履带吊上扬初始角度
        self.arm_angle = 90
        # 履带吊底盘旋转角度
        self.base_angle = 0
        # 是否显示
        self.is_render = False
        self.on_goal = 0
        # 目标点
        self.goal = [10, -7, 0]
        self.goal_arm_angle = self.get_target_up_angel()
        self.goal_base_angle = self.get_target_base_angel()
        print("目标动臂角度：", self.goal_arm_angle, "目标底盘角度：", self.goal_base_angle)

        # 障碍物
        self.obstacle = [14, -7, 0]
        self.obstacle_arm_angle = self.get_obstacle_angel()
        self.obstacle_arm_angle = 84
        self.obstacle_base_angle = self.get_obstacle_base_angel()
        self.obstacle_base_angle = 20
        print("障碍物动臂角度：", self.obstacle_arm_angle, "障碍物底盘角度：", self.obstacle_base_angle)

        # 履带吊顶端坐标
        self.arm_head = self.get_coordinate(self.base_angle, self.arm_angle)
        self.action_space = [[0, -1], [0, 1], [1, 0], [-1, 0]]
        self.pre_distance = (((self.arm_angle - self.goal_arm_angle) ** 2 + (
                self.base_angle - self.goal_base_angle) ** 2) ** 0.5)

        self.space_now[self.obstacle[0], self.obstacle[1], self.obstacle[2]] = 1
        self.space_now[self.goal[0], self.goal[1], self.goal[2]] = 1
        self.s = self.reset()

    def step(self, action):
        # print("action%d"% action)
        self.arm_angle += self.action_space[action][0]
        self.base_angle += self.action_space[action][1]
        if self.arm_angle > 90:
            self.arm_angle = 90
        if self.arm_angle < 60:
            self.arm_angle = 60
        if self.base_angle > 180:
            self.base_angle = 180
        if self.base_angle < 0:
            self.base_angle = 0
        # self.update_observation(self.base_angle, self.arm_angle)
        # print("self.arm_angle:%.2f,self.base_angle:%.2f" % (self.arm_angle, self.base_angle))
        reward = 0
        distance = (((self.arm_angle - self.goal_arm_angle) ** 2 + (
                self.base_angle - self.goal_base_angle) ** 2) ** 0.5)
        # reward = self.pre_distance - distance
        if self.pre_distance - distance > 0:
            reward = 1
        else:
            reward = - 1
        self.pre_distance = distance
        # print("reward%s" % reward)
        done = False
        if abs(self.obstacle_base_angle - self.base_angle) <= 2 and abs(self.obstacle_arm_angle - self.arm_angle) <= 2:
            reward = reward - 20
        if abs(self.goal_base_angle - self.base_angle) <= 1 and abs(self.goal_arm_angle - self.arm_angle) <= 1:
            self.on_goal += 1
            reward = reward + 20
            if self.on_goal > 5:
                done = True
        info = ""

        s = np.concatenate(
            (np.array([self.base_angle, self.arm_angle]), np.array([self.goal_base_angle, self.goal_arm_angle]),
             np.array([self.obstacle_base_angle, self.obstacle_arm_angle]), [0])
        )
        s = get_observation(s)
        return s, reward, done, info

    # 根据角度得到顶端坐标
    def get_coordinate(self, base_angle, arm_angle):
        z = np.sin(math_a(arm_angle)) * self.arm_length
        z = round(z, 0)
        xy = np.cos(math_a(arm_angle)) * self.arm_length
        y = np.cos(math_a(base_angle)) * xy
        x = np.sin(math_a(arm_angle)) * xy
        x = round(x, 0)
        y = round(y, 0)
        return [x + self.base[0], y + self.base[1], z + self.base[2]]

    # 重置履带吊两角度
    def reset(self):
        self.arm_angle = 90
        # 履带吊底盘旋转角度
        self.base_angle = 0
        self.on_goal = 0
        # return self.space_now
        s = np.concatenate((np.array([self.base_angle, self.arm_angle]),
                            np.array([self.goal_base_angle, self.goal_arm_angle]),
                            np.array([self.obstacle_base_angle, self.obstacle_arm_angle]),
                            [0]))
        s = get_observation(s)
        return s

    def update_observation(self, base_angle, arm_angle):
        index_z = np.sin(math_a(arm_angle))
        xy = np.cos(math_a(arm_angle))
        index_x = np.sin(math_a(base_angle)) * xy
        index_y = np.cos(math_a(base_angle)) * xy
        # self.space_now[int(self.base[0] + round(index_x * self.arm_length, 0)), int(self.base[1] + round(index_y * self.arm_length, 0)), int(self.base[0] + round(
        #             index_z * self.arm_length, 0))] = 1
        for i in range(int(self.arm_length)):
            self.space_now[
                int(self.base[0] + round(index_x * i, 0)), int(self.base[1] + round(index_y * i, 0)), int(
                    self.base[0] + round(
                        index_z * i, 0))] = 1

    # 得到目标旋转角度
    def get_target_up_angel(env):
        goal_x = env.goal[0] - env.base[0]
        goal_y = env.goal[1] - env.base[1]
        # 求出地面斜边
        # 斜边长度
        goal_xy = (goal_x * goal_x + goal_y * goal_y) ** 0.5
        return np.arccos(goal_xy / env.arm_length) * 180 / np.pi

    # 得到目标底盘旋转角度
    def get_target_base_angel(env):
        goal_x = env.goal[0] - env.base[0]
        goal_y = env.goal[1] - env.base[1]
        goal_xy = (goal_x * goal_x + goal_y * goal_y) ** 0.5
        return np.arccos(goal_y / goal_xy) * 180 / np.pi

    # 得到障碍物旋转角度
    def get_obstacle_angel(env):
        obstacle_x = env.obstacle[0] - env.base[0]
        obstacle_y = env.obstacle[1] - env.base[1]
        obstacle_z = env.obstacle[2] - env.base[2]
        # 求出地面斜边
        obstacle_xy = (obstacle_x * obstacle_x + obstacle_y * obstacle_y) ** 0.5
        obstacle_length = (obstacle_x * obstacle_x + obstacle_y * obstacle_y + obstacle_z * obstacle_z) ** 0.5
        return np.arccos(obstacle_xy / obstacle_length) * 180 / np.pi

    def get_obstacle_base_angel(env):
        obstacle_x = env.obstacle[0] - env.base[0]
        obstacle_y = env.obstacle[1] - env.base[1]
        obstacle_xy = (obstacle_x * obstacle_x + obstacle_y * obstacle_y) ** 0.5
        return np.arccos(obstacle_y / obstacle_xy) * 180 / np.pi


if __name__ == '__main__':
    env = ArmEnv(space_x=1, space_y=1)
    # 获得目标角度
    target_arm_angle = env.get_target_up_angel()
    target_base_angle = env.get_target_base_angel()
    print("target_base_angle:%f,target_arm_angle:%f" % (target_base_angle, target_arm_angle))
    # 获得障碍物角度
    obstacle_arm_angle = env.get_obstacle_angel()
    obstacle_base_angle = env.get_obstacle_base_angel()
    print("obstacle_base_angle:%f,obstacle_arm_angle:%f" % (obstacle_base_angle, obstacle_arm_angle))
    base = True
    # 当相差角度过大的时候
    while abs(target_base_angle - env.base_angle) > 1 and abs(target_arm_angle - env.arm_angle) > 1:
        # 判断是否转动底盘
        if base:
            # 判断目标角度与底盘角度之差
            if abs(target_base_angle - env.base_angle) > 1.0:
                if abs(obstacle_base_angle - env.base_angle) < 2 and abs(obstacle_arm_angle - env.arm_angle) < 2:
                    base = bool(1 - base)
                else:
                    if (target_base_angle - env.base_angle) > 0:
                        env.base_angle = env.base_angle + 0.5
                    else:
                        env.base_angle = env.base_angle - 0.5
            else:
                base = bool(1 - base)
        else:
            if abs(target_arm_angle - env.arm_angle) > 1:
                if abs(obstacle_base_angle - env.base_angle) < 2 and abs(obstacle_arm_angle - env.arm_angle) < 2:
                    base = bool(1 - base)
                else:
                    if target_arm_angle - env.arm_angle > 0:
                        env.arm_angle = env.arm_angle + 0.5
                    else:
                        env.arm_angle = env.arm_angle - 0.5
            else:
                base = bool(1 - base)
        print("env.base_angele:%f,env.arm_angle:%f" % (env.base_angle, env.arm_angle))
    # print(abs(target_base_angle - env.base_angle)>1.0)
    # print(abs(obstacle_base_angle - env.base_angle))

# def test():
#     env = ArmEnv(space_x=1, space_y=1)
#     array_get_up_angel = [round(i, 1) for i in np.arange(env.arm_angle, get_up_angel(env), -0.5)]
#     array_get_base_angel = [round(i, 1) for i in np.arange(env.base_angle, get_base_angel(env), 0.5)]
#     array_get_up_angel.append(get_up_angel(env))
#     array_get_base_angel.append(get_base_angel(env))
#     obstacle_angel = get_obstacle_angel(env)
#     obstacle_base_angel = get_obstacle_base_angel(env)
#     min_stop = min(len(array_get_base_angel), len(array_get_base_angel))

#     for i in range(0, min_stop):
#         if obstacle_base_angel + 1 > array_get_base_angel[i] > obstacle_base_angel - 1 \
#                 and obstacle_angel + 1 > array_get_up_angel[i] > obstacle_angel - 1:
#             array_get_base_angel.insert(i, array_get_base_angel[i - 1])
