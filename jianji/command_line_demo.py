import numpy as np
import pandas as pd


np.random.seed(2)  # reproducible
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

        # 障碍物的数量 ？
        self.obstacle = [14, -7, 0]#【x,y,z,宽度，高度】
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

env = ArmEnv(space_x=100, space_y=100, space_z=100)
class qlearn(object):
    def __init__(self):
        self.N_STATES = 30 * 180  # the length of the 1 dimensional world
        self.ACTIONS = [0, 1, 2, 3]  # available actions
        self.EPSILON = 0.9  # greedy police 按照最优值选择路径的概率
        self.ALPHA = 0.1  # learning rate 学习率
        self.GAMMA = 0.9  # discount factor 对未来reward的一个衰减值
        self.MAX_EPISODES = 130  # maximum episodes  最大回合数



    def build_q_table(self,n_states, actions):
        table = pd.DataFrame(
            np.zeros((n_states, len(actions))),  # q_table initial values
            columns=actions,  # actions's name
        )
        # print(table)  # show table
        return table

    def choose_action(self, state, q_table):
        # This is how to choose an action
        state_actions = q_table.iloc[state, :]
        if (np.random.uniform() > self.EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(self.ACTIONS)
        else:  # act greedy
            action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
        return action_name

    def get_env_feedback(self, A):
        # This is how agent will interact with the environment
        s_, reward, done, info = env.step(A)
        S = int(90 - s_[0][1]) * 90 + int(s_[0][0])
        return S, reward, done

    def rl(self):
        # main part of RL loop
        q_table = self.build_q_table(self.N_STATES, self.ACTIONS)
        Note = open('x.txt', mode='w')
        for episode in range(self.MAX_EPISODES):
            step_counter = 0
            s = env.reset()

            S = int(90 - s[0][1]) * 90 + int(s[0][0])
            is_terminated = False
            while not is_terminated:
                if episode == self.MAX_EPISODES - 1:
                    self.EPSILON = 1
                    print("self.arm_angle:%d,self.base_angle:%d, 0" % (env.arm_angle, env.base_angle))
                    a = (str(env.base_angle) + ' ' + str(env.arm_angle) + ' 0')
                    Note.write(a + '\n')  # \n 换行符
                A = self.choose_action(S, q_table)
                S_, R, done = self.get_env_feedback(A)  # take action & get next state and reward
                q_predict = q_table.loc[S, A]
                if (1 - done):
                    q_target = R + self.GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
                else:
                    q_target = R  # next state is terminal
                    is_terminated = True  # terminate this episode

                q_table.loc[S, A] += self.ALPHA * (q_target - q_predict)  # update
                S = S_  # move to next state
                step_counter += 1
        # 或：多个sheet页面
        # 生成一个excelWriter
        Note.close()
        # writer = pd.ExcelWriter('./conclusion.xlsx')
        # q_table.to_excel(writer, sheet_name='sheet_1', index=False)
        # writer.save()

        return q_table


if __name__ == "__main__":
    q = qlearn()
    # 获得胡炳帅数据

    # 传入参数
    q_table = q.rl()
    # print('\r\nQ-table:\n')
    # print(q_table)
