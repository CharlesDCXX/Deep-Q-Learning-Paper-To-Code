import numpy as np
from dqn_agent import DQNAgent
from arm_env import ArmEnv
import time
from utils import plot_learning_curve

if __name__ == '__main__':
    env = ArmEnv(space_x=50, space_y=50,space_z=50)

    best_score = -np.inf
    load_checkpoint = False
    n_games = 30

    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=env.space_now.shape,
                     n_actions=len(env.action_space), mem_size=200, eps_min=0.001,
                     batch_size=32, replace=1000, eps_dec=1e-4,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='xcmg')

    if load_checkpoint:
        agent.load_models()
        Note = open('x.txt', mode='w')

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_xcmg_' \
            + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        print(i)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            if load_checkpoint:
                a = ("当前动臂角度：:"+str(env.arm_angle) + ' ' + "当前底盘角度：:" + str(env.base_angle))
                Note.write(a + '\n')
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:

            best_score = avg_score
        if not load_checkpoint:
            agent.save_models()
        eps_history.append(agent.epsilon)

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
