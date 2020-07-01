# -*- coding: utf-8 -*-

from World import World
import numpy as np
import matplotlib.pyplot as plt
import math

# alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]


def get_best_action(curr_Q, curr_state, curr_env):
    return np.argmax([curr_Q[(curr_state, a)] for a in curr_env.Actions]) + 1


def GLIE_1(episode, num_episodes):
    epsilon = 1 - episode/num_episodes
    return epsilon


def GLIE_2(episode, num_episodes):
    epsilon = 1 - np.exp(episode)/np.exp(num_episodes)
    return epsilon


def GLIE_3(episode, num_episodes):
    epsilon = 1 - np.log(episode)/np.log(num_episodes)
    return epsilon


def GLIE_4(t, min_epsilon, divisor=25):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / divisor)))

if __name__ == "__main__":
    """
    - which GLIE to check?
    - what is the plot_actionValue func has to do?
    """
    env = World()
    # for curr_alpha in alphas:
    # # for indx, glie in enumerate([GLIE_4, GLIE_2, GLIE_3]):
    # #     Q, total_rewards, len_paths = env.Qlearning(num_episodes=5000, alpha=curr_alpha, GLIE=GLIE_2, gamma=0.9)
    #     Q, total_rewards, len_paths = env.sarsa(num_episodes=5000, alpha=curr_alpha, GLIE=GLIE_3, gamma=0.9)
    #
    #     plt.subplot(211)
    #     plt.plot(total_rewards, label='alpha = %s' % curr_alpha)
    #     plt.subplot(212)
    #     plt.plot(len_paths, label='alpha = %s' % curr_alpha)
    #
    #     # plt.subplot(211)
    #     # plt.plot(total_rewards, label='GLIE %s' % (indx + 1))
    #     # plt.subplot(212)
    #     # plt.plot(len_paths, label='GLIE %s' % (indx + 1))
    #
    # plt.subplot(211)
    # plt.legend()
    # plt.title('total rewards')
    # plt.subplot(212)
    # plt.legend()
    # plt.title('lengths')
    # plt.show()

    Q, total_rewards1, len_paths = env.sarsa(num_episodes=100, alpha=0.05, GLIE=GLIE_3, gamma=0.9)
    # Q, total_rewards2, len_paths = env.Qlearning(num_episodes=2000, alpha=0.01, GLIE=GLIE_2, gamma=0.9)
    env.plot_actionValues(Q)

    env.plot_policy(Q)
    # plt.plot(total_rewards1)
    # plt.plot(total_rewards2)
    # plt.show()
    # env.plot_value([i for i in range(16)])
    # env.plot()
    tryings = 3
    for trying in range(tryings):
        state = env.reset()
        done = False
        t = 0
        env.show()
        while not done:
            env.render()
            print("state=", state)
            # action = np.random.randint(1, env.nActions + 1)
            action = get_best_action(Q, state, env)
            # print("action=",action)
            state, reward, done = env.step(action)  # take a random action
            #env.render()
            # print("next_state",next_state)
            # print("env.observation[0]",env.observation[0])
            # print("done",done)
            # self.observation = [next_state];
            env.close()
            t += 1
            if done:
                print("Episode finished after {} timesteps and the reward is {}".format(t + 1, reward))
                break
            #input()