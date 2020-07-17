# -*- coding: utf-8 -*-

from World import World
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pickle


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


def GLIE_5(episode, num_episodes):
    epsilon = 0.999**episode
    return epsilon


def GLIE_6(episode, num_episodes):
    epsilon = 0.99**episode
    return epsilon


def GLIE_7(episode, num_episodes):
    epsilon = 0.90**episode
    return epsilon


if __name__ == "__main__":
    """
    
    """
    env = World()
    # # alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    # list_of_GLIE_funcs = [GLIE_1, GLIE_2, GLIE_3, GLIE_4, GLIE_5, GLIE_6, GLIE_7]
    #
    # # for curr_alpha in alphas:
    # for indx, glie in enumerate(list_of_GLIE_funcs):
    #
    #     # print('\n### \nALPHA %s out of %s' % (indx + 1, len(alphas)))
    #     # Q, total_rewards, len_paths = env.sarsa(num_episodes=5000, alpha=curr_alpha, GLIE=GLIE_3, gamma=0.9)
    #     # Q, total_rewards, len_paths = env.Qlearning(num_episodes=5000, alpha=curr_alpha, GLIE=GLIE_2, gamma=0.9)
    #     # plt.subplot(211)
    #     # plt.plot(total_rewards, label='alpha = %s' % curr_alpha)
    #     # plt.subplot(212)
    #     # plt.plot(len_paths, label='alpha = %s' % curr_alpha)
    #
    #     print('\n### \nGLIE func %s out of %s' % (indx + 1, len(list_of_GLIE_funcs)))
    #     # Q, total_rewards, len_paths = env.sarsa(num_episodes=5000, alpha=0.05, GLIE=glie, gamma=0.9)
    #     Q, total_rewards, len_paths = env.Qlearning(num_episodes=5000, alpha=0.05, GLIE=glie, gamma=0.9)
    #     plt.subplot(211)
    #     plt.plot(total_rewards, label='GLIE %s' % (indx + 1))
    #     plt.subplot(212)
    #     plt.plot(len_paths, label='GLIE %s' % (indx + 1))
    #
    # plt.subplot(211)
    # plt.legend()
    # plt.title('total rewards')
    # plt.subplot(212)
    # plt.legend()
    # plt.title('lengths')
    # plt.show()

    start = time.time()
    # ------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------ #
    # ----------------- UNCOMMENT ONE OF THESE LINES IN ORDER TO CHECK THE ALGORITHM ----------------- #
    # ------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------ #
    Q, total_rewards_sarsa, len_paths = env.sarsa(num_episodes=2000, alpha=0.05, GLIE=GLIE_6, gamma=0.9)
    # Q, total_rewards_q_lrng, len_paths = env.Qlearning(num_episodes=2000, alpha=0.01, GLIE=GLIE_2, gamma=0.9)
    # ------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------ #
    # pickle.dump(Q, open("sarsa.p", "wb"))
    # pickle.dump(Q, open("q_learning.p", "wb"))
    end = time.time()
    print('\nIt took {:.2f} minutes to finish the run.'.format((end - start) / 60.0))

    # env.plot_actionValues(Q)

    # pickle
    # Q = pickle.load(open("sarsa.p", "rb"))

    # env.policy_evaluation(Q, 2000, alpha=0.1, gamma=0.9)
    # plt.plot(total_rewards_q_lrng)
    # plt.plot(total_rewards_sarsa)

    plt.show()
    # ------------------------------------------------------------------------------------------------ #
    # --------------------------- HOW MANY GAMES YOU WANT TO PLAY ------------------------------------ #
    # ------------------------------------------------------------------------------------------------ #
    tryings = 3
    # ------------------------------------------------------------------------------------------------ #
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