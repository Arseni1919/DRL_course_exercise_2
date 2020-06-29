# -*- coding: utf-8 -*-

from World import World
import numpy as np


def get_best_action(curr_Q, curr_state, curr_env):
    return np.argmax([curr_Q[(curr_state, a)] for a in curr_env.Actions]) + 1


def GLIE_1(episode, num_episodes):
    epsilon = 1 - episode/num_episodes
    return epsilon


if __name__ == "__main__":

    env = World()
    # Q = env.sarsa(num_episodes=2000, alpha=0.1, GLIE=GLIE_1, gamma=0.9)
    Q = env.Qlearning(num_episodes=2000, alpha=0.1, GLIE=GLIE_1, gamma=0.9)
    env.plot_actionValues(Q)
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
                print("Episode finished after {} timesteps".format(t + 1))
                break
            #input()