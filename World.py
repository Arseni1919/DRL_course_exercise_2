import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
from numpy.random import choice
import pandas as pd
import time
import random
from collections import deque


def calculate_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    rewards.reverse()
    curr_sum = 0
    for indx in range(len(rewards)):
        curr_sum = curr_sum + (gamma**indx) * rewards[indx]
        discounted_rewards.append(curr_sum)

    rewards.reverse()
    discounted_rewards.reverse()
    return discounted_rewards


def is_first_visit(indx, states):
    state = states[indx]
    for curr_indx, curr_state in enumerate(states):
        if curr_state == state:
            return curr_indx == indx
    return ValueError()


class World:

    def __init__(self):

        self.nRows = 4
        self.nCols = 4
        self.stateHoles = [1, 7, 14, 15]
        self.stateGoal = [13]
        self.nStates = 16
        self.States = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.nActions = 4
        self.Actions = [1, 2, 3, 4]
        self.rewards = np.array([-1] + [-0.04] * 5 + [-1] + [-0.04] * 5 + [1, -1, -1] + [-0.04])
        self.stateInitial = [4]
        self.observation = []

    def _plot_world(self):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateHoles:
            (I, J) = np.unravel_index(i, shape=(nRows, nCols), order='F')
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.5")
            plt.plot(xs, ys, "black")
        for ind, i in enumerate([stateGoal]):
            (I, J) = np.unravel_index(i, shape=(nRows, nCols), order='F')
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.8")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols

        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center',
                         verticalalignment='center')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        # plt.show(block=False)
        plt.show()

    def plot_value(self, valueFunction):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        fig = plt.plot(1)
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateGoal:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(valueFunction[k], 3)), fontsize=12,
                             horizontalalignment='center', verticalalignment='center')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_policy(self, policy):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        X1 = X[:-1, :-1]
        Y1 = Y[:-1, :-1]
        X2 = X1.reshape(-1, 1) + 0.5
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        X2 = np.kron(np.ones((1, nActions)), X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        mat = np.cumsum(np.ones((nStates, nActions)), axis=1).astype("int64")
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
        index_no_policy = stateHoles + stateGoal
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        mask = policy.astype("int64") * mat
        mask = mask.reshape(nRows, nCols, nCols)
        X3 = X2.reshape(nRows, nCols, nActions)
        Y3 = Y2.reshape(nRows, nCols, nActions)
        alpha = np.pi - np.pi / 2 * mask
        self._plot_world()
        for ii in index_policy:
            ax = plt.gca()
            j = int(ii / nRows)
            i = (ii + 1 - j * nRows) % nCols - 1
            index = np.where(mask[i, j] > 0)[0]
            h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]), 0.3)
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.25, j - 0.25, str(states[k]), fontsize=16, horizontalalignment='right',
                         verticalalignment='bottom')
                k += 1
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateGoal(self):

        return self.stateGoal

    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions

    def get_transition_model(self, p=0.8):
        nstates = self.nStates
        nrows = self.nRows
        holes_index = self.stateHoles
        goal_index = self.stateGoal
        terminal_index = holes_index + goal_index
        # actions = ["1", "2", "3", "4"]
        actions = [1, 2, 3, 4]  # I changed str to int
        transition_models = {}
        for action in actions:
            transition_model = np.zeros((nstates, nstates))
            for i in range(1, nstates + 1):
                if i not in terminal_index:
                    if action == 1:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                    if action == 2:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if 0 < i % nrows and (i + 1) <= nstates:
                            transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                    if action == 3:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i % nrows and (i + 1):
                            transition_model[i - 1][i + 1 - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                    if action == 4:
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if 0 < i % nrows and (i + 1) <= nstates:
                            transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                elif i in terminal_index:
                    transition_model[i - 1][i - 1] = 1

            transition_models[action] = pd.DataFrame(transition_model, index=range(1, nstates + 1),
                                                     columns=range(1, nstates + 1))
        return transition_models

    def step(self, action):
        observation = self.observation
        state = observation[0]
        prob = {}
        done = False
        transition_models = self.get_transition_model(0.8)
        # print('inside')
        # print(state)
        # print(action)
        prob = transition_models[action].loc[state, :]
        # print(transition_models[action].loc[state, :])
        s = choice(self.States, 1, p=prob)
        next_state = s[0]
        reward = self.rewards[next_state - 1]

        if next_state in self.stateGoal + self.stateHoles:
            done = True
        self.observation = [next_state]
        return next_state, reward, done

    def reset(self, *args):
        # def reset(self):
        if not args:
            observation = self.stateInitial
        else:
            observation = []
            while not (observation):
                observation = np.setdiff1d(choice(self.States), self.stateHoles + self.stateGoal)
        self.observation = observation
        return observation[0]

    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation  # observation
        state = observation[0]

        # state = 3

        J = nRows - (state - 1) % nRows - 1
        I = int((state - 1) / nCols)

        circle = plt.Circle((I + 0.5, J + 0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()

        # plt.ion()
        # plt.show()
        # plt.draw()
        # plt.pause(0.5)
        # plt.ion()
        # plt.show(block=False)
        # time.sleep(1)
        # nRows = self.nRows
        # nCols = self.nCols
        # stateHoles = self.stateHoles
        # stateGoal = self.stateGoal

        # print(state)

        # circle = plt.Circle((0.5, 0.5), 0.1, color='black')
        # fig, ax = plt.subplots()
        # ax.add_artist(circle)

        # k = 0
        # for i in range(nCols):
        #     for j in range(nRows, 0, -1):
        #         if k + 1 not in stateHoles + stateGoal:
        #             plt.text(i + 0.5, j - 0.5, str(self._truncate(valueFunction[k], 3)), fontsize=12,
        #                      horizontalalignment='center', verticalalignment='center')
        #         k += 1

    def close(self):
        plt.pause(0.5)
        plt.close()

    def show(self):
        plt.ion()
        plt.show()

    def epsilon_greedy(self, epsilon, Q, state):
        action = random.choice(self.Actions) if random.random() < epsilon else np.argmax(
            [Q[(state, a)] for a in self.Actions]) + 1
        return action

    def greedy_action(self, Q, state):
        return np.argmax([Q[(state, a)] for a in self.Actions]) + 1

    def loop_of_algorithm(self, num_episodes, alpha, GLIE, gamma, update_func):
        average_on_last = 500
        total_rewards = []
        average_list = deque(maxlen=average_on_last)
        len_of_good_paths = []
        average_list_len_path = deque(maxlen=average_on_last)
        last_len = 0
        # initialization of Q function
        Q = {}
        for s in self.States:
            for a in self.Actions:
                Q[(s, a)] = 0

        for i in range(num_episodes):
            total_reward = 0
            print('\r%s out of %s' % (i + 1, num_episodes), end='')
            epsilon = GLIE(i, num_episodes)
            state = self.reset()
            action = self.epsilon_greedy(epsilon, Q, state)
            t = 0
            done = False
            while not done:
                next_state, reward, done = self.step(action)
                next_action = self.epsilon_greedy(epsilon, Q, next_state)

                Q[(state, action)] = update_func(Q, state, action,
                                                 alpha, reward, gamma, next_state, next_action)

                t += 1
                state = next_state
                action = next_action
                total_reward += reward

                if done:
                    # print("Episode finished after {} timesteps".format(t + 1))
                    # ----------- GRAPHS ----------- #
                    average_list.append(total_reward)
                    total_rewards.append(np.mean(average_list))

                    last_len = t+1 if reward == 1 else last_len
                    average_list_len_path.append(last_len)
                    len_of_good_paths.append(np.mean(average_list_len_path))
                    # len_of_good_paths.append(last_len)

                    break
        print()
        return Q, total_rewards, len_of_good_paths

    def sarsa(self, num_episodes, alpha, GLIE, gamma):

        print('---# SARSA algorithm #---')

        def update_sarsa(Q, state, action, alpha, reward, gamma, next_state, next_action):
            # Q(St, At) ← Q(St, At) + α(Rt+1 + γQ(St+1, At+1) − Q(St, At))
            return Q[(state, action)] + alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])

        return self.loop_of_algorithm(num_episodes, alpha, GLIE, gamma, update_sarsa)

    def plot_actionValues(self, Q):
        special_states = self.stateHoles + self.stateGoal
        self._plot_world()
        k = 1
        for i in range(self.nCols):
            for j in range(self.nRows, 0, -1):
                if k not in special_states:
                    plt.text(i + 0.5, j - 0.2, str(self._truncate(Q[k, 1], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='center')
                    plt.text(i + 0.8, j - 0.5, str(self._truncate(Q[k, 2], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='center')
                    plt.text(i + 0.5, j - 0.8, str(self._truncate(Q[k, 3], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='center')
                    plt.text(i + 0.2, j - 0.5, str(self._truncate(Q[k, 4], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='center')
                    plt.plot([i, i + 1], [j, j - 1], c='k', lw=1, ls='dotted')
                    plt.plot([i, i + 1], [j - 1, j], c='k', lw=1, ls='dotted')
                k += 1
        plt.title('Action-Values Grid World', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def Qlearning(self, num_episodes, alpha, GLIE, gamma):

        print('---# Q-learning algorithm #---')

        def update_Qlearning(Q, state, action, alpha, reward, gamma, next_state, next_action):
            # Q(St, At) ← Q(St, At) + α(Rt+1 + γ maxa Q(St+1, a) − Q(St, At))
            max_action_value = max([Q[(next_state, a)] for a in self.Actions])
            return Q[(state, action)] + alpha * (reward + gamma * max_action_value - Q[(state, action)])

        return self.loop_of_algorithm(num_episodes, alpha, GLIE, gamma, update_Qlearning)

    def choose_state(self):
        special_states = self.stateHoles + self.stateGoal
        curr_state = random.randint(0, 15)
        while True:
            if curr_state + 1 in special_states:
                break
            curr_state = random.randint(0, 15)
        return curr_state

    def policy_evaluation(self, Q, num_of_episodes, alpha, gamma=0.9):
        """
        Check the values of each cell
        :param Q:
        :param num_of_episodes:
        :param alpha:
        :param gamma:
        :return:
        """

        # numVisitsFunction = [0 for i in range(self.nStates)]
        valueFunction = [0 for i in range(self.nStates)]

        for i in range(num_of_episodes):
            print('\r%s out of %s' % (i + 1, num_of_episodes), end='')
            # states = []
            # rewards = []
            done = False
            state = self.reset()

            while not done:
                # states.append(state - 1)
                action = self.greedy_action(Q, state)
                next_state, reward, done = self.step(action)
                valueFunction[state - 1] = valueFunction[state - 1] + alpha * (reward + gamma * valueFunction[next_state - 1] - valueFunction[state - 1])
                # rewards.append(reward)
                state = next_state

            # discounted_rewards = calculate_discounted_rewards(rewards, gamma)
            # for indx in range(len(states)):
            #     if is_first_visit(indx, states):
            #         numVisitsFunction[states[indx]] += 1
            #         val_of_state = valueFunction[states[indx]]
            #         valueFunction[states[indx]] = val_of_state + (1.0/numVisitsFunction[states[indx]])*(discounted_rewards[indx] - val_of_state)

        self.plot_value(valueFunction)
