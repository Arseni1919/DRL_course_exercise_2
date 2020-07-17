

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
# d = deque(maxlen=3)
# d.append(1)
# d.append(2)
# d.append(3)
# d.append(4)
# d.append(10)
# print(d)
# print(np.mean(d))
# print(np.argmax([1,10,20,3]))
# a = [1,2,3,19,200]
# b = a * 2
# plt.plot(range(len(a)), a)
# plt.plot(a)
# plt.show()
# def gli():
#     print('inside')
# gli.name = 'bal bla'
# print(gli.name)
# print(np.log(100))
# print(np.exp(100))
# plt.subplot(211)
# plt.plot([1,2,3,4,5])
# plt.subplot(212)
# plt.plot([1,2,5,6,9,10])
#
# plt.show()
# import time
# # start = time.time()
# # print("hello")
# # time.sleep(5)
# # end = time.time()
# # print('{:.2f}'.format((end - start)/60.0))
# # a = 0
import random
a = [1,2,3]
print(a.reverse())
print(a)
b = [4,5,6]
for i, j in zip(a, b):
    print(i, j)

# pickle.dump( favorite_color, open( "save.p", "wb" ) )
# favorite_color = pickle.load( open( "save.p", "rb" ) )




# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.pyplot import plot, draw, show
from numpy.random import choice
import pandas as pd


# import time

class World:

    def __init__(self):
        self.nRows = 4
        self.nCols = 4
        self.stateHoles = [1, 7, 14, 15]
        self.stateGoal = [13]
        self.nStates = 16
        self.States = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.nActions = 4
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
        prob = transition_models[action].loc[state, :]
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
        return observation

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

    def close(self):
        plt.pause(0.5)
        plt.close()

    def show(self):
        plt.ion()
        plt.show()

    # def SARSA(self, num_episodes,gamma,alpha):
    #     Q = np.zeros((self.nStates,self.nActions))
    #     t_rewards=[]
    #     for i in range (num_episodes):
    #         state = self.reset()
    #         t_reward = 0
    #         done = False
    #         action = np.argmax(Q[state,:] + np.random.randn(1, self.nActions) * (1./(i + 1)))
    #         while not done:
    #             next_state,reward , done = self.step(action)
    #             next_action = np.argmax(Q[next_state, :] + np.random.randn(1, self.nActions) * (1. / (i + 1)))
    #             Q[state,action]= Q[state,action] + alpha *(reward + gamma * Q[next_state, next_action] - Q[state,action])
    #             t_reward += reward
    #             state = next_state
    #             action = next_action
    #         t_rewards.append(t_reward)
    #         if i % 500 ==0 and i is not 0:
    #             print("Success rate:" + str(sum(t_rewards)/ i))
    #     print("Success rate: " + str(sum(t_rewards)/ num_episodes))

    def e_greedy(self, state, Q, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(1, self.nActions + 1)
        else:
            # action = np.argmax(Q[state, :]) +1
            action = Q.loc[state, :].idxmax(axis=1)
        return action

    def SARSA(self, num_episodes, alpha, gamma, epsilon, decay_rate):
        np.random.seed(20)
        modify = (self.nStates, self.nActions)
        Q = pd.DataFrame(np.zeros(modify, dtype=float), index=range(1, self.nStates + 1),
                         columns=range(1, self.nActions + 1))
        # Q = np.zeros((self.nStates,self.nActions))
        # initialization of Q function
        t_rewards = []

        for i in range(num_episodes):
            epsilon = epsilon * decay_rate
            self.reset(True)
            state = self.observation[0]
            t_reward = 0
            action = self.e_greedy(state, Q, epsilon)
            done = False
            while not done:
                # self.render()
                # Getting nextstate
                next_state, reward, done = self.step(action)
                # getting nextaction
                next_action = self.e_greedy(next_state, Q, epsilon)
                # learning the Q value
                Q.loc[state, action] = Q.loc[state, action] + alpha * (
                            reward + gamma * Q.loc[next_state, next_action] - Q.loc[state, action])
                # Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state,next_action] - Q[state, action])
                #
                t_reward += reward
                state = next_state
                action = next_action

                t_rewards.append(t_reward)
        accumReward = (sum(t_rewards) / num_episodes)
        print("Average", str(accumReward))
        self.close()
        return Q

    def Q_learning(self, alpha, gamma, epsilon, num_episodes, decay_rate):
        modify = (self.nStates, self.nActions)
        Q = pd.DataFrame(np.zeros(modify, dtype=float), index=range(1, self.nStates + 1),
                         columns=range(1, self.nActions + 1))
        t_rewards = []
        for i in range(num_episodes):
            epsilon = epsilon * decay_rate
            self.reset(True)
            state = state = self.observation[0]
            action = self.e_greedy(state, Q, epsilon)

            next_state, reward, done = self.step(action)
            Q.loc[state, action] = Q.loc[state, action] + alpha * (
                        reward + gamma * Q.loc[next_state, next_action] - Q.loc[state, action])
            # Q[(state, action)] = Q[(state, action)] + alpha * (reward + gamma * np.max(Q[next_state,:]) - Q[(state, action)])

            state = next_state

            if done:
                t_rewards.append(reward)
                break
        return Q

    def plot_actionValues_OLA(self, Q):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        action_plot_dict = {'1': (0.5, 0.25),
                            '2': (0.75, 0.5),
                            '3': (0.5, 0.75),
                            '4': (0.25, 0.5)}
        fig = plt.plot(1)
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateGoal:
                    for action in Q.columns:
                        w, h = action_plot_dict[action]
                        plt.text(i + w, j - h, str(self._truncate(Q.loc[k + 1, action], 3)), fontsize=8,
                                 horizontalalignment='center', verticalalignment='center')

                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")