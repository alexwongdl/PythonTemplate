"""
 @author: AlexWang
 @date: 2021/1/18 8:24 PM
 @Email: alex.wj@alibaba-inc.com
"""
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
import random

num_states = 4 * 12  # The number of states in simply the number of "squares" in our grid world, in this case 4 * 12
num_actions = 4  # We have 4 possible actions, up, down, right and left
actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']


def change_range(values, vmin=0, vmax=1):
    start_zero = values - np.min(values)
    return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin


class GridWorld:
    terrain_color = dict(normal=[127 / 360, 0, 96 / 100],
                         objective=[26 / 360, 100 / 100, 100 / 100],
                         cliff=[247 / 360, 92 / 100, 70 / 100],
                         player=[344 / 360, 93 / 100, 100 / 100])

    def __init__(self):
        self.player = None
        self._create_grid()
        self._draw_grid()
        self.num_steps = 0

    def _create_grid(self, initial_grid=None):
        self.grid = self.terrain_color['normal'] * np.ones((4, 12, 3))
        self._add_objectives(self.grid)

    def _add_objectives(self, grid):
        grid[-1, 1:11] = self.terrain_color['cliff']
        grid[-2, 1:11] = self.terrain_color['cliff']
        grid[-3, 5] = self.terrain_color['cliff']
        grid[-4, 8] = self.terrain_color['cliff']
        grid[-1, -1] = self.terrain_color['objective']

    def _draw_grid(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.grid(which='minor')
        self.q_texts = [self.ax.text(*self._id_to_position(i)[::-1], '0',
                                     fontsize=11, verticalalignment='center',
                                     horizontalalignment='center') for i in range(12 * 4)]

        self.im = self.ax.imshow(hsv_to_rgb(self.grid), cmap='terrain',
                                 interpolation='nearest', vmin=0, vmax=1)
        self.ax.set_xticks(np.arange(12))
        self.ax.set_xticks(np.arange(12) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(4))
        self.ax.set_yticks(np.arange(4) - 0.5, minor=True)

    def reset(self):
        self.player = (3, 0)
        self.num_steps = 0
        return self._position_to_id(self.player)

    def step(self, action):
        # Possible actions
        if action == 0 and self.player[0] > 0:
            self.player = (self.player[0] - 1, self.player[1])
        if action == 1 and self.player[0] < 3:
            self.player = (self.player[0] + 1, self.player[1])
        if action == 2 and self.player[1] < 11:
            self.player = (self.player[0], self.player[1] + 1)
        if action == 3 and self.player[1] > 0:
            self.player = (self.player[0], self.player[1] - 1)

        self.num_steps = self.num_steps + 1
        # Rules
        if all(self.grid[self.player] == self.terrain_color['cliff']):
            reward = -100
            done = True
        elif all(self.grid[self.player] == self.terrain_color['objective']):
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return self._position_to_id(self.player), reward, done

    def _position_to_id(self, pos):
        ''' Maps a position in x,y coordinates to a unique ID '''
        return pos[0] * 12 + pos[1]

    def _id_to_position(self, idx):
        return (idx // 12), (idx % 12)

    def render(self, q_values=None, action=None, max_q=False, colorize_q=False):
        assert self.player is not None, 'You first need to call .reset()'

        if colorize_q:
            assert q_values is not None, 'q_values must not be None for using colorize_q'
            grid = self.terrain_color['normal'] * np.ones((4, 12, 3))
            values = change_range(np.max(q_values, -1)).reshape(4, 12)
            grid[:, :, 1] = values
            self._add_objectives(grid)
        else:
            grid = self.grid.copy()

        grid[self.player] = self.terrain_color['player']
        self.im.set_data(hsv_to_rgb(grid))

        if q_values is not None:
            xs = np.repeat(np.arange(12), 4)
            ys = np.tile(np.arange(4), 12)

            for i, text in enumerate(self.q_texts):
                if max_q:
                    q = max(q_values[i])
                    txt = '{:.2f}'.format(q)
                    text.set_text(txt)
                else:
                    actions = ['U', 'D', 'R', 'L']
                    txt = '\n'.join(['{}: {:.2f}'.format(k, q) for k, q in zip(actions, q_values[i])])
                    text.set_text(txt)

        if action is not None:
            self.ax.set_title(action, color='r', weight='bold', fontsize=32)

        plt.pause(2)  # 暂停


def play(q_values):
    # simulate the environent using the learned Q values
    env = GridWorld()
    state = env.reset()
    done = False

    while not done:
        # Select action
        action = e_greedy_exploration(q_values, state, 0.0)
        # Do the action
        next_state, reward, done = env.step(action)

        # Update state and action
        state = next_state

        env.render(q_values=q_values, action=actions[action], colorize_q=True)


def e_greedy_exploration(q_values, state, exploration_rate):
    """

    :param q_values:
    :param state:
    :param exploration_rate:
    :return:
    """
    if np.random.random() < exploration_rate:
        return np.random.choice(4)
    else:
        return np.argmax(q_values[state])


def test_cliffwalk():
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    actions_dict = {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}

    env = GridWorld()
    state = env.reset()
    done = False

    while not done:
        # Select action
        action = random.randint(0, 3)
        print(actions[action])
        # Do the action
        next_state, reward, done = env.step(action)

        # Update state and action
        state = next_state
        env.render()


def sarsa(env, num_episodes=500, render=True, exploration_rate=0.1, learning_rate=0.5, gamma=0.9):
    q_values_sarsa = np.zeros((num_states, num_actions))
    ep_rewards = []

    for _ in range(num_episodes):
        state = env.reset()  # 初始状态0, start--> action--> next_state --> next_action
        done = False
        reward_sum = 0
        # 选取一个action
        action = e_greedy_exploration(q_values_sarsa, state, exploration_rate)

        while not done:
            # Do the action
            next_state, reward, done = env.step(action)
            reward_sum += reward
            # 获取下一个action
            next_action = e_greedy_exploration(q_values_sarsa, next_state, exploration_rate)

            # 利用Q-table 计算当前td target
            td_target = reward + gamma * q_values_sarsa[next_state][next_action]
            td_error = td_target - q_values_sarsa[state][action]
            # Update q value
            q_values_sarsa[state][action] += learning_rate * td_error

            # Update state and action
            state = next_state
            action = next_action

            if render:
                env.render(q_values_sarsa, action=actions[action], colorize_q=True)

        ep_rewards.append(reward_sum)
    return ep_rewards, q_values_sarsa


def q_learning(env, num_episodes=500, render=True, exploration_rate=0.1, learning_rate=0.5, gamma=0.9):
    q_values_q_learning = np.zeros((num_states, num_actions))
    ep_rewards = []

    for _ in range(num_episodes):
        state = env.reset()  # 初始状态0, start--> action--> next_state --> next_action = argmax
        done = False
        reward_sum = 0

        while not done:
            # 选取一个action
            action = e_greedy_exploration(q_values_q_learning, state, exploration_rate)
            # Do the action
            next_state, reward, done = env.step(action)
            reward_sum += reward

            # 利用Q-table 计算当前td target
            td_target = reward + gamma * np.max(q_values_q_learning[next_state])
            td_error = td_target - q_values_q_learning[state][action]
            # Update q value
            q_values_q_learning[state][action] += learning_rate * td_error

            # Update state and action
            state = next_state

            if render:
                env.render(q_values_q_learning, action=actions[action], colorize_q=True)

        ep_rewards.append(reward_sum)
    return ep_rewards, q_values_q_learning


if __name__ == '__main__':
    # test_cliffwalk()

    ############ Q-Learning
    env = GridWorld()
    ep_rewards, q_values_q_learning = q_learning(env, exploration_rate=0.1, num_episodes=50000,
                                                 render=False, learning_rate=1, gamma=0.9)
    print("q_learning ep rewrods:{}".format(ep_rewards))

    q_learning_rewards = []
    for i in range(1):
        ep_rewards, q_values_q_learning = q_learning(env, render=False, exploration_rate=0.1,
                                                     learning_rate=1)
        print(i, ep_rewards)
        q_learning_rewards.append(ep_rewards)

    q_learning_rewards = np.asarray(q_learning_rewards)
    print(q_learning_rewards)
    print("shape of q_learning_rewards:{}".format(q_learning_rewards.shape))

    avg_rewards = np.mean(q_learning_rewards, axis=0)
    print("shape of q_learning avg_rewards:{}".format(avg_rewards.shape))

    print(avg_rewards)
    mean_reward = [np.mean(avg_rewards)] * len(avg_rewards)

    fig, ax = plt.subplots()
    ax.set_xlabel('Episodes using Sarsa')
    ax.set_ylabel('Rewards')
    ax.plot(avg_rewards)
    ax.plot(mean_reward, 'g--')
    plt.show()

    print('Mean Reward using q_learning: {}'.format(mean_reward[0]))
    play(q_values_q_learning)

    ############ Sarsa

    # env = GridWorld()
    # ep_rewards, q_values_sarsa = sarsa(env, exploration_rate=0.1, num_episodes=500,
    #                                    render=False, learning_rate=0.8, gamma=0.9)
    # print("sarsa ep rewrods:{}".format(ep_rewards))
    #
    # sarsa_rewards = []
    # for i in range(10):
    #     ep_rewards, q_values_sarsa = sarsa(env, render=False, exploration_rate=0.2)
    #     print(i, ep_rewards)
    #     sarsa_rewards.append(ep_rewards)
    #
    # # aa = [sarsa(env, render=False, exploration_rate=0.2) for _ in range(10)]
    # # sarsa_rewards, _ = zip(*aa)
    # sarsa_rewards = np.asarray(sarsa_rewards)
    # print(sarsa_rewards)
    # print("shape of sarsa_rewards:{}".format(sarsa_rewards.shape))
    #
    # # sarsa_rewards, _ = zip(*[sarsa(env, render=False, exploration_rate=0.2) for _ in range(10)])
    # # for reward in sarsa_rewards:
    # #     print(sarsa_rewards)
    #
    # avg_rewards = np.mean(sarsa_rewards, axis=0)
    # print("shape of avg_rewards:{}".format(avg_rewards.shape))
    #
    # print(avg_rewards)
    # mean_reward = [np.mean(avg_rewards)] * len(avg_rewards)
    #
    # # fig, ax = plt.subplots()
    # # ax.set_xlabel('Episodes using Sarsa')
    # # ax.set_ylabel('Rewards')
    # # ax.plot(avg_rewards)
    # # ax.plot(mean_reward, 'g--')
    # # plt.show()
    #
    # print('Mean Reward using Sarsa: {}'.format(mean_reward[0]))
    # play(q_values_sarsa)
