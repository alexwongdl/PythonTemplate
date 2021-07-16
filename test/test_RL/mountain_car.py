"""
 @author: AlexWang
 @date: 2021/4/5 10:55 AM
 @Email: alex.wj@alibaba-inc.com

 gym: https://gym.openai.com/envs/MountainCarContinuous-v0/
 没有用到梯度下降
"""
import pickle
import numpy as np

import gym
from gym import wrappers

n_states = 40
n_actions = 3


def obs_to_state(env, obs):
    """ Maps an observation to state 离散化 """
    # we quantify the continous state space into discrete space
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    a = a if a < n_states else n_states - 1
    b = b if b < n_states else n_states - 1
    return a, b


def e_greedy_exploration(q_values, state_a, state_b, exploration_rate):
    """
    :param q_values:
    :param state:
    :param exploration_rate:
    :return:
    """
    if np.random.random() < exploration_rate:
        return np.random.choice(3)
    else:
        return np.argmax(q_values[state_a][state_b])


def q_learning(env, num_episodes=500, render=True, exploration_rate=0.1, learning_rate=0.5, gamma=0.9):
    q_values_q_learning = np.zeros((n_states, n_states, n_actions))
    ep_rewards = []
    initial_lr = 1.0  # Learning rate
    min_lr = 0.003

    for i in range(num_episodes):
        learning_rate = max(min_lr, initial_lr * (0.85 ** (i // 100)))
        state = env.reset()  # 初始状态0, start--> action--> next_state --> next_action = argmax
        state_a, state_b = obs_to_state(env, state)
        done = False
        reward_sum = 0

        while not done:
            # 选取一个action
            action = e_greedy_exploration(q_values_q_learning, state_a, state_b, exploration_rate)
            # Do the action
            next_state, reward, done, _ = env.step(action)
            next_state_a, next_state_b = obs_to_state(env, next_state)
            reward_sum += reward

            # 利用Q-table 计算当前td target
            td_target = reward + gamma * np.max(q_values_q_learning[next_state_a][next_state_b])
            td_error = td_target - q_values_q_learning[state_a][state_b][action]
            # Update q value
            q_values_q_learning[state_a][state_b][action] += learning_rate * td_error

            # Update state and action
            state = next_state
            state_a, state_b = obs_to_state(env, state)

            if render:
                env.render()

        ep_rewards.append(reward_sum)
    return ep_rewards, q_values_q_learning


def run_episode(env, q_values, gamma=1.0, render=True):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    a, b = obs_to_state(env, obs)
    total_reward = 0
    step_idx = 0
    while True:
        action = np.argmax(q_values[a][b])
        obs, reward, done, info = env.step(action)
        a, b = obs_to_state(env, obs)
        print(action, a, b, total_reward)

        total_reward += (gamma ** step_idx * reward)
        step_idx += 1

        if render:
            env.render()
        if done:
            break
    return total_reward


def test_mountain_car():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    state = env.reset()
    print("state:{}".format(state))  # [-0.42251278  0.        ]
    done = False

    while not done:
        action = env.action_space.sample()  # 0:全速往左,1:不动,2:全速往右
        # action = -1 # -1 (<class 'int'>) invalid
        # action = 0

        observation, reward, done, info = env.step(action)
        a, b = obs_to_state(env, observation)
        env.render()
        print(observation, action, reward, done, a, b)  # [-0.40611668 -0.0008691 ] 1 -1.0 False 17 19
        if done:
            break

    env_low = env.observation_space.low
    env_high = env.observation_space.high
    print(env_low, env_high)  # [-1.2  -0.07] [0.6  0.07]
    print(env.action_space)  # Discrete(3)
    print(env.observation_space)  # Box(-1.2000000476837158, 0.6000000238418579, (2,), float32)


def test_q_learning():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    ep_rewards, q_values_q_learning = q_learning(env, num_episodes=10000, render=False, exploration_rate=0.1,
                                                 learning_rate=0.5, gamma=0.9)
    pickle.dump(q_values_q_learning, open('mountain_car.pkl', 'wb'))
    print(ep_rewards)
    total_reward = run_episode(env, q_values_q_learning)
    print(total_reward)


def run_q_learning():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    q_values_q_learning = pickle.load(open('mountain_car.pkl', 'rb'))
    total_reward = run_episode(env, q_values_q_learning)
    print(total_reward)


if __name__ == '__main__':
    # test_mountain_car()
    test_q_learning()
    # run_q_learning()