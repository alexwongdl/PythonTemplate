"""
 @author: AlexWang
 @date: 2021/1/2 5:48 PM
 https://github.com/cuhkrlcourse/RLexample/tree/master/MDP

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"

冰面很滑, 在一个状态中选择某个方向, 只能保证一定的概率是往这个方向的, 其他还有一定的概率滑到其他方向, 例如:
0 {0: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False)], 1: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 1, 0.0, False)], 2: [(0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False)], 3: [(0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False)]}
在状态0, action=1(DOWN)的时候,[(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 1, 0.0, False)], 有1/3的概率滑到状态 0/1/4.

(0.3333333333333333, 4, 0.0, False) 表示以 1/3 的概率转移到 4,并且得到奖励0.0, 游戏是否结束--False
"""

import numpy as np
import gym


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


# DiscreteEnv 的 step 函数:  P[s][a] == [(probability, nextstate, reward, done), ...]
# def step(self, a):
#     transitions = self.P[self.s][a]
#     i = categorical_sample([t[0] for t in transitions], self.np_random)
#     p, s, r, d = transitions[i]
#     self.s = s
#     self.lastaction = a
#     return (int(s), r, d, {"prob": p})

def run_episode(env, policy, gamma=1.0, render=False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    迭代计算value function
    """
    v = np.zeros(env.env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            policy_a = policy[s]
            print("state:{}, policy_a:{}, trans:{}".format(s, policy_a, env.env.P[s][policy_a]))
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
        if np.sum((np.fabs(prev_v - v))) <= eps:
            # value converged
            break
    return v


def fronzenlake_policy_iteration():
    env_name = 'FrozenLake-v0'  # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    """ Policy-Iteration algorithm """
    # - nS: number of states
    # - nA: number of actions
    # - P: transitions (*)
    # - isd: initial state distribution (**)
    # 从 nA中随机选取 nS 个数

    # ****** policy iterater ******
    policy = np.random.choice(env.env.nA, size=(env.env.nS))  # 随机初始化一组policy
    print(policy)
    max_iterations = 200000
    gamma = 1.0
    same_times = 0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if np.all(policy == new_policy):
            same_times += 1
            if same_times >= 10:
                print('Policy-Iteration converged at step %d.' % (i + 1))
                break
        else:
            same_times = 0
        policy = new_policy
    optimal_policy = policy
    print("optimal_policy:")
    print(optimal_policy)

    scores = evaluate_policy(env, optimal_policy, gamma=1.0)
    print('Average scores = ', np.mean(scores))


def value_iteration(env, gamma=1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return v


def fronzenlake_value_iteration():
    env_name = 'FrozenLake-v0'  # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    gamma = 1.0
    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(env, optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)


def frozen_lake_test():
    """
        def step(self, action):Accepts an action and returns a tuple (observation, reward, done, info)
        reset: 重置环境
        render: render其实就相当于一个渲染的引擎, 没有render, 也是可以运行的. 但是render可以为了便于直观显示当前环境中物体的状态, 也是为了便于我们进行代码的调试. 不然只看着一堆数字的observation, 我们也是不知道实际情况怎么样了.
        close:
        seed: 环境随机数字生成器的seed
    :return:
    """
    env_name = 'FrozenLake-v0'  # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    print(env.env.P)
    print(env.env.nS)

    for state in env.env.P:
        print(state, env.env.P[state])


if __name__ == '__main__':
    frozen_lake_test()
    print("policy iteration:")
    fronzenlake_policy_iteration()
    print("value iteration:")
    fronzenlake_value_iteration()
