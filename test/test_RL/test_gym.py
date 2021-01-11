"""
 @author: AlexWang
 @date: 2020/11/3 8:02 PM
 @Email: alex.wj@alibaba-inc.com

 openAI： https://openai.com/
"""
import gym
from gym import spaces
from gym import envs


def openAI_env():
    print(envs.registry.all())  # 可以使用的所有环境  https://gym.openai.com/envs/#atari
    # space
    space = spaces.Discrete(8)  # Set with 8 elements {0, 1, 2, ..., 7}
    x = space.sample()
    print(x)
    assert space.contains(x)
    assert space.n == 8

    env = gym.make('CartPole-v0')
    print(env.action_space)  # 可以采取的action空间, Discrete(2)表示[0,1], Discrete(8)表示[0, 1, 2, ..., 7]
    print(env.observation_space)  # 可以观察到的环境空间
    print(env.observation_space.high)
    print(env.observation_space.low)


def cart_pole():
    """
    observation (object): 描述环境的对象, 例如 图像像素信息, 机器人的角度和速度;
    reward (float): 上一个action得到的reward;
    done (boolean): True表示需要 reset 环境, 例如当 乒乓球运动中 失败;
    info (dict):debug信息,不能用于agent的学习;
    :return:
    """
    # [ 0.09409782  1.35118069 -0.14985008 -2.2165633 ]
    # [ 0.12112144  1.54738412 -0.19418135 -2.55146154]
    # Episode finished after 15 timesteps
    # [ 0.11185343  0.60433267 -0.17907317 -1.30391457]
    # [ 0.12394008  0.41187566 -0.20515146 -1.07221196]
    # Episode finished after 27 timesteps
    # [ 0.12680147  0.73063827 -0.17171857 -1.13211941]
    # [ 0.14141423  0.53812851 -0.19436096 -0.89783927]
    # Episode finished after 32 timesteps
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == '__main__':
    openAI_env()
    cart_pole()
