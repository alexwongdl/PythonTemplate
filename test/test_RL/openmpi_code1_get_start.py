"""
 @author: AlexWang
 @date: 2022/1/9 5:06 下午
 @Email: alex.wj@alibaba-inc.com
"""
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

def CartPole_PPO2():
    env = gym.make('CartPole-v1')
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


def LunarLander_v2_DQN(): #TODO : 报错
    # Create environment
    env = gym.make('LunarLander-v2')

    # Instantiate the agent
    model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
    # Train the agent
    model.learn(total_timesteps=100000)
    # Save the agent
    model.save("dqn_lunar")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = DQN.load("dqn_lunar")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(mean_reward, std_reward)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    CartPole_PPO2()
    # LunarLander_v2_DQN()