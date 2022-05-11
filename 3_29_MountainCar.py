# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gym

# A simple agent
class BespokeAgent:
    def __init__(self, env):
        pass
    def decide(self, observation): # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
             0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action # 返回动作
    def learn(self, *args): # 学习
        pass

def play_montercarlo(env, agent, render=False, train=False):
    episode_reward=0
    observation  = env.reset()
    while True:
        if render :
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, info = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    print('观测空间:', env.observation_space)
    print('动作空间：', env.action_space)
    print('观测范围：', env.observation_space.low, '\t', env.observation_space.high)
    print('动作数：', env.action_space.n)

    agent = BespokeAgent(env)
    env.seed(2022)
    episode_reward = play_montercarlo(env, agent, render=True)
    print('回合奖励：', episode_reward)
    env.close()
