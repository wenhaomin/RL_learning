# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gym
import turtle
import random
import math
####################################################
# 装饰器
class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.t = None
        self.unit = 50
        self.max_x = 12
        self.max_y = 4

    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            for i in range(1, self.max_x - 1):
                self.draw_box(i, 0, 'black')
            self.draw_box(self.max_x - 1, 0, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)
####################################################


def get_params():
    import argparse
    parser = argparse.ArgumentParser(description='Entry Point of the code')

    # Train settings
    parser.add_argument('--train_eps', type=int, default=400)
    parser.add_argument('--test_eps',  type=int, default=30)

    # Model settings
    parser.add_argument('--algo_name', type=str, default='QLearning')
    parser.add_argument('--env_name', type=str, default='CliffWalking-v0')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epsilon_start', type=float, default=0.95)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epslion_decay', type=int, default=300)
    args, _ = parser.parse_known_args()
    return args


class QLearning:
    def __init__(self, args={}):
        self.state_num = args['state_num']
        self.action_num = args['action_num']
        self.Q_table = np.zeros((self.state_num, self.action_num)) #这里初始化是全0好，还是随机好？
        self.gamma = args.get('gamma', 1)
        self.lr = args.get('lr', 0.01)

        self.epsilon_start = args.get('epsilon_start', 0.95)
        self.epsilon_end = args.get('epslion_end', 0.01)
        self.epsilon_decay = args.get('epslion_decay', 100)
        self.epsilon = self.epsilon_start
        self.sample_cnt = 1


    def choose_action(self, state): #策略选择
        self.sample_cnt += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_cnt // self.epsilon_decay)
        # epsilon是会递减的，这里选择指数递减
        if random.uniform(0, 1) < self.epsilon:
            # epsilon表示探索的概率，epsilon越小表示探索概率越小
            action =  np.random.choice(self.action_num)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def update(self, state, action,  next_state, reward, done): # 策略更新
        """
        update the Q table
        :return:
        Q(s,a) = Q(s,a) + lr * ( r + \gamma *  max_a{Q(s', a)} - Q(s,a))
        state: s,
        next_state: s'
        action: a
        reward: r
        """
        Q_predict = self.Q_table[state][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * max(self.Q_table[next_state])
        # self.Q_table[state] += self.lr * (Q_target - Q_predict) # 注意：Q table更新的下标
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)


if __name__ == '__main__':
    args = vars(get_params())

    env = gym.make(args['env_name'])
    env = CliffWalkingWapper(env)
    state = env.reset()
    print('初始状态：', state)
    env.seed(1)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    args.update({'state_num': n_states, 'action_num': n_actions})
    agent = QLearning(args)
    rewards = []
    ma_rewards = []
    for i_ep in range(args['train_eps']):
        state = env.reset()
        ep_r  = []
        while True:
            action = agent.choose_action(state) # The agent choose an action
            next_state, reward, done, info = env.step(action) # The environment update
            agent.update(state, action, next_state, reward, done)
            # update the policy (i.e., Q table), Temporal Difference
            env.render()
            if agent.sample_cnt % 1000 == 0:
                print('sample:',agent.sample_cnt, 'epsilon:', agent.epsilon)
            state = next_state
            ep_r.append(reward)
            if done: break
        rewards.append(sum(ep_r))
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9 + sum(ep_r) * 0.1)
        else:
            ma_rewards.append(sum(ep_r))# ma_rewards为空的时候，直接添加这个；
        print('Epoch rewards:', sum(ep_r))
        env.close()

