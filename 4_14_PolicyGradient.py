# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch.nn as nn
import gym
from torch.optim import Adam
import torch
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torch.distributions.categorical import Categorical

def get_params():
    import argparse
    parser = argparse.ArgumentParser(description='Entry Point of the code')

    # Train settings
    # parser.add_argument('--train_eps', type=int, default=10000)
    # parser.add_argument('--test_eps',  type=int, default=30)
    # parser.add_argument('--batch_size',  type=int, default=8)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--step_per_epoch',  type=int, default=10000)

    # Model settings
    parser.add_argument('--algo_name', type=str, default='QLearning')
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--wd', type=float, default=1e-3)
    args, _ = parser.parse_known_args()
    return args


def accmulated_reward(reward_lst, gamma):
    """
    get accmulated reward list through the raw reward list
    :param reward_lst:
    :param gamma:
    :return:
    """
    accumulated_reward = reward_lst[:]
    n = len(accumulated_reward)
    for i in range(n - 1, 0, -1):
        accumulated_reward[i - 1] = accumulated_reward[i - 1] + gamma * accumulated_reward[i]

    mean = np.mean(accumulated_reward)
    std = np.std(accumulated_reward)
    accumulated_reward = [(x - mean) / std for x in accumulated_reward]
    return accumulated_reward


class PolicyGradient:
    def __init__(self, args={}):
        self.state_dim = args['state_dim']
        self.action_num =  args['action_num']
        self.gamma = args['gamma']
        self.policy =  nn.Sequential(nn.Linear(self.state_dim, 64), nn.ReLU(), 
                                     nn.Linear(64,  64), nn.ReLU(),
                                     nn.Linear(64,  self.action_num))
        self.optimizer = Adam(self.policy.parameters(), lr=args['lr'], weight_decay=args['wd'])

    # note that state is numpy.ndarray, it should be converted into tensor
    def choose_action(self, state):
        state = torch.from_numpy(state)
        state  = torch.tensor(state, dtype=torch.float32)
        logits = self.policy(state)
        distribution = Categorical(logits=logits)
        a = distribution.sample()
        return a.numpy()

    def update(self, acc_reward_lst, action_lst, state_lst):
        # note: Gradient Desent
        self.optimizer.zero_grad()
        r, a, s= np.array(acc_reward_lst), np.array(action_lst), np.array(state_lst)
        r, a, s = torch.from_numpy(r), torch.from_numpy(a), torch.from_numpy(s)
        r, a, s = torch.tensor(r, dtype=torch.float32),  \
                  torch.tensor(a, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)
        logits = self.policy(s)
        distributions = Categorical(logits=logits)
        logp = distributions .log_prob(a)
        loss = (-logp * r).mean()
        loss.backward()
        self.optimizer.step()

        entropy = distributions.entropy().mean().item()
        return loss.item(), entropy


if __name__ == '__main__':

    args = vars(get_params())
    env = gym.make(args['env_name'])
    state = env.reset()
    print('初始状态：', state)
    env.seed(1)
    print("observation_space:", env.observation_space) # 状态空间是连续的, 是一个4维的vector
    print('action space:', env.action_space.n)
    print('state dim:', env.observation_space.shape)

    state_dim =  env.observation_space.shape[0]
    action_num = env.action_space.n

    args.update({'state_dim':state_dim, 'action_num': action_num})
    agent = PolicyGradient(args)

    state, ep_len, ep_return = env.reset(), 0, 0
    for epoch in range(args['epochs']):
        ep_lens, ep_returns = [], []
        acc_reward_lst, reward_lst, action_lst, state_lst = [], [], [], []
        step = 0
        while step < args['step_per_epoch']:
            cur_acc_reward_lst, cur_reward_lst, cur_action_lst, cur_state_lst = [], [], [], []
            while True:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                ep_len += 1
                step += 1
                ep_return += reward
                cur_state_lst.append(state)
                cur_action_lst.append(action)
                cur_reward_lst.append(reward)
                state = next_state
                if epoch > 10: env.render() #训练的时候不要去render这个环境。
                if done: 
                    ep_lens.append(ep_len)
                    ep_returns.append(ep_return)
                    state, ep_len, ep_return = env.reset(), 0, 0
                    cur_acc_reward_lst = accmulated_reward(cur_reward_lst, args['gamma'])
                    acc_reward_lst += cur_acc_reward_lst
                    reward_lst += cur_reward_lst
                    action_lst += cur_action_lst
                    state_lst += cur_state_lst
                    break
        loss, entropy = agent.update(acc_reward_lst, action_lst, state_lst)
        reward_lst, action_lst, state_lst = [], [], []
        print('Epoch {}'.format(epoch))
        print('Epoch rewards mean: {:.2f}, std:{:.2f}'.format(np.mean(ep_returns), np.std(ep_returns)))
        print('Epoch length mean: {:.2f}, std:{:.2f}'.format(np.mean(ep_lens), np.std(ep_lens)))
        print('Max reward: {:.2f}, Min reward {:.2f}'.format(np.max(ep_returns), np.min(ep_returns)))
        print('Loss: {:.4f}'.format(loss))
        print('Entropy: {:.4f}'.format(entropy))
    env.close()











