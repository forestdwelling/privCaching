import math
import random
import numpy as np

from collections import namedtuple, Counter
from itertools import count
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

class Agent(object):
    def __init__(self, 
            state_space_dim, 
            action_space_dim, 
            learning_rate=0.005, 
            discount_rate=0.95,
            epsilon={"high":0.95,
                "low":0.05, 
                "decay":100}, 
            capacity=10000, 
            batch_size=64, 
            target_update=10):
        self.gamma = discount_rate
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update = target_update
        self.action_space_dim = action_space_dim

        self.policy_net = DQN(state_space_dim, action_space_dim)
        self.target_net = DQN(state_space_dim, action_space_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = learning_rate)
        self.memory = ReplayMemory(capacity)
        self.steps = 0
        
    def select_action(self, state):
        self.steps += 1
        epsilon = self.epsilon["low"] + (self.epsilon["high"]-self.epsilon["low"]) * (math.exp(-1.0 * self.steps/self.epsilon["decay"]))
        if random.random() < epsilon:
            action = random.randrange(self.action_space_dim)
        else:
            state =  torch.tensor(state, dtype=torch.float).view(1,-1)
            action = torch.argmax(self.policy_net(state)).item()
        return action
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        samples = self.memory.sample(self.batch_size)
        state, action, next_state, reward = zip(*samples)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long).view(self.batch_size, -1)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float).view(self.batch_size, -1)

        expected_state_action_values = reward + self.gamma * torch.max(self.target_net(next_state).detach(), dim=1)[0].view(self.batch_size, -1)
        state_action_values = self.policy_net(state).gather(1, action)

        loss_func = nn.MSELoss()
        loss = loss_func(expected_state_action_values, state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    def modify_reward(state):
        x, x_dot, theta, theta_dot = state
        reward1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        reward2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = reward1 + reward2
        return reward

    state_space_dim=env.observation_space.shape[0]
    action_space_dim=env.action_space.n
    agent = Agent(state_space_dim, action_space_dim)

    for episode in range(200):
        state = env.reset()
        episode_reward = 0
        for t in count():
            # env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = modify_reward(next_state)
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.learn()
            episode_reward += reward
            if done:
                print('Episode:', episode, 't:', t, ' Reward: %i' % int(episode_reward))
                break
