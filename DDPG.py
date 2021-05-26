import numpy as np
import random

from collections import namedtuple
from itertools import count

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import gym

class Actor(nn.Module):
    def __init__(self, 
            state_space_dim,
            action_space_dim, 
            hidden_size=256):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_space_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_space_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        action = torch.sigmoid(x)*state[0, -1]
        return action

class Critic(nn.Module):
    def __init__(self, 
            state_space_dim,
            action_space_dim, 
            hidden_size=256):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_space_dim, hidden_size)
        self.linear2 = nn.Linear(action_space_dim, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = self.linear1(state)
        y = self.linear2(action)
        state_action_value = self.linear3(F.relu(x+y))
        return state_action_value

class Agent(object):
    def __init__(self,
            state_space_dim, 
            action_space_dim, 
            capacity=10000, 
            batch_size=64, 
            actor_learning_rate=0.005,
            critic_learning_rate=0.005,
            discount_rate=0.95,  
            target_update_rate=0.01, 
            exploration_noise=5, 
            noise_decay_rate=0.95):
        self.batch_size = batch_size
        self.gamma = discount_rate
        self.tau = target_update_rate
        self.noise = exploration_noise
        self.decay_rate = noise_decay_rate

        self.actor = Actor(state_space_dim, action_space_dim)
        self.critic = Critic(state_space_dim, action_space_dim)

        self.actor_target = Actor(state_space_dim, action_space_dim)
        self.critic_target = Critic(state_space_dim, action_space_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = ReplayMemory(capacity)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        action = np.clip(np.random.normal(action, self.noise), 0, state[0, -1])
        return action
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        samples = self.memory.sample(self.batch_size)
        state, action, next_state, reward = zip(*samples)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float).view(self.batch_size, -1)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float).view(self.batch_size, -1)

        # critic learn
        next_action = self.actor_target(next_state).detach()
        y_true = reward + self.gamma*self.critic_target(next_state, next_action).detach()
        y_pred = self.critic(state, action)

        critic_loss_fn = nn.MSELoss()
        critic_loss = critic_loss_fn(y_pred, y_true)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor learn
        actor_loss = - torch.mean(self.critic(state, self.actor(state)))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft update target
        for target_param, param  in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param  in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # decay exploration noise
        self.noise *= self.decay_rate

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
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

if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    action_range = {"high":env.action_space.high, 
                    "low":env.action_space.low}
    
    agent = Agent(state_space_dim, action_space_dim)
    
    RENDER = False

    for episode in range(100):
        state = env.reset()
        episode_reward = 0
        for t in count():
            if RENDER:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.learn()
            episode_reward += reward
            if done:
                print('Episode:', episode, ' Reward: %i' % int(episode_reward))
                break