import math as m
import random as rm
import numpy as np
import scipy.stats as stats

import torch

from data_generator import *
from local_randomizer import *
from sliding_window import *
from metric import *
from DDPG import Agent as DDPGAgent
from DQN import Agent as DQNAgent

eps = np.finfo(np.float64).eps

itemset_len = 1000
k = 100

user_num = 10000
data_len = 10

omega = 5
epsilon = 5
data_budget = 0.9*epsilon
hit_budget = 0.1*epsilon

total_step = 1000

data_randomizer = RAPPOR(itemset_len, data_budget/omega)
hit_randomizer = Harmony(hit_budget/omega, 0, data_len)
generator = ZipfGenerator(itemset_len, user_num, data_len)
window = SlidingWindow(omega, data_budget)

data_randomizer.enable_render()
generator.enable_render()

hits = np.zeros(user_num)
last_true_counts = np.ones(itemset_len)

record = np.zeros((total_step, 6))

# first release
window.update(data_budget/omega)
interval = 1
# collect data
data = generator.generate()
generator.transit()
true_counts = calc_counts(data, itemset_len)
private_counts = data_randomizer.randomize_group(data)
counts = data_randomizer.aggregate(private_counts)
top_k = calc_top_k(counts, k)
# collect hit num
for i in range(user_num):
    hits[i] = len(np.intersect1d(data[i], top_k))
private_hits = hit_randomizer.randomize_group(hits)
hit_ratio = hit_randomizer.aggregate(private_hits)/data_len

def run_model():
    global counts
    global baseline
    baseline_count = 1
    allocator = DQNAgent(state_space_dim=12, action_space_dim=omega+1)
    allocator.policy_net = torch.load('model/first_model.pkl')
    allocator.policy_net.eval()
    allocator.target_net = torch.load('model/first_model.pkl')
    allocator.target_net.eval()
    state = np.array([0]*9 + [1, epsilon/omega, window.remain()])
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        data = generator.generate()
        generator.transit()
        true_counts = calc_counts(data, itemset_len)
        # allocate budget
        action = allocator.act(state)
        remain = window.remain()
        budget = min(action*epsilon/omega, remain)
        print('remain:', remain, 'action:', action*epsilon/omega, 'budget:', budget)
        window.update(budget)
        if budget!=0:
            data_randomizer.set_budget(budget)
            # randomize and aggregate
            private_counts = data_randomizer.randomize_group(data)
            new_counts = data_randomizer.aggregate(private_counts)
            next_state = np.concatenate((state[3:], [mean_absolute_error(new_counts, counts), budget, remain]))
            counts = new_counts
            if budget==epsilon/omega:
                baseline_count += 1
                baseline += (calc_CHR(true_counts, calc_top_k(counts, k)) - baseline)/baseline_count
        else:
            next_state = np.concatenate((state[3:], [1, budget, remain]))
        # record metrics
        record[step] = utility_metrics(counts, true_counts, k)
        # learn
        reward = calc_CHR(true_counts, calc_top_k(counts, k)) - baseline
        print('state:', state)
        print('action:', action*epsilon/omega, 'remain:', remain)
        print('next_state:', next_state)
        print('reward:', reward, calc_CHR(true_counts, calc_top_k(counts, k)), baseline, (remain - action*epsilon/omega)/epsilon)
        allocator.memory.push(state, action, next_state, reward)
        allocator.learn()
        state = next_state
    print('Mean:', record.mean(0))

def run_RLBD_continuous_model():
    global counts
    global top_k
    global interval
    global hit_ratio
    global true_counts, last_true_counts
    allocator = DDPGAgent(state_space_dim=3, action_space_dim=1)
    allocator.actor = torch.load('model/rlbd_continuous_actor.pt')
    allocator.actor_target = torch.load('model/rlbd_continuous_actor.pt')
    allocator.critic = torch.load('model/rlbd_continuous_actor.pt')
    allocator.critic_target = torch.load('model/rlbd_continuous_actor.pt')
    state = np.array([interval, min(stats.entropy(last_true_counts + eps, true_counts + eps), 6), window.remain()])
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        data = generator.generate()
        generator.transit()
        true_counts = calc_counts(data, itemset_len)
        # allocate budget
        action = allocator.select_action(state)
        budget = action if action>=1 else 0
        window.update(budget)
        if budget!=0:
            interval = 1
            data_randomizer.set_budget(budget)
            # collect data
            private_counts = data_randomizer.randomize_group(data)
            new_counts = data_randomizer.aggregate(private_counts)
            next_state = np.array([interval, min(stats.entropy(last_true_counts + eps, true_counts + eps), 6), window.remain()])
            counts = new_counts
            top_k = calc_top_k(counts, k)
        else:
            interval += 1
            next_state = np.array([interval, min(stats.entropy(last_true_counts + eps, true_counts + eps), 6), window.remain()])
        # generate hit num
        for i in range(user_num):
            hits[i] = len(np.intersect1d(data[i], top_k))
        # collect hit num
        private_hits = hit_randomizer.randomize_group(hits)
        hit_ratio = hit_randomizer.aggregate(private_hits)/data_len
        # record metrics
        record[step] = utility_metrics(counts, calc_counts(data, itemset_len), k)
        # learn
        reward = hit_ratio
        print('state:', state)
        print('action:', action)
        print('next_state:', next_state)
        print('reward:', reward, calc_CHR(true_counts, calc_top_k(counts, k)))
        state = next_state
    print('Mean:', record.mean(0))
