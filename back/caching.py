import math
import random
import numpy as np

from collections import namedtuple, Counter
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def local_randomizer(data, itemset_len, budgets, rate=False):
    p = np.full(len(budgets), 0.5)
    q = 1/(np.exp(budgets)+1)
    release_data = np.zeros((len(data), itemset_len))
    iterator = range(len(data))
    if rate:
        print('Perturbating users\' data...')
        iterator = tqdm(iterator)
    for i in iterator:
        release_data[i] = np.zeros(itemset_len)
        release_data[i][data[data!=0]-1] = 1
        for j in range(itemset_len):
            if release_data[i][j]==1:
                if random.random()<1-p[i]:
                    release_data[i][j] = 0
            else:
                if random.random()<q[i]:
                    release_data[i][j] = 1
    return release_data

def sampling_randomizer(data, itemset_len, budgets, rate=False):
    p = 0.5
    q = 1/(np.exp(budgets)+1)
    release_data = np.zeros((len(data), itemset_len))
    iterator = range(len(data))
    if rate:
        print('Perturbating users\' data...')
        iterator = tqdm(iterator)
    for i in iterator:
        item = random.choice(data[i])
        if item!=0:
            release_data[i][item-1] = 1
        for j in range(itemset_len):
            if release_data[i][j]==1:
                if random.random()<1-p:
                    release_data[i][j] = 0
            else:
                if random.random()<q[i]:
                    release_data[i][j] = 1
    return release_data

def personalized_frequency_estimation(release_data, itemset_len, budgets):
    regime = np.array(list(set(budgets)))
    p = np.full(len(regime), 0.5)
    q = 1/(np.exp(regime) + 1)
    weight = (p-q)/(q*(1-q))
    weight = weight/weight.sum()
    regime_freq = np.zeros((len(regime), itemset_len))
    for i in range(len(regime)):
        regime_freq[i] = release_data[budgets==regime[i]].sum(0)
        regime_freq[i] = (regime_freq[i] - (budgets==regime[i]).sum()*q[i])/(p[i]-q[i])
    return np.dot(weight, regime_freq)

def personalized_frequency_estimation_weight(release_data, itemset_len, budgets):
    p = np.full(len(budgets), 0.5)
    q = 1/(np.exp(budgets) + 1)
    weight = (p-q)/(q*(1-q))
    weight = weight/weight.sum()
    freq = np.dot(weight, release_data)
    freq = len(release_data)*(freq - np.dot(weight, q))/np.dot(weight, p-q)
    return freq

def personalized_frequency_estimation_sample(data, itemset_len, budgets, times, threshold='mean'):
    sample_freq = np.zeros((times, itemset_len))
    freq_weight = np.zeros(times)
    for i in range(times):
        if threshold=='mean':
            t = budgets.mean()
        elif threshold=='max':
            t = budgets.max()
        p = 0.5
        q = 1/(math.exp(t) + 1)
        user_num = len(data)
        print(f'Sample {i+1}: Sampling users...')
        sample_prob = np.zeros(user_num)
        sample_flag = np.zeros(user_num, dtype=bool)
        weight = np.zeros(user_num)
        for j in range(user_num):
            if budgets[j]>=t:
                sample_prob[j] = 1
                sample_flag[j] = True
            elif budgets[j]>math.log((math.exp(t)+1)/2):
                sample_prob[j] = (2*math.exp(budgets[j])-math.exp(t)-1)/(math.exp(budgets[j])-1)
                sample_flag[j] = np.random.choice(a=[True, False], p=[sample_prob[j], 1-sample_prob[j]])

        print('Sample rate:', sample_flag.sum()/len(sample_flag))
        sample_data = data[sample_flag]
        sample_user_num = len(sample_data)

        print('Perturbating and aggregating users\' data...')
        for j in trange(sample_user_num):
            sample_freq[i] += local_randomizer(sample_data[j], itemset_len, p, q)
        
        sample_freq[i] = (sample_freq[i] - sample_user_num*q)/(p - q)
        freq_weight[i] = sample_user_num*t

        data = data[(budgets>t)|(sample_prob==0)]
        budgets = budgets[(budgets>t)|(sample_prob==0)]
        budgets[budgets>t] -= t

        if budgets.size==0:
            break

    return np.dot(freq_weight/freq_weight.sum(), sample_freq)

def personalized_frequency_estimation_sample_weight(data, itemset_len, budgets, t):
    p = 0.5
    q = 1/(math.exp(t) + 1)
    user_num = len(data)
    
    print('Sampling users\' bits...')
    sample_prob = np.zeros(user_num)
    sample_flag = np.zeros(user_num, dtype=bool)
    weight = np.zeros(user_num)
    for i in range(user_num):
        if budgets[i]>=t:
            sample_prob[i] = 1
            sample_flag[i] = True
        elif budgets[i]>math.log((math.exp(t)+1)/2):
            sample_prob[i] = (2*math.exp(budgets[i])-math.exp(t)-1)/(math.exp(budgets[i])-1)
            sample_flag[i] = np.random.choice(a=[True, False], p=[sample_prob[i], 1-sample_prob[i]])
        weight[i] = (p-q)/(q*(1-sample_prob[i]*q))

    print('Sample rate:', sample_flag.sum()/len(sample_flag))
    sample_data = data[sample_flag]
    sample_weight = weight[sample_flag]
    sample_user_num = len(sample_data)

    print('Perturbating and aggregating users\' data...')
    freq = np.zeros(itemset_len)
    for i in trange(sample_user_num):
        freq += sample_weight[i]*local_randomizer(sample_data[i], itemset_len, p, q)
    
    param_1 = np.dot(weight[(sample_prob>0)&(sample_prob<1)], sample_prob[(sample_prob>0)&(sample_prob<1)])
    param_2 = weight[sample_prob==1].sum()
    freq = sample_user_num*(freq - q*(param_1 + param_2))/((p - q)*(param_1 + param_2))

    return freq
    
def calc_freq(data, itemset_len, rate=False):
    user_num, data_len = data.shape
    freq = np.zeros(itemset_len)
    iterator = range(user_num)
    if rate:
        print('Calculating item frequency...')
        iterator = tqdm(iterator)
    for i in iterator:
        for j in range(data_len):
            if data[i, j]!=0:
                freq[data[i, j] - 1] += 1
    return freq

def calc_top_k(freq, k):
    return np.argsort(-freq)[0:k] + 1

def calc_CHR(freq, top_k):
    return freq[top_k - 1].sum()/freq.sum()
        
class SlideWindow():
    def __init__(self, num, omega, epsilon):
        self.num = num
        self.omega = omega
        self.epsilon = epsilon

        self.budgets_window = np.zeros((num, omega))
        self.cwp = 0 # current window pointer

    def update_window(self, budgets):
        self.budgets_window[:, self.cwp] = budgets
        self.cwp = (self.cwp+1)%self.omega
    
    def budgets_remain(self):
        return self.epsilon-self.budgets_window.sum(1)+self.budgets_window[:, self.cwp]

def disimilarity_randomizer(window, data, last_data):
    p = 1/(math.exp(window.dis_budget/window.omega)+1)
    dis_flag = np.zeros(window.num, dtype=bool)
    for i in range(window.num):
        dis = np.isin(data[i], last_data[i]).sum()/window.data_len
        if dis>window.t:
            dis_flag[i] = True
        if random.random()<2*p:
            if random.random()<0.5:
                dis_flag[i] = True
            else:
                dis_flag[i] = False
    return dis_flag

def local_budget_absorption(dis_budget, window, data, last_data, last_budget, interval):
    dis_flag = disimilarity_randomizer(window, data, last_data)
    to_nullify = round(last_budget/(window.epsilon/window.omega)) - 1
    if interval<=to_nullify:
        return 0
    to_absorb = interval - to_nullify
    budget = (window.epsilon/window.omega)*min(to_absorb, window.omega)
    if dis<=1/budget:
        return 0
    return budget

def budget_distribution(data, itemset_len, last_release, epsilon_rm, omega, epsilon):
    dis = 0
    if list(last_release):
        for i in range(itemset_len):
            dis += abs(last_release[i] - data[i])
        lambda1 = (2*omega)/(epsilon*itemset_num)
        dis = dis + np.random.laplace(0, lambda1, 1)
    lambda2 = 2/epsilon_rm
    if last_release is None or dis>lambda2:
        return epsilon_rm/2
    else:
        return 0

def budget_absorption(dis_budget, window, data, last_data, last_budget, interval):
    dis = np.isin(data, last_data).sum()/len(data)
    dis = dis + np.random.laplace(0, 1/dis_budget, 1)
    to_nullify = round(last_budget/(window.epsilon/window.omega)) - 1
    if interval<=to_nullify:
        return 0
    to_absorb = interval - to_nullify
    budget = (window.epsilon/window.omega)*min(to_absorb, window.omega)
    if dis<=1/budget:
        return 0
    return budget

class Agent():
    def __init__(self, 
            state_space_dim, 
            action_space_dim, 
            epsilon,  
            learning_rate=1e-6, 
            discount_rate=0.95, 
            capacity=1000000, 
            batch_size=10000, 
            target_update=10000):
        self.gamma = discount_rate
        self.epsilon = (math.exp(epsilon/4)-1)/(math.exp(epsilon/4)+action_space_dim-1)
        self.batch_size = batch_size
        self.target_update = target_update
        self.action_space_dim = action_space_dim

        self.policy_net = DQN(state_space_dim, action_space_dim)
        self.target_net = DQN(state_space_dim, action_space_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(capacity)
        self.steps = 0
        
    def act(self, state):
        self.steps += len(state)
        state =  torch.tensor(state, dtype=torch.float)
        action = torch.argmax(self.policy_net(state), dim=1).numpy()
        # exponential mechanism epsilon greedy
        for i in range(len(action)):
            if random.random() < self.epsilon:
                continue
            action[i] = np.random.randint(0, self.action_space_dim)
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
        
        if self.steps%self.target_update==0:
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
    def __init__(self, state_dim, action_dim, hidden_size=30):
        super(DQN, self).__init__()
        self.state_input = nn.Linear(state_dim, hidden_size)
        self.value_output = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.state_input(state))
        state_action_value = self.value_output(x)
        return state_action_value

def zipf_generation(a, x, num, data_len, rate=False):
    p = np.arange(1, len(x)+1)**(-a)
    p /= p.sum()
    data = np.zeros((num, data_len), dtype=int)
    iterator = range(num)
    if rate:
        print('Generating Zipf distribution users\' data...')
        iterator = tqdm(iterator)
    for i in iterator:
        data[i] = np.random.choice(a=x, size=data_len, replace=False, p=p) - 1
    return data

def transition(x, p):
    items = np.random.choice(a=np.arange(len(x)), size=int(p*len(x)), replace=False)
    x[items], x[-int(p*len(x)):] = x[-int(p*len(x)):], x[items]

def transition_matrix(itemset_len):
    pass

def markov_transition(item, itemset_len, p):
    if random.random()<p:
        item = random.randint(1, itemset_len)
    return item

def mean_absolute_error(freq_1, freq_2):
    return np.abs(freq_1/freq_1.sum() - freq_2/freq_2.sum()).mean()

def relative_error(freq, true_freq, k):
    true_top_k = calc_top_k(true_freq, k)
    return np.median(np.abs(freq[true_top_k-1]-true_freq[true_top_k-1])/true_freq[true_top_k-1])

def discounted_cumulative_gain(freq, true_freq):
    def reverse(sort):
        rank = np.zeros(len(sort))
        for i in range(len(sort)):
            rank[sort[i]] = i
        return rank
    rank, true_rank = reverse(freq.argsort()), reverse(true_freq.argsort())
    rel = np.log2(np.abs(len(freq)-np.abs(rank-true_rank)))
    return rel[0]+(rel[1:]/np.log2(np.arange(2, len(freq)+1))).sum()

if __name__=='__main__':
    itemset_len = 1000
    k = 100

    user_num = 1000
    data_len = 50

    omega = 10
    epsilon = 3

    total_step = 500
    learning_step = 100

    alpha_1 = 0.5
    alpha_2 = 0.5

    window = SlideWindow(num=user_num, omega=omega, epsilon=epsilon)
    allocator = Agent(state_space_dim=data_len+1, action_space_dim=omega+1)

    data = np.load('./data/zipf_500000.npy').reshape((total_step, user_num, data_len))
    
    state = torch.tensor(data[0])
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        action = allocator.act(state)
        print(Counter(action))
        budgets = action*epsilon/omega
        budgets_rm = epsilon - window.budgets_window.sum(1) - budgets
        budgets[budgets_rm<0] = budgets_rm[budgets_rm<0]
        # print(budgets)

        # delete users where epsilon = 0
        usable_data = data[step][budgets!=0]
        usable_budgets = budgets[budgets!=0]
        print(len(usable_data))

        freq = personalized_frequency_estimation(usable_data, itemset_len, usable_budgets)
        top_k = calc_top_k(freq, k)

        true_freq = calc_freq(data[step], itemset_len)
        true_top_k = calc_top_k(true_freq, k)

        correct_num = len(set(top_k)&set(true_top_k))
        print('Number of correct items:', correct_num)
        oracle_CHR = calc_CHR(true_freq, true_top_k)
        print('Oracle caching hit ratio:', oracle_CHR)
        LDP_CHR = calc_CHR(true_freq, top_k)
        print('LDP caching hit ratio:', LDP_CHR)

        next_state = torch.tensor(data[step+1])

        if step<learning_step:
            hit_num = np.zeros(user_num)
            for i in range(user_num):
                hit_num[i] = len([k for k in data[step][i] if k in top_k])
            reward = alpha_1*budgets_rm + alpha_2*hit_num
            allocator.memory.push(state, action, next_state, reward)
            allocator.learn()

        window.update_window(budgets)
        state = next_state
        if step==10:
            break
