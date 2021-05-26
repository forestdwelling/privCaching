import math as m
import random as rm
import numpy as np

from tqdm import tqdm, trange

class RAPPOR:
    def __init__(self, itemset_len, epsilon):
        # the probability of 1->1
        self.p = 1/2
        # the probability of 0->1
        self.q = 1/(m.exp(epsilon)+1)
        # the size of buckets
        self.itemset_len = itemset_len
        self.render = False
    
    def set_budget(self, epsilon):
        self.q = 1/(m.exp(epsilon)+1)

    def randomize(self, data):
        # onehot encoding
        item = rm.choice(data)
        if item!=0:
            private_data[item-1] = 1
            
        # randomized response
        private_data = np.where(private_data==1, 
            np.random.binomial(1, self.p, self.itemset_len),
            np.random.binomial(1, self.q, self.itemset_len))
        return private_data

    def randomize_group(self, data):
        user_num = data.shape[0]
        private_data = np.zeros((user_num, self.itemset_len), dtype=int)

        iterator = range(user_num)
        if self.render:
            print('Perturbating users\' data...')
            iterator = tqdm(iterator)

        # onehot encoding
        for i in iterator:
            item = rm.choice(data[i])
            if item!=0:
                private_data[i][item-1] = 1

        # randomized response
        private_data = np.where(private_data==1, 
            np.random.binomial(1, self.p, (user_num, self.itemset_len)),
            np.random.binomial(1, self.q, (user_num, self.itemset_len)))
        return private_data

    def aggregate(self, private_data):
        return (private_data.sum(0) - private_data.shape[0]*self.q)/(self.p - self.q)

    def enable_render(self, render='True'):
        self.render = render

class SuccinctHistogram:
    def __init__(self):
        pass
    
    def enable_render(self, render='True'):
        self.render = render

class Harmony:
    def __init__(self, lower, upper, epsilon):
        # the probability of 1->1 or 0->0
        self.p = m.exp(epsilon)/(1 + m.exp(epsilon))
        self.lower = lower
        self.upper = upper
    
    def set_budget(self, epsilon):
        self.p = m.exp(epsilon)/(1 + m.exp(epsilon))
        
    def randomize(self, data):
        # discretize
        discrete_data = np.random.choice(a=[self.upper, self.lower], 
            p=[(data - self.lower)/(self.upper - self.lower), (self.upper - data)/(self.upper - self.lower)])

        # randomize
        private_data = np.where(discrete_data==self.upper, 
            np.random.choice(a=[self.upper, self.lower], p=[self.p, 1 - self.p]), 
            np.random.choice(a=[self.upper, self.lower], p=[1 - self.p, self.p]))
        return private_data
    
    def randomize_group(self, data):
        # discretize
        discrete_data = np.zeros(len(data))
        for i in range(len(data)):
            discrete_data[i] = np.random.choice(a=[self.upper, self.lower], 
                p=[(data[i] - self.lower)/(self.upper - self.lower), (self.upper - data[i])/(self.upper - self.lower)])

        # randomize
        private_data = np.where(discrete_data==self.upper, 
            np.random.choice(a=[self.upper, self.lower], p=[self.p, 1 - self.p], size=len(data)), 
            np.random.choice(a=[self.upper, self.lower], p=[1 - self.p, self.p], size=len(data)))
        return private_data

    def aggregate(self, private_data):
        upper_count = (private_data==self.upper).sum()
        lower_count = (private_data==self.lower).sum()
        upper_count= np.clip((len(private_data)*(self.p - 1) + upper_count)/(2*self.p - 1), 0, len(private_data))
        lower_count= np.clip((len(private_data)*(self.p - 1) + lower_count)/(2*self.p - 1), 0, len(private_data))
        return (self.upper*upper_count + self.lower*lower_count)/len(private_data)

if __name__=='__main__':
    from data_generator import *
    from metric import *

    user_num = 500000
    data_len = 10
    itemset_len = 1000
    k = 40
    epsilon = 1

    randomizer = RAPPOR(itemset_len, epsilon)
    generator = ZipfGenerator(itemset_len, user_num, data_len)

    randomizer.enable_render()
    generator.enable_render()

    data = generator.generate()
    true_counts = calc_counts(data, itemset_len)
    private_counts = randomizer.randomize_group(data)
    counts = randomizer.aggregate(private_counts)
    utility_metrics(counts, true_counts, k)