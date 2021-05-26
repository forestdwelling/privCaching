import math as m
import numpy as np
import scipy.stats as stats

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

class ZipfGenerator:
    def __init__(self, itemset_len, user_num, data_len):
        # x = [0, 1, ..., L]
        self.x = np.arange(1, itemset_len+1)
        self.p = np.arange(1, itemset_len+1)**(-1.5)
        self.p /= self.p.sum()
        mu, sigma = 0.2, 0.2
        self.transit_ratio = stats.truncnorm(-mu/sigma, (1 - mu)/sigma, loc=mu, scale=sigma)
        # self.transit_ratio = stats.binom(1, 0.1)
        self.user_num, self.data_len, self.itemset_len = user_num, data_len, itemset_len
        self.data = np.zeros((user_num, data_len), dtype=int)
        self.render = False

    def generate(self):
        iterator = range(self.user_num)
        if self.render:
            print('Generating Zipf distribution users\' data...')
            iterator = tqdm(iterator)
        for i in iterator:
            self.data[i] = np.random.choice(a=self.x, size=self.data_len, replace=False, p=self.p)
        return self.data

    def transit(self):
        self.r = self.transit_ratio.rvs()
        items = np.random.choice(a=self.x, size=int(self.r*self.itemset_len), replace=False) - 1
        self.x[items] = np.random.permutation(self.x[items])
    
    def enable_render(self, render='True'):
        self.render = render

if __name__=='__main__':
    generator = ZipfGenerator(1.5, 1000, 0.2, 10000, 20)
    generator.enableRender()
    data = generator.generate()
    plt.hist(data.flatten(), bins=np.arange(1001))
    plt.show()