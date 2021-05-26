import math as m
import random as rm
import numpy as np

class SlidingWindow():
    def __init__(self, omega, epsilon):
        self.omega = omega
        self.epsilon = epsilon

        self.budgets_window = np.zeros(omega)
        self.cwp = 0 # current window pointer

    def update(self, budget):
        self.budgets_window[self.cwp] = budget
        self.cwp = (self.cwp+1)%self.omega
    
    def remain(self):
        return self.epsilon-self.budgets_window.sum()+self.budgets_window[self.cwp]

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