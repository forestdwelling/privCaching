import math as m
import random as rm
import numpy as np
import scipy.stats as stats

from data_generator import *
from local_randomizer import *
from sliding_window import *
from metric import *

itemset_len = 1000
k = 100

user_num = 50000
data_len = 10

omega = 5
epsilon = 5

total_step = 100

sim_budget, data_budget = 0.1*epsilon, 0.9*epsilon
sim_randomizer = Harmony(0, data_len, sim_budget/omega)
data_randomizer = RAPPOR(itemset_len, data_budget/omega)
generator = ZipfGenerator(itemset_len, user_num, data_len)
window = SlidingWindow(omega, data_budget)

data_randomizer.enable_render()
generator.enable_render()

hits = np.zeros(user_num)
record = np.zeros((total_step, 6))

# first release
window.update(epsilon/omega)
# collect data
data = generator.generate()
true_counts = calc_counts(data, itemset_len)
generator.transit()
private_counts = data_randomizer.randomize_group(data)
counts = data_randomizer.aggregate(private_counts)
top_k = calc_top_k(counts, k)

def run():
    global counts, top_k
    for step in range(total_step):
        print(f'\nSTEP {step+1}')
        # generate data
        data = generator.generate()
        true_counts = calc_counts(data, itemset_len)
        generator.transit()
        # calculate hit_num
        for i in range(user_num):
            hits[i] = len(np.intersect1d(data[i], top_k))
        # collect similarity
        private_hits = sim_randomizer.randomize_group(hits)
        similarity = sim_randomizer.aggregate(private_hits)/data_len

        remain = window.remain()
        print('similarity:', similarity, hits.mean()/data_len)
        print('remain:', remain)
        if similarity<0.35 and remain>0:
            print('collect')
            window.update(remain)
            data_randomizer.set_budget(remain)
            private_counts = data_randomizer.randomize_group(data)
            counts = data_randomizer.aggregate(private_counts)
            top_k = calc_top_k(counts, k)
        else:
            print('skip')
            window.update(0)

        record[step] = utility_metrics(counts, true_counts, k)
    print('Mean:', record.mean(0))

if __name__=='__main__':
    run()