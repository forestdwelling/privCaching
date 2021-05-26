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

randomizer = RAPPOR(itemset_len, epsilon/omega)
generator = ZipfGenerator(itemset_len, user_num, data_len)
window = SlidingWindow(omega, epsilon)

randomizer.enable_render()
generator.enable_render()

record = np.zeros((total_step, 6))

# first release
window.update(epsilon/omega)
# collect data
data = generator.generate()
generator.transit()
true_counts = calc_counts(data, itemset_len)
private_counts = randomizer.randomize_group(data)
counts = randomizer.aggregate(private_counts)
top_k = calc_top_k(counts, k)

def run_random_continuous():
    global counts
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        data = generator.generate()
        generator.transit()
        # allocate budget
        budget = rm.uniform(0, window.remain())
        print('remain:', window.remain(), 'budget:', budget)
        window.update(budget)
        if budget!=0:
            randomizer.set_budget(budget)
            # randomize and aggregate
            private_counts = randomizer.randomize_group(data)
            counts = randomizer.aggregate(private_counts)
        # record metrics
        true_counts= calc_counts(data, itemset_len)
        record[step] = utility_metrics(counts, true_counts, k)
    print('Mean:', record.mean(0))

def run_random_discrete():
    global counts
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        data = generator.generate()
        generator.transit()
        # allocate budget
        budget = rm.randint(0, window.remain()*omega/epsilon)
        print('remain:', window.remain(), 'budget:', budget)
        window.update(budget)
        if budget!=0:
            randomizer.set_budget(budget)
            # randomize and aggregate
            private_counts = randomizer.randomize_group(data)
            counts = randomizer.aggregate(private_counts)
        # record metrics
        true_counts= calc_counts(data, itemset_len)
        record[step] = utility_metrics(counts, true_counts, k)
    print('Mean:', record.mean(0))

def run_sample():
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        data = generator.generate()
        generator.transit()
        # allocate budget
        if step%omega==0:
            randomizer.set_budget(epsilon)
            # randomize and aggregate
            private_counts = randomizer.randomize_group(data)
            counts = randomizer.aggregate(private_counts)
        # record metrics
        true_counts= calc_counts(data, itemset_len)
        record[step] = utility_metrics(counts, true_counts, k)
    print('Mean:', record.mean(0))

def run_uniform():
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        data = generator.generate()
        generator.transit()
        # allocate budget
        randomizer.set_budget(epsilon/omega)
        # randomize and aggregate
        private_counts = randomizer.randomize_group(data)
        counts = randomizer.aggregate(private_counts)
        # record metrics
        true_counts= calc_counts(data, itemset_len)
        record[step] = utility_metrics(counts, true_counts, k)
    print('Mean:', record.mean(0))

if __name__=='__main__':
    run_sample()