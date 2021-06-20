import math as m
import random as rm
import numpy as np
import scipy.stats as stats

from data_generator import *
from local_randomizer import *
from sliding_window import *
from metric import *

k = 100
itemset_len = 1000
# synthetic data size
user_num = 10000
data_len = 10
# window size
omega = 5
# total budget and divsion
epsilon = 1
sim_budget, data_budget = 0.5*epsilon, 0.5*epsilon
# budget allocation threshold
threshold = 0.3
# upper bound of release number
cop = 5

# total runing step number
total_step = 100

randomizer = RAPPOR(itemset_len, epsilon/omega)
sim_randomizer = Harmony(0, data_len, sim_budget/omega)
generator = ZipfGenerator(itemset_len, user_num, data_len)
window = SlidingWindow(omega, epsilon)

randomizer.enable_render()
generator.enable_render()

record = np.zeros((total_step, 6))