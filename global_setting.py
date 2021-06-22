import math as m
import random as rm
import numpy as np
import scipy.stats as stats

from data_generator import *
from local_randomizer import *
from sliding_window import *
from metric import *

BASELINE = False

k = 100
itemset_len = 1000
# synthetic data size
user_num = 500000
data_len = 10
# window size
omega = 5
# total budget and division
epsilon = 1
# budget allocation threshold
threshold = 0.35
# upper bound of release number
cop = 2

# total runing step number
total_step = 100

generator = ZipfGenerator(itemset_len, user_num, data_len)
generator.enable_render(False) 

record = np.zeros((total_step, 6))