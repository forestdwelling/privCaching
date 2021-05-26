import math
import numpy as np

def calc_counts(data, itemset_len, rate=False):
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

def utility_metrics(freq, true_freq, k):
    top_k, true_top_k = calc_top_k(freq, k), calc_top_k(true_freq, k)
    hit_num = np.isin(top_k, true_top_k).sum()
    MAE = mean_absolute_error(freq, true_freq)
    RE = relative_error(freq, true_freq, k)
    NDCG = normalized_discounted_cumulative_gain(freq, true_freq)
    CHR = calc_CHR(true_freq, top_k)
    oracle = calc_CHR(true_freq, true_top_k)
    print('hit_num', 'MAE', 'RE', 'NDCG', 'CHR', 'oracle')
    print(hit_num, MAE, RE, NDCG, CHR, oracle)
    return np.array([hit_num, MAE, RE, NDCG, CHR, oracle])

def mean_absolute_error(freq_1, freq_2):
    return np.abs(freq_1/freq_1.sum() - freq_2/freq_2.sum()).mean()

def relative_error(freq, true_freq, k):
    true_top_k = calc_top_k(true_freq, k)
    return np.median(np.abs(freq[true_top_k-1]-true_freq[true_top_k-1])/true_freq[true_top_k-1])

def normalized_discounted_cumulative_gain(freq, true_freq):
    def reverse(sort):
        rank = np.zeros(len(sort))
        for i in range(len(sort)):
            rank[sort[i]] = i
        return rank
    rank, true_rank = reverse(freq.argsort()), reverse(true_freq.argsort())
    rel = np.log2(np.abs(len(freq)-np.abs(rank-true_rank)))
    return (rel/np.log2(np.arange(1, len(freq) + 1) + 1)).sum()/(np.log2(len(freq)/np.log2(np.arange(1, len(freq) + 1) + 1))).sum()