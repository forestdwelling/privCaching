from global_setting import *

def run():
    sim_budget, data_budget = 0.1*epsilon, 0.9*epsilon
    data_randomizer = RAPPOR(itemset_len, data_budget/omega)
    data_randomizer.enable_render(False)
    sim_randomizer = Harmony(0, data_len, sim_budget/omega)
    window = SlidingWindow(omega, data_budget)
    hits = np.zeros(user_num)
    # first release
    window.update(data_budget)
    data_randomizer.set_budget(data_budget)
    sample_data = generator.sample()
    private_counts = data_randomizer.randomize_group(sample_data)
    estimate_counts = data_randomizer.aggregate(private_counts)
    top_k = calc_top_k(estimate_counts, k)
    true_counts= calc_counts(generator.data, itemset_len)
    record[0] = utility_metrics(estimate_counts, true_counts, k)
    for step in range(1, total_step):
        print(f'\nSTEP {step+1}')
        # generate data
        generator.transit()
        # calculate hit_num
        for i in range(user_num):
            hits[i] = len(np.intersect1d(generator.data[i], top_k))
        # collect similarity
        private_hits = sim_randomizer.randomize_group(hits)
        similarity = sim_randomizer.aggregate(private_hits)/data_len

        remain = window.remain()
        print('similarity:', similarity, hits.mean()/data_len)
        print('remain:', remain)
        if similarity < threshold and remain > 0:
            print('collect')
            window.update(remain)
            data_randomizer.set_budget(remain)
            sample_data = generator.sample()
            private_counts = data_randomizer.randomize_group(sample_data)
            estimate_counts = data_randomizer.aggregate(private_counts)
            top_k = calc_top_k(estimate_counts, k)
        else:
            print('skip')
            window.update(0)
        true_counts = calc_counts(generator.data, itemset_len)
        record[step] = utility_metrics(estimate_counts, true_counts, k)
    result = record.mean(0)
    print('Mean:', result[0], result[1], result[2], result[3], result[4], result[5])

if __name__=='__main__':
    run()