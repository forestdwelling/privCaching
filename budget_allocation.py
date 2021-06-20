from global_setting import *

def run():
    # first release
    window.update(epsilon/omega)
    # collect data
    data = generator.generate()
    true_counts = calc_counts(data, itemset_len)
    generator.transit()
    private_counts = data_randomizer.randomize_group(data)
    counts = data_randomizer.aggregate(private_counts)
    top_k = calc_top_k(counts, k)
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