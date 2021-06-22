from global_setting import *

def run_uniform():
    randomizer.set_budget(epsilon/omega)
    for step in range(total_step):
        print(f'STEP {step+1} :')
        generator.transit()
        # randomize and aggregate
        sample_data = generator.sample()
        private_counts = randomizer.randomize_group(sample_data)
        estimate_counts = randomizer.aggregate(private_counts)
        # record metrics
        true_counts= calc_counts(generator.data, itemset_len)
        record[step] = utility_metrics(estimate_counts, true_counts, k)
    result = record.mean(0)
    print('Mean:', result[0], result[1], result[2], result[3], result[4], result[5])
    
if __name__=='__main__':
    randomizer = RAPPOR(itemset_len, epsilon/omega)
    randomizer.enable_render(False)
    run_uniform()