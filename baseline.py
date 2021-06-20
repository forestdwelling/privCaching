from global_setting import *

def run_sample():
    randomizer.set_budget(epsilon)
    for step in range(total_step):
        print(f'STEP {step+1} :')
        generator.transit()
        if step%omega==0:
            # randomize and aggregate
            sample_data = generator.sample()
            private_counts = randomizer.randomize_group(sample_data)
            estimate_counts = randomizer.aggregate(private_counts)
        # record metrics
        true_counts= calc_counts(generator.data, itemset_len)
        record[step] = utility_metrics(estimate_counts, true_counts, k)
    print('Mean:', record.mean(0))

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
    print('Mean:', record.mean(0))

def run_DSFT():
    global threshold
    global cop
    generator.generate()
    randomizer.set_budget(epsilon/cop)
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
    print('Mean:', record.mean(0))

# # first release
# window.update(epsilon/omega)
# # collect data
# generator.generate()
# true_counts = calc_counts(data, itemset_len)
# sample_data = np.zeros(user_num)
# for i in range(user_num):
#     sample_data[i] = np.random.choice(generator.data[i])
# private_counts = randomizer.randomize_group(sample_data)
# counts = randomizer.aggregate(private_counts)
# top_k = calc_top_k(counts, k)

# def run_random_continuous():
#     global counts
#     for step in range(total_step):
#         print(f'\n=========================================== STEP {step+1} ===========================================\n')
#         # generate data
#         data = generator.generate()
#         generator.transit()
#         # allocate budget
#         budget = rm.uniform(0, window.remain())
#         print('remain:', window.remain(), 'budget:', budget)
#         window.update(budget)
#         if budget!=0:
#             randomizer.set_budget(budget)
#             # randomize and aggregate
#             private_counts = randomizer.randomize_group(data)
#             counts = randomizer.aggregate(private_counts)
#         # record metrics
#         true_counts= calc_counts(data, itemset_len)
#         record[step] = utility_metrics(counts, true_counts, k)
#     print('Mean:', record.mean(0))

# def run_random_discrete():
#     global counts
#     for step in range(total_step):
#         print(f'\n=========================================== STEP {step+1} ===========================================\n')
#         # generate data
#         data = generator.generate()
#         generator.transit()
#         # allocate budget
#         budget = rm.randint(0, window.remain()*omega/epsilon)
#         print('remain:', window.remain(), 'budget:', budget)
#         window.update(budget)
#         if budget!=0:
#             randomizer.set_budget(budget)
#             # randomize and aggregate
#             private_counts = randomizer.randomize_group(data)
#             counts = randomizer.aggregate(private_counts)
#         # record metrics
#         true_counts= calc_counts(data, itemset_len)
#         record[step] = utility_metrics(counts, true_counts, k)
#     print('Mean:', record.mean(0))

if __name__=='__main__':
    run_sample()