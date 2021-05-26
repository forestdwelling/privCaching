from caching import *

import time

def utility_metrics(freq, true_freq, k):
    top_k, true_top_k = calc_top_k(freq, k), calc_top_k(true_freq, k)
    print('Number of correct items:', np.isin(top_k, true_top_k).sum())
    print('MAE:', mean_absolute_error(freq, true_freq))
    print('RE:', relative_error(freq, true_freq, k))
    print('DCG:', discounted_cumulative_gain(freq, true_freq))
    print('CHR:', calc_CHR(true_freq, top_k))
    print('Oracle', calc_CHR(true_freq, true_top_k))
    return np.isin(top_k, true_top_k).sum()

def lr_contrast():
    pass

def pce_contrast():
    itemset_len = 1000
    k = 100

    omega = 10
    epsilon = 5

    data = np.load('./data/zipf_10000_1000_50.npy')
    budgets = np.random.choice(a=(np.arange(omega)+1)*epsilon/omega, size=len(data), p=[1/omega]*omega)
    # budgets = np.random.rand(len(data))*epsilon
    # budgets = np.full(len(data), epsilon)
    # budgets = np.random.choice(a=[0.1, 1], size=500000, p=[0.1, 0.9])
    # budgets = np.zeros(user_num)
    # for i in range(1, omega+1):
    #     print('Privacy budget:', epsilon*i/omega, 'Range:', int(user_num*(i-1)/omega), '-', int(user_num*i/omega))
    #     budgets[int(user_num*(i-1)/omega):int(user_num*i/omega)] = epsilon*i/omega

    true_freq = calc_freq(data, itemset_len)
    true_top_k = calc_top_k(true_freq, k)
    oracle_CHR = calc_CHR(true_freq, true_top_k)
    print('Oracle caching hit ratio:', oracle_CHR)

    episodes = 10
    results = np.zeros((episodes, 7, 3))
    for i in range(episodes):
        print(f'\n======================================== EPISODE {i+1} ========================================\n')
        release_data = local_randomizer(data, itemset_len, budgets)
        freq = personalized_frequency_estimation(release_data.copy(), itemset_len, budgets)
        top_k = calc_top_k(freq, k)
        results[i, 0, 0] = len(set(top_k) & set(true_top_k))
        results[i, 0, 1] = mean_absolute_error(true_freq, freq)
        results[i, 0, 2] = calc_CHR(true_freq, top_k)

        weight_freq = personalized_frequency_estimation_weight(release_data.copy(), itemset_len, budgets)
        weight_top_k = calc_top_k(weight_freq, k)
        results[i, 1, 0] = len(set(weight_top_k) & set(true_top_k))
        results[i, 1, 1] = mean_absolute_error(true_freq, weight_freq)
        results[i, 1, 2] = calc_CHR(true_freq, weight_top_k)

        # sample_freq_mean = personalized_frequency_estimation_sample(data.copy(), itemset_len, budgets.copy(), 1)
        # sample_top_k_mean = calc_top_k(sample_freq_mean, k)
        # results[i, 2, 0] = len(set(sample_top_k_mean) & set(true_top_k))
        # results[i, 2, 1] = mean_absolute_error(true_freq, sample_freq_mean)
        # results[i, 2, 2] = calc_CHR(true_freq, sample_top_k_mean)

        # sample_freq_max = personalized_frequency_estimation_sample(data.copy(), itemset_len, budgets.copy(), 1, 'max')
        # sample_top_k_max = calc_top_k(sample_freq_max, k)
        # results[i, 3, 0] = len(set(sample_top_k_max) & set(true_top_k))
        # results[i, 3, 1] = mean_absolute_error(true_freq, sample_freq_max)
        # results[i, 3, 2] = calc_CHR(true_freq, sample_top_k_max)

        # sample_freq_twice = personalized_frequency_estimation_sample(data.copy(), itemset_len, budgets.copy(), 2)
        # sample_top_k_twice = calc_top_k(sample_freq_twice, k)
        # results[i, 4, 0] = len(set(sample_top_k_twice) & set(true_top_k))
        # results[i, 4, 1] = mean_absolute_error(true_freq, sample_freq_twice)
        # results[i, 4, 2] = calc_CHR(true_freq, sample_top_k_twice)

        # sample_freq_twice_max = personalized_frequency_estimation_sample(data.copy(), itemset_len, budgets.copy(), 2, 'max')
        # sample_top_k_twice_max = calc_top_k(sample_freq_twice_max, k)
        # results[i, 5, 0] = len(set(sample_top_k_twice_max) & set(true_top_k))
        # results[i, 5, 1] = mean_absolute_error(true_freq, sample_freq_twice_max)
        # results[i, 5, 2] = calc_CHR(true_freq, sample_top_k_twice_max)

        # sample_weight_freq = personalized_frequency_estimation_sample_weight(data.copy(), itemset_len, budgets.copy(), epsilon)
        # sample_weight_top_k = calc_top_k(sample_weight_freq, k)
        # results[i, 6, 0] = len(set(sample_weight_top_k) & set(true_top_k))
        # results[i, 6, 1] = mean_absolute_error(true_freq, sample_weight_freq)
        # results[i, 6, 2] = calc_CHR(true_freq, sample_weight_top_k)

        print('\tNumber of correct items', '\tMAE', '\tCHR')
        print('(mle):             ', '\t'+str(results[i, 0, 0]), '\t'+str(results[i, 0, 1]), '\t'+str(results[i, 0, 2]))
        print('(weight):          ', '\t'+str(results[i, 1, 0]), '\t'+str(results[i, 1, 1]), '\t'+str(results[i, 1, 2]))
        print('(sample mean):     ', '\t'+str(results[i, 2, 0]), '\t'+str(results[i, 2, 1]), '\t'+str(results[i, 2, 2]))
        print('(sample max):      ', '\t'+str(results[i, 3, 0]), '\t'+str(results[i, 3, 1]), '\t'+str(results[i, 3, 2]))
        print('(sample twice):    ', '\t'+str(results[i, 4, 0]), '\t'+str(results[i, 4, 1]), '\t'+str(results[i, 4, 2]))
        print('(sample twice max):', '\t'+str(results[i, 5, 0]), '\t'+str(results[i, 5, 1]), '\t'+str(results[i, 5, 2]))
        print('(sample weight):   ', '\t'+str(results[i, 6, 0]), '\t'+str(results[i, 6, 1]), '\t'+str(results[i, 6, 2]))
        
    print('\tMean number of correct items', '\tMean MAE', '\tMean CHR')
    print('(mle):             ', '\t'+str(results.mean(0)[0, 0]), '\t'+str(results.mean(0)[0, 1]), '\t'+str(results.mean(0)[0, 2]))
    print('(weight):          ', '\t'+str(results.mean(0)[1, 0]), '\t'+str(results.mean(0)[1, 1]), '\t'+str(results.mean(0)[1, 2]))
    print('(sample mean):      ', '\t'+str(results.mean(0)[2, 0]), '\t'+str(results.mean(0)[2, 1]), '\t'+str(results.mean(0)[2, 2]))
    print('(sample max):      ', '\t'+str(results.mean(0)[3, 0]), '\t'+str(results.mean(0)[3, 1]), '\t'+str(results.mean(0)[3, 2]))
    print('(sample twice):    ', '\t'+str(results.mean(0)[4, 0]), '\t'+str(results.mean(0)[4, 1]), '\t'+str(results.mean(0)[4, 2]))
    print('(sample twice max):', '\t'+str(results.mean(0)[5, 0]), '\t'+str(results.mean(0)[5, 1]), '\t'+str(results.mean(0)[5, 2]))
    print('(sample weight):   ', '\t'+str(results.mean(0)[6, 0]), '\t'+str(results.mean(0)[6, 1]), '\t'+str(results.mean(0)[6, 2]))

def ba_test():
    itemset_len = 1000
    k = 100
    total_step = 10
    user_num = 1
    data_len = 50
    omega = 3
    epsilon = 5
    window = SlideWindow(user_num, omega, epsilon, data_len, itemset_len, 0.5)
    release_data = np.zeros((user_num, itemset_len), dtype=int)
    release_budgets = np.zeros(user_num)
    for step in range(100):
        data = zipf_generation(1.3, user_num, data_len)
        budgets, last_budgets, last_release = budget_absorption(window, data)
        release_data[budgets!=0] = sampling_randomizer(data[budgets!=0], itemset_len, budgets)
        release_data[budgets==0] = last_release[budgets==0]
        release_budgets[budgets!=0] = budgets[budgets!=0]
        release_budgets[budgets==0] = last_budgets[budgets==0]
        window.update_window(budgets, data, release_data)
        print(budgets[0])

def rl_test():
    pass

def exp_rl():
    itemset_len = 1000
    k = 100
    total_step = 10
    user_num = 100000
    data_len = 20
    omega = 5
    epsilon = 5
    learning_step = 100
    alpha_1 = -1
    alpha_2 = 1

    window = SlideWindow(user_num, omega, epsilon/2)
    allocator = Agent(2, omega+1, epsilon/2)

    dis = np.zeros(user_num)
    release_data = np.zeros((user_num, itemset_len), dtype=int)
    release_budgets = np.zeros(user_num)

    # first allocation
    x = np.arange(1, itemset_len+1)
    last_data = zipf_generation(1.3, x, user_num, data_len, rate=True)
    last_budgets = np.full(user_num, window.epsilon/omega)
    last_release = sampling_randomizer(last_data, itemset_len, last_budgets, rate=True)
    interval = np.ones(user_num, dtype=int)
    window.update_window(last_budgets)

    # generate first state
    transition(x, 0.2)
    data = zipf_generation(1.3, x, user_num, data_len)
    for i in range(window.num):
        dis[i] = np.isin(data[i], last_data[i]).sum()/data_len
    budgets_rm = window.budgets_remain()
    state = np.squeeze(np.dstack((dis, budgets_rm)))

    record = np.zeros(total_step)
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # allocate budgets
        action = allocator.act(state)
        budgets = action*window.epsilon/omega
        print(Counter(budgets))
        budgets_rm = window.budgets_remain()
        budgets[budgets_rm<budgets] = budgets_rm[budgets_rm<budgets]
        print(Counter(budgets))
        # perturb data
        release_data[budgets!=0] = sampling_randomizer(data[budgets!=0], itemset_len, budgets[budgets!=0], rate=True)
        release_data[budgets==0] = last_release[budgets==0]
        release_budgets[budgets!=0] = budgets[budgets!=0]
        release_budgets[budgets==0] = last_budgets[budgets==0]
        print(Counter(release_budgets))
        # update window
        interval[budgets!=0] = 0
        interval += 1
        last_data[budgets!=0] = data[budgets!=0]
        last_release[budgets!=0] = release_data[budgets!=0]
        last_budgets[budgets!=0] = budgets[budgets!=0]
        window.update_window(budgets)
        # aggregate and estimate
        freq = personalized_frequency_estimation(release_data, itemset_len, release_budgets)
        top_k = calc_top_k(freq, k)
        true_freq = calc_freq(data, itemset_len)
        record[step] = utility_metrics(freq, true_freq, k)
        # generate next state
        transition(x, 0.2)
        data = zipf_generation(1.3, x, user_num, data_len, rate=True)
        for i in range(window.num):
            dis[i] = np.isin(data[i], last_data[i]).sum()/data_len
        next_state = np.squeeze(np.dstack((dis, budgets_rm)))
        # learn
        if step<learning_step:
            hit_num = np.zeros(user_num)
            for i in range(user_num):
                reward = alpha_1*abs(action[i]/omega-budgets_rm[i]/window.epsilon) + alpha_2*np.isin(data[i], top_k).sum()/data_len
                allocator.memory.push(state[i], action[i], next_state[i], reward)
            allocator.learn()
        state = next_state
    print('Mean:', record.mean())

def exp_ba():
    itemset_len = 1000
    k = 100
    total_step = 10
    user_num = 100000
    data_len = 20
    omega = 5
    epsilon = 5
    a = 1.5
    
    window = SlideWindow(user_num, omega, epsilon/2)

    budgets = np.zeros(user_num)
    release_data = np.zeros((user_num, itemset_len), dtype=int)
    release_budgets = np.zeros(user_num)

    # first allocation
    x = np.arange(1, itemset_len+1)
    last_data = zipf_generation(a, x, user_num, data_len, rate=True)
    last_budgets = np.full(user_num, epsilon/(omega*2))
    last_release = sampling_randomizer(last_data, itemset_len, last_budgets, rate=True)
    interval = np.ones(user_num, dtype=int)
    window.update_window(last_budgets)

    record = np.zeros(total_step)
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        transition(x, 0.2)
        data = zipf_generation(a, x, user_num, data_len, rate=True)
        # allocate budgets
        for i in trange(user_num):
            budgets[i] = budget_absorption(epsilon/(2*omega), window, data[i], last_data[i], last_budgets[i], interval[i])
        print(Counter(budgets))
        # perturb data
        release_data[budgets!=0] = sampling_randomizer(data[budgets!=0], itemset_len, budgets[budgets!=0], rate=True)
        release_data[budgets==0] = last_release[budgets==0]
        release_budgets[budgets!=0] = budgets[budgets!=0]
        release_budgets[budgets==0] = last_budgets[budgets==0]
        # update window
        interval[budgets!=0] = 0
        interval += 1
        last_data[budgets!=0] = data[budgets!=0]
        last_release[budgets!=0] = release_data[budgets!=0]
        last_budgets[budgets!=0] = budgets[budgets!=0]
        window.update_window(budgets)
        # aggregate and estimate
        freq = personalized_frequency_estimation(release_data, itemset_len, release_budgets)
        true_freq = calc_freq(data, itemset_len)
        record[step] = utility_metrics(freq, true_freq, k)
    print('Mean:', record.mean())

def exp_random():
    itemset_len = 1000
    k = 100
    total_step = 10
    user_num = 100000
    data_len = 20
    omega = 5
    epsilon = 5
    a = 1.5

    window = SlideWindow(user_num, omega, epsilon)
    release_data = np.zeros((user_num, itemset_len), dtype=int)
    release_budgets = np.zeros(user_num)

    # first step
    x = np.arange(1, itemset_len+1)
    last_data = zipf_generation(a, x, user_num, data_len, rate=True)
    last_budgets = np.full(user_num, epsilon/omega)
    last_release = sampling_randomizer(last_data, itemset_len, last_budgets, rate=True)
    window.update_window(last_budgets)

    record = np.zeros(total_step)
    for step in range(total_step):
        transition(x, 0.2)
        data = zipf_generation(1.3, x, user_num, data_len)
        print(window.budgets_remain())
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        action = np.random.randint(omega+1, size=user_num)
        budgets = action*epsilon/omega
        print(Counter(budgets))
        budgets_rm = window.budgets_remain()
        budgets[budgets_rm<budgets] = budgets_rm[budgets_rm<budgets]
        print(Counter(budgets))
        release_data[budgets!=0] = sampling_randomizer(data[budgets!=0], itemset_len, budgets[budgets!=0], rate=True)
        release_data[budgets==0] = last_release[budgets==0]
        release_budgets[budgets!=0] = budgets[budgets!=0]
        release_budgets[budgets==0] = last_budgets[budgets==0]
        print(Counter(release_budgets))
        last_data[budgets!=0] = data[budgets!=0]
        last_release[budgets!=0] = release_data[budgets!=0]
        last_budgets[budgets!=0] = budgets[budgets!=0]
        window.update_window(budgets)

        freq = personalized_frequency_estimation(release_data, itemset_len, release_budgets)
        true_freq = calc_freq(data, itemset_len)
        record[step] = utility_metrics(freq, true_freq, k)
    print(record.mean())

def exp_uniform():
    itemset_len = 10000
    k = 100
    total_step = 10
    user_num = 100000
    data_len = 20
    omega = 5
    epsilon = 5
    a = 1.5
    x = np.arange(1, itemset_len+1)
    release_data = np.zeros((user_num, itemset_len), dtype=int)
    record = np.zeros(total_step)
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        data = zipf_generation(a, x, user_num, data_len, rate=True)
        transition(x, 0.2)
        budgets = np.full(user_num, epsilon/omega)
        release_data = sampling_randomizer(data, itemset_len, budgets, rate=True)
        freq = personalized_frequency_estimation(release_data, itemset_len, budgets)
        true_freq = calc_freq(data, itemset_len)
        record[step] = utility_metrics(freq, true_freq, k)
    print('Mean:', record.mean())

def exp_sample():
    itemset_len = 1000
    k = 100
    total_step = 10
    user_num = 100000
    data_len = 20
    omega = 5
    epsilon = 5
    a = 1.5
    x = np.arange(1, itemset_len+1)
    release_data = np.zeros((user_num, itemset_len), dtype=int)
    record = np.zeros(total_step)
    for step in range(total_step):
        print(f'\n=========================================== STEP {step+1} ===========================================\n')
        # generate data
        data = zipf_generation(a, x, user_num, data_len, rate=True)
        transition(x, 0.2)
        if step%omega==0:
            budgets = np.full(user_num, epsilon)
            release_data = sampling_randomizer(data, itemset_len, budgets, rate=True)
        freq = personalized_frequency_estimation(release_data, itemset_len, budgets)
        true_freq = calc_freq(data, itemset_len)
        record[step] = utility_metrics(freq, true_freq, k)
    print('Mean:', record.mean())

def sample_pce_test():
    pass

def allocator_contrast():
    pass

if __name__=='__main__':
    #pce_contrast()
    #ba_test()
    #exp_ba()
    #exp_uniform()
    #random_test()
    exp_rl()