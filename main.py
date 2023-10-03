import numpy as np
import time
from algorithms.CDE import CDE
import numpy as np
import matplotlib.pyplot as plt

# set random seed to meke the results reproducible, comment this line to get different results.
random_seed = 20231003
np.random.seed(random_seed)

# experiment setting
POLICY = 'ss_policy'
problem_case = 1
run_time = 5

if problem_case == 1:
    # case 1
    N_FACILITY = 6
    MAX_FE = 500
elif problem_case == 2:
    # case 2
    N_FACILITY = 12
    MAX_FE = 1500

if POLICY == 'ss_policy':
    n_var = 2 * N_FACILITY

POPSIZE = n_var * 2
algorithms = ['CDE']

for ALGO in algorithms:

    con_graph = np.zeros((run_time, MAX_FE))
    accuracy_rate = np.zeros((run_time, MAX_FE))
    comp_time = np.zeros((run_time, ))

    if ALGO == 'CDE':
        algo = CDE(n_facility=N_FACILITY, n_var=n_var, popsize=POPSIZE,
                   n_fe=MAX_FE, plot_graph=False, policy=POLICY)

    solution = np.zeros((run_time, algo.n_var))
    function_value = np.zeros((run_time, 1))

    for i in range(run_time):
        time_start = time.process_time()
        solution[i, :], function_value[i, :], con_graph[i,
                                                        :], accuracy_rate[i, :] = algo.run()
        comp_time[i] = (time.process_time() - time_start)

    results = np.hstack((solution, function_value))

    np.save('/results/case_{}_{}_{}_results'.format(problem_case, ALGO, random_seed), results)
    np.save('/results/case_{}_{}_{}_con_graph'.format(problem_case, ALGO, random_seed), con_graph)
    np.save('/results/case_{}_{}_{}_comp_time'.format(problem_case, ALGO, random_seed), comp_time)
    np.save('/results/case_{}_{}_{}_accuracy'.format(problem_case, ALGO, random_seed), accuracy_rate)

# plot figure
def plot_graph(file, size, algorithm,ax):
    x = np.arange(size)
    data = file
    data_mean = np.mean(data, axis=0)
    ax.semilogy(x, data_mean, label=algorithm)

fig, ax = plt.subplots(figsize=[7, 7])
MAX_FE = 500

plot_graph(con_graph, MAX_FE, 'CDE', ax)

plt.xlabel('Number of simulations')
plt.ylabel('Total costs')
plt.savefig('/results/con_graph.png')
