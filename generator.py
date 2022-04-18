import random
from drs import drs
import sys
import os
import getopt
import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

from generation import graph_generator 
from solver.helper import longest_path_value


# Return all nodes with no critical resource allocated
def get_normal_nodes(task):
    nodes = []
    for column in range(1, len(task) - 1):
        if task[0][column] == -1:
            nodes.append(column)
    
    return nodes

# Return all nodes with a critical resource allocated
def get_critical_nodes(task):
    nodes = []
    for column in range(1, len(task) - 1):
        if not task[0][column] == -1:
            nodes.append(column)
    
    return nodes

# Set the utilizations and executions times of all nodes except start and end
def set_util(task, util_mode, util_normal_total, util_critical_total, period, mode):
    # store the task utilization on task[0][-1]
    task[0][-1] = util_normal_total + util_critical_total

    tries = 0

    while tries < 10: #
        normal_nodes = get_normal_nodes(task)
        critical_nodes = get_critical_nodes(task)

        # Calculate the utilizations for normal and critical nodes respectively
        if util_mode == 0:
            util_normal = calc_util_drs(len(normal_nodes), util_normal_total, ut_max=1)
            util_critical = calc_util_drs(len(critical_nodes), util_critical_total, ut_max=1)
        elif util_mode == 1:
            util_normal = calc_util_uunifast(len(normal_nodes), util_normal_total)
            util_critical = calc_util_uunifast(len(critical_nodes), util_critical_total)
        elif util_mode == 2:
            util_normal = cacl_util_show_subtasks(len(normal_nodes))
            util_critical = cacl_util_show_subtasks(len(critical_nodes))
        else:
            print(f'ERROR: Unsupported Utilization mode  "{util_mode}"')
            sys.exit(1)

        # write down the execution time for each normal node
        for n in range(len(normal_nodes)):
            index = normal_nodes[n]
            
            task[index][index] = period * util_normal[n]
            # the utilization for each node is stored in the last column of task matrix
            task[index][-1] = util_normal[n]

        # write down the execution time for each node
        for n in range(len(critical_nodes)):
            index = critical_nodes[n]

            if mode == 2:
                task[index][index] = util_critical[n]
            else:
                task[index][index] = period * util_critical[n]
            # the utilization for each node is stored in the last column of task matrix
            task[index][-1] = util_critical[n]

        # check the longest path
        longest_path = longest_path_value(task)

        # store the critical path len(G) in task[3][-2]
        task[3][-2] = longest_path

        # store the period of task in the end node (not the execution time here)
        # please note the execution time for the general source node and end node is 0 for all tasks!
        task[-1][-1] = period

        # store the number of nodes for each task in the source node (speed up the execution)
        task[0][0] = int(len(task) - 2)

        return task
        
        #tries += 1

    #print("Maximum Number of tries reached. Stopping.")
    return None

# Set the random critical resources
def set_critical_resources_percent(task, num_res, critical_num):
    # Calculate the chance for one node to use a critical resource
    nodes = random.sample(range(1, len(task) - 2), critical_num)

    for n in range(len(task)):
        if n in nodes:
            task[0][n] = np.random.randint(0, num_res)
        else:
            task[0][n] = -1

    return task

# Calculate utilization vector with DRS
def calc_util_drs(num: int, total_util, ut_max=0.5) -> list:
    tries = 0
    while 10:#TODO tries < 100
        utilizations = drs(num, total_util)
        # max util of 0.5
        if all((ut <= ut_max) for ut in utilizations):
            return utilizations

        # safety Net
        tries += 1

    print(f'Couldnt find one drs where every task was under 0.5 in 100 tries')
    exit()

# Calculate uitilization vector with UUnifastDiscard
def calc_util_uunifast(num: int, total_util, ut_max=0.8, ut_min=0) -> list:
    tries = 0
    while tries < 10:
        # Classic UUniFast algorithm:
        utilizations = []
        sumU = total_util
        for i in range(1, num):
            nextSumU = sumU * np.random.random() ** (1.0 / (num - i))
            utilizations.append(sumU - nextSumU)
            sumU = nextSumU
        utilizations.append(sumU)

        # Discard according to specific condition:
        if all((ut <= ut_max and ut > ut_min) for ut in utilizations):
            return utilizations
        
        # safety Net
        tries += 1

def cacl_util_show_task(num: int):
    utilizations = []
    for _ in range(num):
        utilizations.append(random.randint(13, 18))

    return utilizations

def cacl_util_show_subtasks(num: int):
    utilizations = []
    for _ in range(num):
        utilizations.append(random.randint(1, 5))

    return utilizations

# Main Method
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:p:u:", ["mode=", "set-num=", "util-mode=", "util=", "proc=", "crit-num=", "crit-mode=", "prob-mode=", "period-mode="])
    except getopt.GetoptError:
        print("TODO")
        sys.exit(2)

    # Vordefinierte Werte
    period_rand = [[1, 2 , 5, 10, 20, 50, 100], [1, 10]]
    period = 1
    # number of nodes
    n_min = 20
    n_max = 50 
    task_multi = 10
    amount_sets = 100 #TODO 100
    ut_max = 0.5
    ut_min = 0.1

    path = "experiments"

    crit = [(0.05, 0.10), (0.10, 0.40), (0.40, 0.50), [0.01, 0.05]]
    prob = [(0.05, 0.10), (0.10, 0.40), (0.40, 0.50)]

    for opt, arg in opts:
        if opt == '-h':
            print("TODO Read Readme")
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = int(arg)
        elif opt in ("--set-num"):
            set_num = str(arg)
        elif opt in ("-u", "--util"):
            util = float(arg)
        elif opt in ("--util-mode"):
            util_mode = int(arg)
        elif opt in ("-p", "--proc"):
            processors = int(arg)
        elif opt in ("--crit-num"):
            crit_num = int(arg)
        elif opt in ("--crit-mode"):
            crit_mode = int(arg)
        elif opt in ("--prob-mode"):
            prob_mode = int(arg)
        elif opt in ("--period-mode"):
            period_mode = int(arg)
        else:
            print("Unknown parameter")
            sys.exit(1)
    

    real_mode = f"{mode}"

    # Different modes to override default values
    if mode == 5: # Used for comparison to other
        task_multi = 0.5
        ut_max = processors

        n_min = 50
        n_max = 70

        mode = 0

    if mode == 2: # Used for comparison to other, but in the end not used
        task_multi = 0.5
        ut_max = processors

    if mode == 3: # Used to generate example Tasks
        task_multi = 0.5
        n_min = 6
        n_max = 9
        amount_sets = 1

    if mode == 4: # Used to generate lower complexity Tasks
        n_min = 10
        n_max = 20
        task_multi = 5
        ut_max = processors

        mode = 0

    (crit_min, crit_max) = crit[crit_mode]
    (p_min, p_max) = prob[prob_mode]
    amount_tasks = int(task_multi * processors)

    if mode == 2: 
        num_nodes = [[(int(np.random.uniform(5, 20)), int(np.random.uniform(20, 40))) for _ in range(amount_tasks)] for _ in range(amount_sets)]

    file_path = f"{path}/tasksets/{set_num}/{real_mode}/{util_mode}/{crit_mode}/{prob_mode}/p_{processors}_u_{util}_cn_{crit_num}_pm_{period_mode}.npy"
    if os.path.exists(file_path):
        print(f'Already exists')
        exit(0)

    file_path = f"{path}/graphs/{set_num}/{real_mode}/{util_mode}/{crit_mode}/{prob_mode}/p_{processors}_u_{util}_cn_{crit_num}_pm_{period_mode}"
    
    error = True
    counter = 0
    while(error):
        counter += 1
        print(f'Try {counter}')
        error = False
        
        if mode == 2:
            orig_tasksets = graph_generator.create_graphs_gnp(file_path, amount_tasks, amount_sets, n_min, n_max, p_min, p_max, num_nodes)
        else:
            orig_tasksets = graph_generator.create_graphs_gnp(file_path, amount_tasks, amount_sets, n_min, n_max, p_min, p_max)
        
        # In case of those modes, preselect random period array
        if period_mode == 1 or period_mode == 2:
            period_rand_sel = period_rand[period_mode - 1]
            period_rand_len = len(period_rand_sel)

        tasksets = []    
        
        for (s, orig_taskset) in enumerate(orig_tasksets):
            taskset = []

            # Set Utilizations for each task
            if util_mode == 0:
                util_task = calc_util_drs(len(orig_taskset), processors * util, ut_max)
            elif util_mode == 1:
                util_task = calc_util_uunifast(len(orig_taskset), processors * util, ut_max)
            elif util_mode == 2:
                util_task = cacl_util_show_task(len(orig_taskset))
            else:
                print(f'ERROR: Unsupported Utilization mode  "{util_mode}"')
                sys.exit(1)

            for (t, task) in enumerate(orig_taskset):
                
                if mode == 2:
                    critical_num = num_nodes[s][t][0]
                elif mode == 0:
                    critical_num = int((len(task) - 2) * np.random.uniform(crit_min, crit_max))
                elif mode == 3:
                    critical_num = random.randint(2, 3)

                task = set_critical_resources_percent(task, crit_num, critical_num)

                if period_mode == 1 or period_mode == 2:
                    period = period_rand_sel[np.random.randint(0, period_rand_len)]

                critical_util = np.random.uniform(crit_min, crit_max) * util_task[t]
                
                if mode == 2:
                    normal_util = util_task[t] - (critical_util/period)
                elif mode == 0:
                    normal_util = util_task[t] - critical_util
                elif mode == 3:
                    critical_util = 1
                    normal_util = 1

                # Take default period
                if period_mode == 0:
                    task = set_util(task, util_mode, normal_util, critical_util, period, mode)
                # Get a random period from the predefined ranges
                elif period_mode == 1:
                    period = period_rand_sel[np.random.randint(0, period_rand_len)]
                    task = set_util(task, util_mode, normal_util, critical_util, period, mode)

                    if task is None:
                        error = True
                elif period_mode == 2:
                    period = 1
                    task = set_util(task, util_mode, normal_util, critical_util, period, mode)

                    task[-1][-1] = 30

                taskset.append(task)

            tasksets.append(taskset)
            
    # save tasksets
    f = f"{path}/tasksets/{set_num}/{real_mode}/{util_mode}/{crit_mode}/{prob_mode}/p_{processors}_u_{util}_cn_{crit_num}_pm_{period_mode}"

    # Make sure the directories exist
    if not os.path.exists(os.path.dirname(f)):
        os.makedirs(os.path.dirname(f))

    np.save(f, tasksets)

if __name__ == "__main__":
    main(sys.argv[1:])
