

# average value of list
def average(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

# Calculate the order of nodes in longest path of a task
def longest_path_order(task) -> float:
    runtimes = longest_path_calc(task)
    return longest_path_order_with_runtimes(runtimes, len(task))

# Calculate the order of nodes in longest path of a  task
def longest_path_order_with_runtimes(runtimes, len_task) -> float:
    longest_path = [len_task - 1]
    
    curr_index = len_task - 1

    # Check if the current node has a precedence
    while(not runtimes[curr_index][0] == -1):
        # add the current precedence to the longest path
        longest_path.append(runtimes[curr_index][0])
        # set current precedence as new index
        curr_index = runtimes[curr_index][0]

    longest_path.reverse()
    return longest_path

# Calculate the duration of longest path of a  task
def longest_path_value(task) -> float:
    runtimes = longest_path_calc(task)
    return runtimes[-1][1]

# Calculate array to determine longest_path
def longest_path_calc(task, reverse=False):
    runtimes = []
    for i in range(len(task)):
        runtimes.append((-1, -1))
    
    if reverse:
        runtimes = longest_path_rec_reverse(task, len(task) - 1, len(task), runtimes)
    else:
        runtimes = longest_path_rec(task, 0, -1, runtimes)

    return runtimes

# Recursive function to calculate the longest runtime of the dag
def longest_path_rec(task, curr_node, prev_node, runtimes):
    # If its the last Node, end and set last runtime in last row
    if curr_node == len(task) - 1:
        if runtimes[-1][1] <= runtimes[prev_node][1]:
            runtimes[-1] = (prev_node, runtimes[prev_node][1])
        return runtimes
    
    for i in range(curr_node + 1, len(task)):
        if task[i][curr_node] == 1:
            # calculate runtime from previous plus current node
            if not prev_node == -1:
                runtime = runtimes[prev_node][1] + task[curr_node][curr_node]
            else:
                runtime = 0

            # compare runtime with current longest runtime for this node
            if runtimes[curr_node][1] <= runtime:
                runtimes[curr_node] = (prev_node, runtime)
                runtimes = longest_path_rec(task, i, curr_node, runtimes)
    
    return runtimes

# Recursive function to calculate the longest runtime of the dag
def longest_path_rec_reverse(task, curr_node, prev_node, runtimes):
    # If its the last Node, end and set last runtime in last row
    if curr_node == 0:
        if runtimes[0][1] <= runtimes[prev_node][1]:
            runtimes[0] = (prev_node, runtimes[prev_node][1])
        return runtimes
    
    for i in range(0, curr_node):
        if task[curr_node][i] == 1:
            # calculate runtime from previous plus current node
            if not prev_node == len(task):
                runtime = runtimes[prev_node][1] + task[curr_node][curr_node]
            else:
                runtime = 0

            # compare runtime with current longest runtime for this node
            if runtimes[curr_node][1] <= runtime:
                runtimes[curr_node] = (prev_node, runtime)
                runtimes = longest_path_rec_reverse(task, i, curr_node, runtimes)
    
    return runtimes

# Get all Predecessors
def get_predecessors(task, sub_task_id):
    predecessors = []
    for column in range(sub_task_id):
        if task[sub_task_id][column] == 1:
            predecessors.append(column)
    
    return predecessors


def calculate_deadline(taskId, subTaskId, periodId, simulationTaskset):
    lowestDeadline = simulationTaskset[taskId][subTaskId][periodId]["period_deadline"]
    for (t, s, p) in simulationTaskset[taskId][subTaskId][periodId]["successor"]:
        if simulationTaskset[t][s][p]["deadline"] == 0:
            #print(f'{(t, s, p)}')
            calculate_deadline(t, s, p, simulationTaskset)
        
        lowestDeadline = min(lowestDeadline, simulationTaskset[t][s][p]["deadline"] - simulationTaskset[t][s][p]["execution"])

    simulationTaskset[taskId][subTaskId][periodId]["deadline"] = lowestDeadline
    #print(f'Selected Deadline {}')

    return lowestDeadline

def calculate_deadline_print(taskId, subTaskId, periodId, simulationTaskset, prevT, prevS, prevP):# Testing method for more informations

    lowestDeadline = simulationTaskset[taskId][subTaskId][periodId]["period_deadline"]
    for (t, s, p) in simulationTaskset[taskId][subTaskId][periodId]["successor"]:
        if simulationTaskset[t][s][p]["deadline"] == 0:
            calculate_deadline_print(t, s, p, simulationTaskset, taskId, subTaskId, periodId)

        lowestDeadline = min(lowestDeadline, simulationTaskset[t][s][p]["deadline"] - simulationTaskset[t][s][p]["execution"])

    simulationTaskset[taskId][subTaskId][periodId]["deadline"] = lowestDeadline
    print(f'Selected Deadline {simulationTaskset[taskId][subTaskId][periodId]["deadline"]} for {taskId} {subTaskId} {periodId} from {prevT} {prevS} {prevP}')

    return lowestDeadline

# convert to taskset for simulation
def convert_for_simulation(taskset, crit_machines, assigned_jobs, horizon, skip):
    simulationTaskset = [[[] for _ in range(len(task))] for (_, task) in enumerate(taskset)]

    # task = {
    #   id: ,
    #   execution: ,
    #   predecessor: counter,
    #   successor: [],
    #   deadline: TODO,
    #   crit: ,
    #   released: false,
    #   release: ,
    # }

    # Convert all tasks
    for (t, task) in enumerate(taskset):
        period = task[-1][-1]
        periods = int(horizon/period)
        #runtimes = longest_path_calc(task, reverse=True)

        for (x, _) in enumerate(task):
            simulationTaskset[t][x] = [None] * periods
            for period_id in range(periods):
                period_release = int(period * period_id)
                period_deadline = int(period * (period_id + 1))
                if x == 0 or x == len(task) - 1:
                    simulationTaskset[t][x][period_id] = {
                        "id": (t, x, period_id),
                        "period_deadline": period_deadline,
                        "execution": 0,
                        "predecessor": 0,
                        "successor": [],
                        "deadline": 0,
                        "crit": -1,
                        "status": 0,
                        "release": period_release 
                    }
                else:
                    simulationTaskset[t][x][period_id] = {
                        "id": (t, x, period_id),
                        "period_deadline": period_deadline,
                        "execution": task[x][x],
                        "predecessor": 0,
                        "successor": [],
                        "deadline": 0,
                        "crit": task[0][x],
                        "status": 0,
                        "release": period_release 
                    }

                # compute predecessors/successors inside of tasks
                for y in range(0, x):
                    if task[x][y] == 1:
                        simulationTaskset[t][y][period_id]["successor"].append((t, x, period_id))
                        simulationTaskset[t][x][period_id]["predecessor"] += 1

    # compute solver predecessors/successors
    if not skip:
        for machine in crit_machines:
            if len(assigned_jobs[machine]) == 0:
                continue    

            assigned_jobs[machine].sort(key=lambda x: x.start)

            lastTask = assigned_jobs[machine][0].task
            lastSubTask = assigned_jobs[machine][0].index
            lastPeriod = assigned_jobs[machine][0].period

            for assigned_task in assigned_jobs[machine][1:]:
                task = assigned_task.task
                subTask = assigned_task.index
                period_id = assigned_task.period
                simulationTaskset[lastTask][lastSubTask][lastPeriod]["successor"].append((task, subTask, period_id))

                simulationTaskset[task][subTask][period_id]["predecessor"] += 1

                lastTask = task
                lastSubTask = subTask
                lastPeriod = period_id

    error = False

    # Go from latest to earliest to minimize chance of recursion error depth
    for p in range(horizon - 1, -1, -1):
        for (t, task) in enumerate(taskset):
            period_id = p/task[-1][-1]
            if not period_id.is_integer():
                continue

            period_id = int(period_id)

            for sub_task_id in range(len(task[0]) - 1, -1, -1):
                try:
                    if simulationTaskset[t][sub_task_id][period_id]["deadline"] != 0:
                        continue
                except IndexError:
                    print(f'{t=} {sub_task_id=} {period_id=}')
                    print(f'{task[-1][-1]=} {p=} {p/task[-1][-1]=}')
                    print(f'{len(simulationTaskset[t])=}')
                    print(f'{len(simulationTaskset[t][sub_task_id])=}')
                    print(f'{len(simulationTaskset[t][sub_task_id][period_id])=}')


                deadline = calculate_deadline(t, sub_task_id, period_id, simulationTaskset)

            if deadline < simulationTaskset[t][0][period_id]["release"]:
                print(f'SubTask ({t}, {sub_task_id}, {period_id}) has Deadline lower than its release => unscheduable')
                error = True
                return None

    if error:
        return None
    return simulationTaskset
    #print(simulationTaskset)

def addHints(solver, model, all_tasks):
    for (t, s, p) in all_tasks:
        model.AddHint(all_tasks[t, s, p].start, solver.Value(all_tasks[t, s, p].start))
        #model.AddHint(all_tasks[t, s, p].end, solver.Value(all_tasks[t, s, p].end))

def addHintsSpread(solver, model, bool_vars):
    for (t, s, p) in bool_vars:
        bool_var = bool_vars[t, s, p]
        model.AddHint(bool_var, solver.BooleanValue(bool_var))