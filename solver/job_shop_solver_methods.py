from ortools.sat.python import cp_model
from helper import get_predecessors

def init_globals(integer_multiplier_value, periods_value, horizon_value):
    global integer_multiplier
    integer_multiplier = integer_multiplier_value

    global all_periods
    all_periods = periods_value

    global horizon
    horizon = horizon_value

############################ DELAY #####################

def delay_task_to_job(task, machine_nr):
    job = {}
    
    for x in range(len(task)):
        if x == 0 or x == len(task) - 1:
            job[x] = (machine_nr, 0)
        else:
            if task[0][x] == -1:
                pass #job.append((machine_nr, 0))# int(task[x][x] * integer_multiplier)
            else:
                job[x] = (int(task[0][x]), int(task[x][x] * integer_multiplier))

    return job

def delay_create_delays(taskset, model, all_tasks):
    for (task_id, task) in enumerate(taskset):
        period = task[-1][-1]
        periods = int(horizon/period)

        for x in range(len(task)):
            #delay_calc_delays(periods, task, task_id, x, x, 0, model, all_tasks)
            # Using below instead of above gets significant time reduce, it changes nothing regarding the end result
            if not task[0][x] == -1 or x == len(task) - 1:
                delay_calc_delays(periods, task, task_id, x, x, 0, model, all_tasks)

def delay_calc_delays(periods, task, task_id, starting_node, curr_node, curr_delay, model, all_tasks):
    if not starting_node == curr_node:
        if curr_node == 0 or not task[0][curr_node] == -1:
            for period_id in range(periods):
                delay_var = model.NewIntVar(curr_delay, curr_delay, f"delay_{period_id}_{task_id}_{starting_node}_{curr_node}_{curr_delay}")
                model.Add(all_tasks[task_id, starting_node, period_id].start >= delay_var + all_tasks[task_id, curr_node, period_id].end)
            return

        curr_delay += int(task[curr_node][curr_node] * integer_multiplier)

    for x in range(curr_node - 1, -1, -1):#range(curr_node): 
        if task[curr_node][x] == 1:
            delay_calc_delays(periods, task, task_id, starting_node, x, curr_delay, model, all_tasks)

def delay_find_prev_crit(task, starting_node, curr_node):
    prev_crit = set()
    if not starting_node == curr_node:
        if curr_node == 0 or not task[0][curr_node] == -1:
            return {curr_node}

    for x in range(curr_node):
        if task[curr_node][x] == 1:
            prev_crit.union(delay_find_prev_crit(task, starting_node, x))
    
    return prev_crit

def delay_check(jobs_data, taskset, solver: cp_model.CpSolver, all_tasks, assigned_jobs, assigned_task_type):
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        periods = range(int(horizon/taskset[job_id][-1][-1]))
        for (sub_job_id, sub_job) in job.items():
            # get all Predecessors for the subtask
            #predecessors = get_predecessors(taskset[job_id], task_id)
            predecessors = delay_find_prev_crit(taskset[task_id], sub_job_id, sub_job_id)

            if taskset[task_id][0][sub_job_id] == -1:
                continue

            # Check if the end of a predecessor is after the start
            for pred in predecessors:# TODO Completly wrong momentan
                if taskset[task_id][0][pred] == -1 and not pred == 0:
                    for period_id in periods:
                        start = solver.Value(all_tasks[job_id, sub_job_id, period_id].start)
                        # if normal node the end is useless, you need the delay
                        if solver.Value(all_tasks[job_id, pred, period_id].start) + int(taskset[task_id][pred][pred] * integer_multiplier) >= start:
                            print(f'Error 1 with Task {job_id}:{task_id} for Predesscor {pred} in period {period_id}, self is critical {taskset[task_id][0][sub_job_id]}, pred is critical {taskset[task_id][0][pred]}, Difference {solver.Value(all_tasks[job_id, pred, period_id].start) + int(taskset[task_id][pred][pred] * integer_multiplier) - start}')
                            print(f'{solver.Value(all_tasks[job_id, pred, period_id].start)=}, {int(taskset[task_id][pred][pred] * integer_multiplier)=}, {start=}')
                else:
                    for period_id in periods:
                        start = solver.Value(all_tasks[job_id, sub_job_id, period_id].start)
                        if solver.Value(all_tasks[job_id, pred, period_id].end) > start:
                            print(f'Error 2 with Task {job_id}:{sub_job_id} for Predesscor {pred} in period {period_id}, self is critical {taskset[task_id][0][sub_job_id]}, pred is critical {taskset[task_id][0][pred]}')

            machine = sub_job[0]
            for period_id in periods:
                assigned_jobs[machine].append(
                    assigned_task_type( start=solver.Value(all_tasks[job_id, sub_job_id, period_id].start),
                                        job=job_id,
                                        task=task_id,
                                        index=sub_job_id,
                                        duration=sub_job[1],
                                        period=period_id))

#################### END DELAY ##############################

#################### DECOMPOSED ##############################

def decomposed_task_to_job(task, curr_index, task_id):# TODO Refactor jobs to dict
    critical_job = {}

    # go through task and create Job with only critical Nodes
    for i in range(len(task)):
        if i == 0 or i == len(task) - 1:
            critical_job[i] = (curr_index, 0)
        else:
            if task[0][i] == -1:
                pass #job.append((machine_nr, 0))# int(task[x][x] * integer_multiplier)
            else:
                critical_job[i] = (int(task[0][i]), int(task[i][i] * integer_multiplier))
            
    curr_index += 1

    # start recursive function
    jobs = [(task_id, critical_job)]
    curr_index = decomposed_rec_task_to_job(task, {}, 0, curr_index, jobs, task_id)
    return (jobs, curr_index)

def decomposed_rec_task_to_job(task, current_job, last_node, curr_index, jobs, task_id):
    # if last Node, add current job as new job 
    if last_node == len(task) - 1:
        # Update all machines
        current_job[last_node] = (curr_index, 0)
        jobs.append((task_id, current_job))
        return curr_index + 1

    if last_node == 0:
        current_job[0] = (curr_index, 0)
    else:
        # add actual node
        current_job[last_node] = (curr_index, int(task[last_node][last_node] * integer_multiplier))

    for i in range(last_node + 1, len(task)):
        if task[i][last_node] == 1:
            # If new way opens up, add it to current job
            copied_job = current_job.copy()

            curr_index = decomposed_rec_task_to_job(task, copied_job, i, curr_index, jobs, task_id)

            for (sub_job_id, sub_job) in current_job.items():
                current_job[sub_job_id] = (curr_index, sub_job[1])
    
    return curr_index

def decomposed_create_constraints(taskset, jobs_data, model, all_tasks):
    critical_job_id = 0
    current_task_id = 0
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        task = taskset[task_id]
        period = task[-1][-1]

        if not current_task_id == task_id:
            current_task_id = task_id
            critical_job_id = job_id

        for (sub_task_id, sub_job) in job.items():
            for column in range(0, sub_task_id):
                if task[sub_task_id][column] == 1:
                    #print(f'{row=}, {column=}, {len(job)=}')
                    try:
                        if not task[0][sub_task_id] == -1 and not sub_task_id == 0 and not sub_task_id == len(task) - 1:
                            if not task[0][column] == -1 and not sub_task_id == 0 and not sub_task_id == len(task) - 1:
                                for period_id in range(int(horizon/period)):
                                    model.Add(all_tasks[critical_job_id, sub_task_id, period_id].start >= all_tasks[critical_job_id, column, period_id].end)
                            else:
                                for period_id in range(int(horizon/period)):
                                    model.Add(all_tasks[critical_job_id, sub_task_id, period_id].start >= all_tasks[job_id, column, period_id].end)
                        else:
                            if not task[0][column] == -1 and not sub_task_id == 0 and not sub_task_id == len(task) - 1:
                                for period_id in range(int(horizon/period)):
                                    model.Add(all_tasks[job_id, sub_task_id, period_id].start >= all_tasks[critical_job_id, column, period_id].end)
                            else:
                                for period_id in range(int(horizon/period)):
                                    model.Add(all_tasks[job_id, sub_task_id, period_id].start >= all_tasks[job_id, column, period_id].end) 
                    except KeyError:
                        # The Predecessor is just not part of the path
                        pass
                        
def decomposed_check(jobs_data, taskset, solver, all_tasks, assigned_jobs, assigned_task_type): 
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        for (sub_job_id, sub_job) in job.items():
            machine = sub_job[0]

            for period_id in range(int(horizon/taskset[task_id][-1][-1])):
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(all_tasks[job_id, sub_job_id, period_id].start),
                                        job=job_id,
                                        task=task_id,
                                        index=sub_job_id,
                                        duration=sub_job[1],
                                        period=period_id))

#################### END DECOMPOSED ##############################

#################### UNCOMPOSED ##############################

def uncomposed_task_to_job(task, curr_index):
    job = {}
    
    for i in range(len(task)):
        if i == 0 or i == len(task) - 1:
            job[i] = (curr_index, 0)
        else:
            machine = curr_index if task[0][i] == -1 else int(task[0][i])
            job[i] = (machine, int(task[i][i] * integer_multiplier))

    return job

def uncomposed_create_constraints(taskset, model, all_tasks):
    for (task_id, task) in enumerate(taskset):
        period = task[-1][-1]
        job_id = task_id
        for row in range(len(task)):
            for column in range(row):
                if task[row][column] == 1:
                    for period_id in range(int(horizon/period)):
                        model.Add(all_tasks[job_id, row, period_id].start >= all_tasks[job_id, column, period_id].end)

def uncomposed_check(jobs_data, taskset, solver: cp_model.CpSolver, all_tasks, assigned_jobs, assigned_task_type):
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        periods = range(int(horizon/taskset[job_id][-1][-1]))
        for (sub_job_id, sub_job) in job.items():
            # get all Predecessors for the subtask
            predecessors = get_predecessors(taskset[job_id], task_id)

            # Check if the end of a predecessor is after the start
            for pred in predecessors:
                for period_id in periods:
                    start = solver.Value(all_tasks[job_id, sub_job_id, period_id].start)
                    if solver.Value(all_tasks[job_id, pred, period_id].end) > start:
                        print(f'Error 1 with Task {job_id}:{task_id} for Predesscor {pred} in period {period_id}, self is critical {taskset[task_id][0][sub_job_id]}, pred is critical {taskset[task_id][0][pred]}, Difference {solver.Value(all_tasks[job_id, pred, period_id].start) + int(taskset[task_id][pred][pred] * integer_multiplier) - start}')
                        print(f'{solver.Value(all_tasks[job_id, pred, period_id].start)=}, {int(taskset[task_id][pred][pred] * integer_multiplier)=}, {start=}')
            
            machine = sub_job[0]
            for period_id in periods:
                assigned_jobs[machine].append(
                    assigned_task_type( start=solver.Value(all_tasks[job_id, sub_job_id, period_id].start),
                                        job=job_id,
                                        task=task_id,
                                        index=sub_job_id,
                                        duration=sub_job[1],
                                        period=period_id))

#################### END UNCOMPOSED ##############################

#################### OBJECTIVES ##############################

def maximum_distance(taskset, jobs_data, model: cp_model.CpModel, all_tasks):# Maximize distance to Perioddeadline.
    obj_vars = []
    # go through all jobs
    for (job_id, (task_id, jobs)) in enumerate(jobs_data):
        # this dictates the amount of releases for the period
        max_period = int(horizon/taskset[task_id][-1][-1])
        task = taskset[task_id]

        # go through all releases and add new var
        for p in range(max_period):
            int_var = model.NewIntVar(0, int(horizon * integer_multiplier), f'{job_id}_{p}_obj_1')
            obj_vars.append(int_var)

            # Contstraint: end of that period - end of last task in current job = distance
            model.Add(int_var == model.NewConstant(int(task[-1][-1] * (p + 1) * integer_multiplier)) - all_tasks[job_id, len(task) - 1, p].end)

    # create objective var, take the minimal distance and try to maximize that
    obj = model.NewIntVar(0, int(horizon * integer_multiplier), 'objective_1')
    model.AddMinEquality(obj, [obj_var for obj_var in obj_vars])  
    model.Maximize(obj)
    return obj

def minimum_lateness(taskset, jobs_data, model: cp_model.CpModel, all_tasks):# Minimize the latest last task of a job
    # All distance to last Periods objective.
    obj_vars = []
    # go through all jobs
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        # this dictates the amount of releases for the period
        max_period = int(horizon/taskset[task_id][-1][-1])
        task = taskset[task_id]

        # go through all releases and add new var
        for p in range(max_period):
            # TODO Inhalt auslagern in eine Methode mit 1
            int_var = model.NewIntVar(0, int(horizon * integer_multiplier), f'{job_id}_{p}_obj_2')
            obj_vars.append(int_var)
            # pretend its in the last possible release by adding the value of the last release minus current release value
            # Contstraint: extra release time + end of last task in current job = latest release
            model.Add(int_var == model.NewConstant(int(task[-1][-1] * (max_period - (p + 1)) * integer_multiplier)) + all_tasks[job_id, len(task)-1, p].end)

    # create objective var, take the maxima latest task and try to minize that
    obj = model.NewIntVar(0, int(horizon * integer_multiplier), 'objective_2')
    model.AddMaxEquality(obj, [obj_var for obj_var in obj_vars])  
    model.Minimize(obj)
    return obj

def spread_with_distance(taskset, jobs_data, model: cp_model.CpModel, all_tasks, crit_num):
    # 4 is similar but way faster
    # All distance to last Periods objective.
    critical_jobs = [[[] for _ in range(7)] for _ in range(crit_num)]
    bool_vars = {}

    for (job_id, (task_id, job)) in enumerate(jobs_data):
        task = taskset[task_id]
        period = task[-1][-1]
        index = all_periods.index(period)
        if period > 1:
            for (sub_task_id, sub_job) in job.items():
                (machine, execution) = sub_job
                
                if machine < crit_num:
                    critical_jobs[machine][index].append((job_id, sub_task_id, execution))
                    
                    for period_id in range(int(horizon/period)):
                        bool_jobs_vars = []
                        for p in range(int(period_id * period), int((period_id + 1) * period), 1):
                            bool_var = model.NewBoolVar(f'{job_id}_{sub_task_id}_{p}_bool_3')
                            bool_vars[job_id, sub_task_id, p] = bool_var
                            bool_jobs_vars.append(bool_var)


                            model.Add(model.NewConstant(int(p * integer_multiplier)) <= all_tasks[job_id, sub_task_id, period_id].start).OnlyEnforceIf(bool_var)
                            model.Add(model.NewConstant(int((p + 1) * integer_multiplier)) > all_tasks[job_id, sub_task_id, period_id].start).OnlyEnforceIf(bool_var)

                        # Only the amount of times this job is released can be the sum for the bool_vars.
                        model.Add(sum(bool_jobs_vars) == 1)

    period_vars = []
    distance_periods = []
    for c in range(crit_num):
        for p in range(int(horizon)):
            int_var = model.NewIntVar(0, int(integer_multiplier * horizon), f'{c}_{p}_obj_3')
            period_vars.append(int_var)

            #sum_vars = [( job_id, sub_task_id, execution) for c_jobs in critical_jobs[1:] for (job_id, sub_task_id, execution) in c_jobs]
            #[model.Add(sum_var == execution).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p]) for (sum_var, job_id, sub_task_id, execution) in sum_vars]
            #[model.Add(sum_var == 0).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p].Not()) for (sum_var, job_id, sub_task_id, execution) in sum_vars]
            
            sum_vars = []
            for c_jobs in critical_jobs[c][1:]:
                for (job_id, sub_task_id, execution) in c_jobs:
                    sum_var = model.NewIntVar(0, execution, f'{job_id}_{sub_task_id}_{p}_sum_var_3')
                    model.Add(sum_var == execution).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p])
                    model.Add(sum_var == 0).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p].Not())
                    sum_vars.append(sum_var)

            model.Add(int_var == sum(sum_vars))

                
        max_period = model.NewIntVar(0, int(horizon * integer_multiplier), f'{c}_max_period_3')
        model.AddMaxEquality(max_period, [period_var for period_var in period_vars])
        min_period = model.NewIntVar(0, int(horizon * integer_multiplier), f'{c}_min_period_3')
        model.AddMinEquality(min_period, [period_var for period_var in period_vars])
        distance_period = model.NewIntVar(0, int(horizon * integer_multiplier), f'{c}_distance_period_3')
        model.Add(distance_period == max_period - min_period)
        distance_periods.append(distance_period)

    obj = model.NewIntVar(0, int(horizon * integer_multiplier), "distance_3")
    model.AddMaxEquality(obj, distance_periods)
    model.Minimize(obj)
    return (obj, bool_vars)

def spread_with_maximum(taskset, jobs_data, model: cp_model.CpModel, all_tasks, crit_num):# Balance workload of the lowest period brackets (always assumed 1), faster version of 3
    # list for all critical jobs sorted by critical_resource and period
    critical_jobs = [[[] for _ in range(7)] for _ in range(crit_num)]
    bool_vars = {}

    # go through all jobs
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        task = taskset[task_id]
        period = task[-1][-1]
        index = all_periods.index(period) # index in critical_jobs
        # only look at periods higher 1, because tasks with period 1 are always there and so dont make a difference, just add performance loss
        if period > 1:
            # go through all subjobs
            for (sub_task_id, sub_job) in job.items():
                (machine, execution) = sub_job
                
                # if critical
                if machine < crit_num:
                    # add to our list
                    critical_jobs[machine][index].append((job_id, sub_task_id, execution))
                    # go through all possible releases
                    for period_id in range(int(horizon/period)):
                        # list for all bool_vars for each period in a release
                        # e.g. Period of 5 -> all vars for period 0-1, 1-2, 2-3, 3-4, 4-5 then next 5-6, 6-7 etc.
                        bool_jobs_vars = []
                        # go through all 
                        for p in range(int(period_id * period), int((period_id + 1) * period), 1):
                            bool_var = model.NewBoolVar(f'{job_id}_{sub_task_id}_{p}_bool_4')
                            bool_vars[job_id, sub_task_id, p] = bool_var
                            bool_jobs_vars.append(bool_var)


                            model.Add(model.NewConstant(int(p * integer_multiplier)) <= all_tasks[job_id, sub_task_id, period_id].start).OnlyEnforceIf(bool_var)
                            model.Add(model.NewConstant(int((p + 1) * integer_multiplier)) > all_tasks[job_id, sub_task_id, period_id].start).OnlyEnforceIf(bool_var)

                        # Only the amount of times this job is released can be the sum for the bool_vars.
                        model.Add(sum(bool_jobs_vars) == 1)

    period_vars = []
    for c in range(crit_num):
        for p in range(int(horizon)):
            int_var = model.NewIntVar(0, int(integer_multiplier * horizon), f'{c}_{p}_obj_4')
            period_vars.append(int_var)

            sum_vars = []
            for c_jobs in critical_jobs[c][1:]:
                for (job_id, sub_task_id, execution) in c_jobs:
                    sum_var = model.NewIntVar(0, execution, f'{job_id}_{sub_task_id}_{p}_sum_var_4')
                    
                    model.Add(sum_var == execution).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p])
                    model.Add(sum_var == 0).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p].Not())
                    sum_vars.append(sum_var)

            model.Add(int_var == sum(sum_vars))

                
    obj = model.NewIntVar(0, int(horizon * integer_multiplier), "max_period_4")
    model.AddMaxEquality(obj, period_vars)
    model.Minimize(obj)
    return (obj, bool_vars)

def minimum_distance_of_lateness(taskset, jobs_data, model: cp_model.CpModel, all_tasks):# try to minimize distance of lowest and highest period
    # All distance to last Periods objective.
    obj_vars = []
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        max_period = int(horizon/taskset[task_id][-1][-1])
        task = taskset[task_id]

        for p in range(max_period):
            # Maximum ist die größe einer Periode
            int_var = model.NewIntVar(0, int(task[-1][-1] * integer_multiplier), f'{job_id}_{p}_obj_5')
            obj_vars.append(int_var)

            # Contstraint: ende der Periode - ende letzten Tasks = wert
            model.Add(int_var == model.NewConstant(int(task[-1][-1] * (p + 1) * integer_multiplier)) - all_tasks[job_id, len(task) - 1, p].end)

    min_obj = model.NewIntVar(0, int(horizon * integer_multiplier), 'min_objective_5')
    max_obj = model.NewIntVar(0, int(horizon * integer_multiplier), 'max_objective_5')
    obj = model.NewIntVar(0, int(horizon * integer_multiplier), 'objective_5')
    
    model.AddMinEquality(min_obj, [obj_var for obj_var in obj_vars])
    model.AddMaxEquality(max_obj, [obj_var for obj_var in obj_vars])
    model.Add(obj == (max_obj - min_obj))

    model.Maximize(obj)
    return obj

def check_critical_machines_for_overlap(assigned_jobs, crit_num):
    for machine in range(crit_num):
        if len(assigned_jobs[machine]) == 0:
            continue

        assigned_jobs[machine].sort(key=lambda x: x.start)
        lastStart = assigned_jobs[machine][0].start
        lastDuration = assigned_jobs[machine][0].duration

        for assigned_job in assigned_jobs[machine][1:]:
            currentStart = assigned_job.start
            currentDuration = assigned_job.duration

            if lastStart + lastDuration > currentStart:
                if lastDuration == 0 or currentDuration == 0:
                    print("One Critical Job with duration 0 results in overlap")
                else:
                    print("ERROR no overlap where it should be")

            lastStart = currentStart
            lastDuration = currentDuration


def spread_with_maximum_no_crit_num(taskset, jobs_data, model: cp_model.CpModel, all_tasks, crit_num):# Balance workload of the lowest period brackets (always assumed 1), faster version of 3
    # list for all critical jobs sorted by critical_resource and period
    critical_jobs = [[] for _ in range(7)]
    bool_vars = {}

    # go through all jobs
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        task = taskset[task_id]
        period = task[-1][-1]
        index = all_periods.index(period) # index in critical_jobs
        # only look at periods higher 1, because tasks with period 1 are always there and so dont make a difference, just add performance loss
        if period > 1:
            # go through all subjobs
            for (sub_task_id, sub_job) in job.items():
                (machine, execution) = sub_job
                
                # if critical
                if machine < crit_num:
                    # add to our list
                    critical_jobs[index].append((job_id, sub_task_id, execution))
                    # go through all possible releases
                    for period_id in range(int(horizon/period)):
                        # list for all bool_vars for each period in a release
                        # e.g. Period of 5 -> all vars for period 0-1, 1-2, 2-3, 3-4, 4-5 then next 5-6, 6-7 etc.
                        bool_jobs_vars = []
                        # go through all 
                        for p in range(int(period_id * period), int((period_id + 1) * period), 1):
                            bool_var = model.NewBoolVar(f'{job_id}_{sub_task_id}_{p}_bool_7')
                            bool_vars[job_id, sub_task_id, p] = bool_var
                            bool_jobs_vars.append(bool_var)


                            model.Add(model.NewConstant(int(p * integer_multiplier)) <= all_tasks[job_id, sub_task_id, period_id].start).OnlyEnforceIf(bool_var)
                            model.Add(model.NewConstant(int((p + 1) * integer_multiplier)) > all_tasks[job_id, sub_task_id, period_id].start).OnlyEnforceIf(bool_var)

                        # Only the amount of times this job is released can be the sum for the bool_vars.
                        model.Add(sum(bool_jobs_vars) == 1)

    period_vars = []
    for p in range(int(horizon)):
        int_var = model.NewIntVar(0, int(integer_multiplier * horizon), f'{p}_obj_7')
        period_vars.append(int_var)

        sum_vars = []
        for c_jobs in critical_jobs[1:]:
            for (job_id, sub_task_id, execution) in c_jobs:
                sum_var = model.NewIntVar(0, execution, f'{job_id}_{sub_task_id}_{p}_sum_var_7')
                
                model.Add(sum_var == execution).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p])
                model.Add(sum_var == 0).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p].Not())
                sum_vars.append(sum_var)

        model.Add(int_var == sum(sum_vars))

                
    obj = model.NewIntVar(0, int(horizon * integer_multiplier), "max_period_7")
    model.AddMaxEquality(obj, period_vars)
    model.Minimize(obj)
    return (obj, bool_vars)



def spread_with_maximum_show(taskset, jobs_data, model: cp_model.CpModel, all_tasks, crit_num):# Balance workload of the lowest period brackets (always assumed 1), faster version of 3
    # list for all critical jobs sorted by critical_resource and period
    critical_jobs = [[[] for _ in range(7)] for _ in range(crit_num)]
    bool_vars = {}

    # go through all jobs
    for (job_id, (task_id, job)) in enumerate(jobs_data):
        task = taskset[task_id]
        period = task[-1][-1]
        index = [1, 30].index(period) # index in critical_jobs
        #index = all_periods.index(period) # index in critical_jobs
        # only look at periods higher 1, because tasks with period 1 are always there and so dont make a difference, just add performance loss
        if period > 1:
            # go through all subjobs
            for (sub_task_id, sub_job) in job.items():
                (machine, execution) = sub_job
                
                # if critical
                if machine < crit_num:
                    # add to our list
                    critical_jobs[machine][index].append((job_id, sub_task_id, execution))
                    # go through all possible releases
                    for period_id in range(int(horizon/period)):
                        # list for all bool_vars for each period in a release
                        # e.g. Period of 5 -> all vars for period 0-1, 1-2, 2-3, 3-4, 4-5 then next 5-6, 6-7 etc.
                        bool_jobs_vars = []
                        # go through all 
                        for p in range(int(period_id * period), int((period_id + 1) * period), 5):
                        #for p in range(int(period_id * period), int((period_id + 1) * period), 1):
                            bool_var = model.NewBoolVar(f'{job_id}_{sub_task_id}_{p}_bool_6')
                            bool_vars[job_id, sub_task_id, p] = bool_var
                            bool_jobs_vars.append(bool_var)


                            model.Add(model.NewConstant(int(p)) <= all_tasks[job_id, sub_task_id, period_id].start).OnlyEnforceIf(bool_var)
                            model.Add(model.NewConstant(int((p + 5))) > all_tasks[job_id, sub_task_id, period_id].start).OnlyEnforceIf(bool_var)

                        # Only the amount of times this job is released can be the sum for the bool_vars.
                        model.Add(sum(bool_jobs_vars) == 1)

    period_vars = []
    for c in range(crit_num):
        for p in range(0, horizon, 5):
            int_var = model.NewIntVar(0, int(integer_multiplier * horizon), f'{c}_{p}_obj_6')
            period_vars.append(int_var)

            sum_vars = []
            for c_jobs in critical_jobs[c][1:]:
                for (job_id, sub_task_id, execution) in c_jobs:
                    sum_var = model.NewIntVar(0, execution, f'{job_id}_{sub_task_id}_{p}_sum_var_6')
                    
                    model.Add(sum_var == execution).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p])
                    model.Add(sum_var == 0).OnlyEnforceIf(bool_vars[job_id, sub_task_id, p].Not())
                    sum_vars.append(sum_var)

            model.Add(int_var == sum(sum_vars))

                
    obj = model.NewIntVar(0, int(horizon * integer_multiplier), "max_period_6")
    model.AddMaxEquality(obj, period_vars)
    model.Minimize(obj)
    return (obj, bool_vars)