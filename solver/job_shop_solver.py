import collections
import sys
import getopt
import json
from ortools.sat.python import cp_model
#import plotly.figure_factory as ff
import numpy as np
import simulator.simulator as Simulator
import os
import time

from helper import convert_for_simulation, average, addHintsSpread
import solver.job_shop_solver_methods as jsm
from solver.enums import Method

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:p:u:", ["mode=", "set-num=", "util-mode=", "util=", "proc=", "crit-num=", "crit-mode=", "prob-mode=", "period-mode=", "method=", "obj="])
    except getopt.GetoptError:
        print("TODO")
        sys.exit(2)

    # Vordefinierte Werte
    show = False
    path = "experiments"
    objectives = [4, 2]
    periods = [1, 2 , 5, 10, 20, 50, 100]
    allowedTime = 900
    numberWorkers = 64

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
        elif opt in ("--method"):
            method = int(arg)
        elif opt in ("--obj"):
            objectives = list(map(int, arg.split(",")))
        else:
            print("Unknown parameter")
            sys.exit(1)

    results = f"experiments/tasksets/{set_num}/{mode}/{util_mode}/{crit_mode}/{prob_mode}/p_{processors}_u_{util:.2f}_cn_{crit_num}_pm_{period_mode}_method_{method}_obj_{'_'.join(str(e) for e in objectives)}.json"
    if os.path.exists(results):
        print("Results already exists")
        exit(0)

    # read taskset
    f = f"{path}/tasksets/{set_num}/{mode}/{util_mode}/{crit_mode}/{prob_mode}/p_{processors}_u_{util}_cn_{crit_num}_pm_{period_mode}.npy"

    tasksets = np.load(f)
    # Set few values to track performance of Taskset
    result = {
        "success": 0,
        "failed": 0,
        "total": len(tasksets),
        "noSolution": {
            "counter": 0,
            "timeout": 0,
            "prevFound": 0,
            "incomplete": 0,
            "incompletePass": 0
        },
        "ratio": 0,
        "deadlineUnscheduable": 0,
        "sets": []
    }

    for (set_id, taskset) in enumerate(tasksets):
        jobs_data = []
        
        horizon = 1
        for task in taskset:
            horizon = int(max(horizon, task[-1][-1]))
        
        integer_multiplier = 1.0e10
        
        # Problems can occur because of the Spreading objectives
        if horizon > 1:
            horizon = 100 
        
        # init integer multiplier in jsm
        jsm.init_globals(integer_multiplier, periods, horizon)

        # create jobs for each task
        curr_index = crit_num # for decomposed
        for (task_id, task) in enumerate(taskset):
            if method == Method.DELAY.value:
                job = jsm.delay_task_to_job(task, curr_index)
                curr_index += 1

                jobs_data.append((task_id, job))
            elif method == Method.DECOMPOSED.value:
                (jobs, curr_index) = jsm.decomposed_task_to_job(task, curr_index, task_id)

                jobs_data.extend(jobs)
            elif method == Method.UNCOMPOSED.value:
                job = jsm.uncomposed_task_to_job(task, curr_index)
                curr_index += 1

                jobs_data.append((task_id, job))


        if method == Method.DELAY.value:
            machines_count = crit_num + len(taskset)
        elif method == Method.DECOMPOSED.value:
            machines_count = curr_index
        elif method == Method.UNCOMPOSED.value:
            machines_count = crit_num + len(taskset)

        all_machines = range(machines_count)

        # Create the model.
        model = cp_model.CpModel()

        # Named tuple to store information about created variables. TODO an bessere Stelle bauen
        task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information. TODO an bessere Stelle bauen
        assigned_task_type = collections.namedtuple('assigned_task_type',
                                                    'start job task index duration period')

        # Creates job intervals and add to the corresponding machine lists.
        print(f'{set_id}/{100}: Creating all Tasks', end="\r")
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for (job_id, (task_id, job)) in enumerate(jobs_data):
            period = taskset[task_id][-1][-1]
            for (sub_task_id, sub_job) in job.items():
                machine = sub_job[0]
                duration = sub_job[1]
                for period_id in range(int(horizon/period)):
                    suffix = '_%i_%i_%i' % (job_id, sub_task_id, period_id)
                    start = int(period_id * period * integer_multiplier) 
                    end = int((period_id + 1) * period * integer_multiplier) 
                    start_var = model.NewIntVar(start, end, 'start' + suffix)
                    end_var = model.NewIntVar(start, end, 'end' + suffix)
                    interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
                    all_tasks[job_id, sub_task_id, period_id] = task_type(start=start_var, end=end_var, interval=interval_var)
                    
                    # The start is always at the beginning of a period
                    if sub_task_id == 0:
                        model.AddHint(start_var, int(period_id * period))

                    machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        print(f'{set_id}/{100}: Adding Constraints', end="\r")
        if method == Method.DELAY.value:
            for machine in range(crit_num):
                model.AddNoOverlap(machine_to_intervals[machine])

            # Precedences inside a job.
            jsm.delay_create_delays(taskset, model, all_tasks)
        elif method == Method.DECOMPOSED.value:
            for machine in all_machines:
                model.AddNoOverlap(machine_to_intervals[machine])

            # Precedences inside a job.
            jsm.decomposed_create_constraints(taskset, jobs_data, model, all_tasks)
        elif method == Method.UNCOMPOSED.value:
            for machine in range(crit_num):
                model.AddNoOverlap(machine_to_intervals[machine])

            # Precedences inside a job.
            jsm.uncomposed_create_constraints(taskset, model, all_tasks)

        last_objective = None
        last_solver = None
        incomplete_solver = False
        obj_results = []
        time_result = []

        for (i, objective) in enumerate(objectives):
            if objective == 1:# EFD. Maximize distance to Perioddeadline.
                obj = jsm.maximum_distance(taskset, jobs_data, model, all_tasks)
            elif objective == 2:# EFL. Minimize the latest last task of a job
                obj = jsm.minimum_lateness(taskset, jobs_data, model, all_tasks)
            elif objective == 3:# 4 is similar but way faster
                (obj, bool_vars) = jsm.spread_with_distance(taskset, jobs_data, model, all_tasks, crit_num)
            elif objective == 4:# SM. Balance workload of the lowest period brackets (always assumed 1), faster version of 3
                (obj, bool_vars) = jsm.spread_with_maximum(taskset, jobs_data, model, all_tasks, crit_num)
            elif objective == 5:
                obj = jsm.minimum_distance_of_lateness(taskset, jobs_data, model, all_tasks)
            elif objective == 6: # Used for Examples Tasks
                (obj, bool_vars) = jsm.spread_with_maximum_show(taskset, jobs_data, model, all_tasks, crit_num)
            elif objective == 7: # SMNC
                (obj, bool_vars) = jsm.spread_with_maximum_no_crit_num(taskset, jobs_data, model, all_tasks, crit_num)

            # Creates the solver and solve.
            print(f'{set_id}/{100}: Starting {i+1}. solver', end="\r")
            solver = cp_model.CpSolver()
            # only on server
            solver.parameters.num_search_workers = numberWorkers
            solver.parameters.max_time_in_seconds = allowedTime


            status = solver.Solve(model)
            time_result.append(solver.WallTime())

            # check for status, throw away if no solution
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                last_solver = solver
                last_objective = int(solver.ObjectiveValue())
                obj_results.append(last_objective)
                
                #addHints(solver, model, all_tasks)
                if objective == 3 or objective == 4 or objective == 7:
                    addHintsSpread(solver, model, bool_vars)
                
                # Set Value of Objective for next Solver run
                model.AddHint(obj, solver.Value(obj))
                model.Add(obj == model.NewConstant(last_objective))
            else:
                # If no Solution could be found, but an previous objective found something, use that instead
                if not last_solver == None:
                    status = cp_model.FEASIBLE
                    solver = last_solver
                    incomplete_solver = True
                    result["noSolution"]["incomplete"] += 1
                break

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)

            if method == Method.DELAY.value:
                jsm.delay_check(jobs_data, taskset, solver, all_tasks, assigned_jobs, assigned_task_type)
            elif method == Method.DECOMPOSED.value:
                jsm.decomposed_check(jobs_data, taskset, solver, all_tasks, assigned_jobs, assigned_task_type)
            elif method == Method.UNCOMPOSED.value:
                jsm.uncomposed_check(jobs_data, taskset, solver, all_tasks, assigned_jobs, assigned_task_type)

            jsm.check_critical_machines_for_overlap(assigned_jobs, crit_num)
            
            
            f = f"{path}/tasksets/{set_num}/{mode}/{util_mode}/{crit_mode}/{prob_mode}/p_{processors}_u_{util}_cn_{crit_num}_pm_{period_mode}_method_{method}_obj_{'_'.join(str(e) for e in objectives)}/{set_id}.json"
            directory_path = os.path.join(f.rsplit('/', 1)[0])
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            with open(f, "w") as file:
                j = {"jobs": assigned_jobs, "objective": last_objective, "feasible": incomplete_solver}
                json.dump(j, file)

        else:
            print("")
            print('No solution found...')
            result["failed"] += 1
            result["noSolution"]["counter"] += 1
            if solver.WallTime() >= allowedTime:
                result["noSolution"]["timeout"] += 1

            result["sets"].append({"id": set_id, "solution": status, "objectives": obj_results, "times": time_result})
            continue

        # visualize the resource machines, extra imports needed
        """ if show:
            r = lambda: random.randint(0,255)
            colors = {}
            for i in range(len( jobs_data)):
                colors[f"{i}"] = f'rgb({r()}, {r()}, {r()})'


            df = []
            for machine in all_machines[:]: #[:crit_num]
                for assigned_task in assigned_jobs[machine]:
                    df.append(dict(Task="Machine " + str(machine), Start=assigned_task.start, 
                            Finish=assigned_task.start + assigned_task.duration, Resource=f"{assigned_task.job}"))

            fig = ff.create_gantt(df, colors=colors, index_col='Resource', group_tasks=True, show_colorbar=True) #, colors=colors show_colorbar=True, group_tasks=True index_col='Resource',
            fig.layout.xaxis.type = 'linear'

            fig.show() """

        # Print out of the JSS-Results
        if show:
            for machine in all_machines:
                print(f'{machine=}')
                for assigned_task in assigned_jobs[machine]:
                    print(f'{assigned_task}')

        simulationTaskset = convert_for_simulation(taskset, range(crit_num), assigned_jobs, horizon, skip=False)
        if simulationTaskset == None:
            print("")
            print('Simullationtaskset unscheduable...')
            result["failed"] += 1
            result["deadlineUnscheduable"] += 1
            result["sets"].append({"id": set_id, "solution": status, "objectives": obj_results, "deadlineUnscheduable": True, "usedLastSolver": incomplete_solver})
            continue
            

        simulator = Simulator.Simulator(simulationTaskset, processors, horizon, scheduling=2)
        
        start = int(time.time())
        simulator.dispatcher(horizon, set_id)
        end = (time.time())
        time_result.append(int(end - start))

        if simulator.hasDeadlineMiss():
            result["failed"] += 1
        else:
            result["success"] += 1
            if incomplete_solver:
                result["noSolution"]["incompletePass"] += 1
        result["sets"].append(simulator.jsonStats(status, obj_results, time_result, incomplete_solver, set_id))

    print("")
    print(f'In Total: {result["success"]} succeded, {result["failed"]} failed, {result["success"]/result["total"]} Acceptance Ratio')

    f = f"experiments/tasksets/{set_num}/{mode}/{util_mode}/{crit_mode}/{prob_mode}/p_{processors}_u_{util:.2f}_cn_{crit_num}_pm_{period_mode}_method_{method}_obj_{'_'.join(str(e) for e in objectives)}.json"
    result["ratio"] = result["success"]/result["total"]
    result["avgDeadlineMiss"] = average([result_set["error"]["deadline"] for result_set in result["sets"] if "error" in result_set and not result_set["error"] == None])

    json_string = json.dumps(result)

    directory_path = os.path.join(f.rsplit('/', 1)[0])
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open(f, "w") as outfile:
        outfile.write(json_string)
    

if __name__ == '__main__':
    main(sys.argv[1:])
