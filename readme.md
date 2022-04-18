# DAG-Generator for parallel Tasks

## Generating Graphs

To generate the Graphs the [generate.sh](./generate.sh) bash file is used to call the [generator.py](./generator.py) and [generate_graphs.py](./generation/generate_graphs.py). [generator.py](./generator.py) sets the configurations for a Taskset and calculates the utilizations for all Tasks and Subtasks, as well as assigning the critical Resources. [generate_graphs.py](./generation/generate_graphs.py) generates the DAGs and contains other useful methods to manipulate them.

The taskset are saved as ".npy" files under: 

experiments/tasksets/{set_num}/{real_mode}/{util_mode}/{crit_mode}/{prob_mode}/p\_{processors}\_u\_{util}\_cn\_{crit_num}\_pm\_{period_mode}.npy

### Usage Script

Calling [generate.sh](./generate.sh) calls [generator.py](./generator.py) with multiple configurations and makes it easier to generate and quickly change settings. Multiple scripts can be set up for different purposes. [generate.sh](./generate.sh) is only there as an example. 

| Parameter | Values | Effect |
| -- | -- | -- |
| set | any Int | Set ID |
| mode | any Int | see [Usage generator](#usage-generator) |
| util | List  | see [Usage generator](#usage-generator) |
| util_mode | List | see [Usage generator](#usage-generator) |
| processors | List | number of processors |
| crit_nums | List | number of critical resources |
| crit_modes | List | see [Usage generator](#usage-generator) |
| prob_modes | List | see [Usage generator](#usage-generator) |
| period_mode | 0, 1 | see [Usage generator](#usage-generator) |

### Usage Generator

| Parameter | Values | Effect |
| -- | -- | -- |
| -m / --mode | 0,... | Custom modes can be used to overwrite the default values. Look into sourcecode for examples |
| --set-num | any Int | Set ID |
| -u / --util | float [0, 1]  | Utilization of a Taskset |
| --util-mode| 0, 1, 2 | Distribution of utilization: DRS, Uunifast or custom for examples |
| -p / --proc | any Int | number of processors |
| --crit-num | any Int | number of critical resources |
| --crit-mode | 0, 1, 2 | 5-10%, 10-40% or 40-50% critical utilization |
| --prob-mode | 0, 1, 2 | 5-10%, 10-40% or 40-50% probality for creation of Connection in G(n,p) |
| --period-mode | 0, 1 | framebased 0 or periodic 1 |


## Evaluating Tasksets

To evaluate the taskset the [evaluate.sh](./evaluate.sh) bash file is used to call the [job_shop_solver.py](./job_shop_solver.py). [job_shop_solver.py](./job_shop_solver.py) goes through the whole process of evaluation. First it creates and runs the JS-Solver. Then it reads the results and converts the taskset and last starts the [simulator](./simulator/simulator.py). The constructing approaches and objectives can be found in the [jobs shop solver methods](./solver/job_shop_solver_methods.py). [Helper](./solver/helper.py) has some useful functions, for example the conversion to a simulation taskset.

### Usage Script

Calling [evaluate.sh](./evaluate.sh) calls [job_shop_solver.py](./job_shop_solver.py) with multiple configurations and makes it easier to evaluate and quickly change settings. Multiple scripts can be set up for different purposes. [evaluate.sh](./evaluate.sh) is only there as an example. 

| Parameter | Values | Effect |
| -- | -- | -- |
| set | any Int | Set ID |
| mode | any Int | see [Usage solver](#usage-solver) |
| util | List  | see [Usage solver](#usage-solver) |
| util_mode | List | see [Usage solver](#usage-solver) |
| processors | List | number of processors |
| crit_nums | List | number of critical resources |
| crit_modes | List | see [Usage solver](#usage-solver) |
| prob_modes | List | see [Usage solver](#usage-solver) |
| period_mode | 0, 1 | see [Usage solver](#usage-solver) |
| method | 0, 1 | see [Usage solver](#usage-solver) |
| obj | comma seperated list of Int | see [Usage solver](#usage-solver) |

### Usage Solver

| Parameter | Values | Effect |
| -- | -- | -- |
| -m / --mode | 0,... | Custom modes can be used to overwrite the default values. Look into sourcecode for examples |
| --set-num | any Int | Set ID |
| -u / --util | float [0, 1]  | Utilization of a Taskset |
| --util-mode| 0, 1, 2 | Distribution of utilization: DRS, Uunifast or custom for examples |
| -p / --proc | any Int | number of processors |
| --crit-num | any Int | number of critical resources |
| --crit-mode | 0, 1, 2 | 5-10%, 10-40% or 40-50% critical utilization |
| --prob-mode | 0, 1, 2 | 5-10%, 10-40% or 40-50% probality for creation of Connection in G(n,p) |
| --period-mode | 0, 1 | framebased 0 or periodic 1 |
| --method | 1, 2, 3 | Delay, SharedShop/Uncompose, Decomposed  |
| --obj | comma seperated list of Int | see below |

All of the parameters except for "--method" and "--obj" are used to find the generated sets. 

The ".json"  results files of the tasks are saved under: 

experiments/tasksets/{set_num}/{mode}/{util_mode}/{crit_mode}/{prob_mode}/p_{processors}\_u\_{util:.2f}\_cn\_{crit_num}\_pm\_{period_mode}\_method\_{method}\_obj\_{'_'.join(str(e) for e in objectives)}.json"

With the results also some added information for the sets are saved.

For the objectives the input can be done for example with: "2, 4" for EFL-SM

| Value | Objective |
| -- | -- |
| 1 | EFD |
| 2 | EFL |
| 3 | SD |
| 4 | SM |
| 5 | Testing |
| 6 | SM Custom for examples |
| 7 | SMAC |

