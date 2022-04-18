# DAG-Generator for parallel Tasks

## Generating Graphs

To generate the Graphs the [generate.sh](./generate.sh) bash file is used to call the [generator.py](./generator/generator.py) and [generate_graphs.py](./generator/generate_graphs.py). [generator.py](./generator/generator.py) sets the configurations for a Taskset and calculates the utilizations for all Tasks and Subtasks, as well as assigning the critical Resources. [generate_graphs.py](./generator/generate_graphs.py) generates the DAGs and contains other useful methods to manipulate them.

### Usage Script

Calling [generate.sh](./generate.sh) calls [generator.py](./generator/generator.py) with multiple configurations and makes it easier to generate and quickly change settings. Multiple scripts can be set up for different purposes. [generate.sh](./generate.sh) is only there as an example. 

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

