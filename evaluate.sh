#!/bin/bash

set=$1
prob_modes=(0) # 1 2
mode=0
util_mode=0
util=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
proc=$2
crit_nums=(4 8 16)
crit_modes=(0 1 2) #
period_mode=$3
obj=$4

for crit_num in "${crit_nums[@]}"
do
    for crit_mode in "${crit_modes[@]}"
    do
        for prob_mode in "${prob_modes[@]}"S
        do
            # Go through a single taskset
            for u in "${util[@]}"
            do  
                echo "Evaluate crit_num=$crit_num, proc=$proc, crit=$crit_mode, prob=$prob_mode, u=$u"
                python3 solver/job_shop_solver.py -m 0 --set-num $set --proc $proc --util $u --util-mode $util_mode --crit-mode $crit_mode --crit-num $crit_num --prob-mode $prob_mode --period-mode $period_mode --method 1 --obj "$obj"
            done
        done
    done
done
