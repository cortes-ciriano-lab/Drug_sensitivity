#!/bin/bash

for file in results/single_cell/pancancer/out*
do
    cat ${file} | grep "Training" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Validation" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Testing" >> loss_results_pancancer.txt #tr -dc '0-9'
    echo ${file} >> loss_results_pancancer.txt
    echo "$file"
done


python "py_scripts/parse_results.py"
