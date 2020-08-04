#!/bin/bash

for file in results/single_cell/pancancer/out*
do
    cat ${file} | grep "Training loss" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Validation loss" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Testing loss" >> loss_results_pancancer.txt #tr -dc '0-9'
    
    cat ${file} | grep "Training correlation" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Validation correlation" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Testing correlation" >> loss_results_pancancer.txt #tr -dc '0-9'
    
    cat ${file} | grep "Training real values max" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Training real values min" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Training predicted values max" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Training predicted values min" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Validation real values max" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Validation real values min" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Validation predicted values max" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Validation predicted values min" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Testing real values max" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Testing real values min" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Testing predicted values max" >> loss_results_pancancer.txt #tr -dc '0-9'
    cat ${file} | grep "Testing predicted values min" >> loss_results_pancancer.txt #tr -dc '0-9'
    echo ${file} >> loss_results_pancancer.txt
    echo "$file"
done


python "py_scripts/parse_results.py"
