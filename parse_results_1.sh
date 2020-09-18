#!/bin/bash
type_data="single_cell"
rm list_best_parameters_pancancer.txt loss_results_${type_data}_pancancer.txt check_cases.txt e_parse.log o_parse.log summary_results.csv
for file in new_results/${type_data}/pancancer/out*
do
    echo ${file} >> loss_results_${type_data}_pancancer.txt
done

#bsub -P gpu -gpu - -M 30G -e e_parse.log -o o_parse.log -J parse_res "python py_scripts/parse_results.py $type_data"
python py_scripts/parse_results.py $type_data
