#!/bin/bash

cd /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity

type="secondary" #"primary single_cell"
for type_data in $type ; do
    for type_network in "lGBM" "NNet" ; do
        if [ "${type}" == "secondary" ] ;  then
            data_from="pancancer_ic50"
        elif [ "${type}" == "single_cell" ] || [ "${type}" == "primary" ] ; then
            data_from="pancancer"
        fi
    
        rm loss_results_${type_data}_${data_from}_${type_network}.txt check_cases_${type}_${data_from}_${type_network}.txt e_parse.log o_parse.log summary_results_${type}_${data_from}_${type_network}.csv
        
        for file in `find new_results/$type/ -name output_${type_network}*` ; do
            echo ${file} >> loss_results_${type}_${data_from}_${type_network}.txt
        done
        
        python py_scripts/parse_results.py $type $data_from $type_network
        
    done
done