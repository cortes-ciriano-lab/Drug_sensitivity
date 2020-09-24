#!/bin/bash
type="secondary" #"primary single_cell"
if [ "${type}" == "secondary" ] ;  then
    data_from="pancancer_auc"

elif [ "${type}" == "single_cell" ] || [ "${type}" == "primary" ] ; then
    data_from="pancancer"
fi

rm loss_results_${type_data}_${data_from}.txt check_cases_${type}_${data_from}.txt e_parse.log o_parse.log summary_results_${type}_${data_from}.csv

for type_data in $type ; do
    for file in new_results/${type_data}/${data_from}/out*
    do
        echo ${file} >> loss_results_${type}_${data_from}.txt
    done
done

#bsub -P gpu -gpu - -M 30G -e e_parse.log -o o_parse.log -J parse_res "python py_scripts/parse_results.py $type_data"
python py_scripts/parse_results.py $type $data_from
