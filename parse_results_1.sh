#!/bin/bash
type="secondary" #"primary single_cell"
type_network="Light Gradient Boosted Machine" #"Light Gradient Boosted Machine" "Neural Network"
if [ "${type}" == "secondary" ] ;  then
    data_from="pancancer_ic50"

elif [ "${type}" == "single_cell" ] || [ "${type}" == "primary" ] ; then
    data_from="pancancer"
fi

if [ "${type_network}" == "Light Gradient Boosted Machine" ] ;  then
    output_file="output_lGBM*"

elif [ "${type_network}" == "Neural Network" ] ; then
    output_file="output_NNet*"
fi

rm loss_results_${type_data}_${data_from}.txt check_cases_${type}_${data_from}.txt e_parse.log o_parse.log summary_results_${type}_${data_from}.csv

for type_data in $type ; do
    for file in `find new_results/${type_data}/${data_from} -name $output_file`
    do
        echo ${file} >> loss_results_${type}_${data_from}.txt
    done
done

#bsub -P gpu -gpu - -M 30G -e e_parse.log -o o_parse.log -J parse_res "python py_scripts/parse_results.py $type_data"
for type_network in "Light Gradient Boosted Machine" "Neural Network" ;
do
    python py_scripts/parse_results.py $type $data_from $type_network
done
