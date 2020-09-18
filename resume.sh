#!/bin/bash

bgadd -L 5 /drug_resume

#NNet
run_type="resume"
more_epoch="400"
for jobs in `cat jobs_to_resume.txt` ; do
    echo "${jobs}"
    arrJ=(${jobs//// })
    type_data="${arrJ[-3]}"
    data_from="${arrJ[-2]}"
    values="${arrJ[-1]}"
    cd "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}/${data_from}/${values}"
    echo "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}/${data_from}/${values}"
    mkdir -p pickle model_values plots
    bsub -g /drug_resume -P gpu -gpu - -M 10G -e e.log -o o.log -J drug_res "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity.py $more_epoch $run_type"
done