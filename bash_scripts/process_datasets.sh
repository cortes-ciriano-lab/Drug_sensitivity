#!/bin/bash

#run before: activar_rdkit
rm -r data_gdsc
rm o_gdsc.log e_gdsc.log
values_from="ic50" # "auc" "ic50"
for drug_from in "gdsc" ; do # "gdsc" "prism"
    for sc_from in "pancancer" ; do
        mkdir -p data_gdsc/
        mkdir -p data_gdsc/${sc_from}_${drug_from}_${values_from}
        mkdir -p data_gdsc/${sc_from}_${drug_from}_${values_from}/molecular/
        mkdir -p data_gdsc/${sc_from}_${drug_from}_${values_from}/single_cell/
        
        bsub -M 15G -P gpu -gpu - -e e_gdsc.log -o o_gdsc.log -J drug_sc_proc "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets.py $sc_from $drug_from $values_from"
        #python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets.py $sc_from $drug_from $values_from
    done
done