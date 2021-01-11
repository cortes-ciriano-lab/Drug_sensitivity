#!/bin/bash

#run before: activar_rdkit
rm -r data_gdsc
rm *.log
for drug_from in "gdsc_ctrp" ; do #"gdsc_ctrp" "gdsc" "prism"
    if ["${drug_from}" == "gdsc_ctrp"] ; then
        values_from="auc"
    else
        values_from="ic50"
    fi
    for sc_from in "integrated" ; do #"pancancer" "integrated"
        mkdir -p data_gdsc/
        mkdir -p data_gdsc/${sc_from}_${drug_from}_${values_from}
        mkdir -p data_gdsc/${sc_from}_${drug_from}_${values_from}/molecular/
        mkdir -p data_gdsc/${sc_from}_${drug_from}_${values_from}/single_cell/
        
        bsub -M 15G -P gpu -gpu - -e e_gdsc.log -o o_gdsc.log -J drug_sc_proc "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets.py $sc_from $drug_from $values_from"
        #python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets.py $sc_from $drug_from $values_from
    done
done