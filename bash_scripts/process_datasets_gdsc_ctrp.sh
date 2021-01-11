#!/bin/bash

#run before: activar_rdkit
rm -r data_gdsc

for sc_from in "pancancer" "integrated"; do
    rm o_gdsc_ctrp_${sc_from}.log e_gdsc_ctrp_${sc_from}.log
    mkdir -p data_gdsc_ctrp/
    mkdir -p data_gdsc_ctrp/${sc_from}
    mkdir -p data_gdsc_ctrp/${sc_from}/molecular/
    mkdir -p data_gdsc_ctrp/${sc_from}/single_cell/
    if [ "${model}" == "pancancer" ] ; then
        memory="20G"
    else
        memory="150G"
    fi
    bsub -M $memory -e e_gdsc_ctrp_${sc_from}.log -o o_gdsc_ctrp_${sc_from}.log -J ${sc_from} "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_gdsc_ctrp.py $sc_from"
done