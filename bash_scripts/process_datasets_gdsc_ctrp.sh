#!/bin/bash

#run before: activar_rdkit
#rm -r data_gdsc_ctrp

for sc_from in "pancancer" "integrated" ; do #"pancancer" "integrated"
    #rm o_gdsc_ctrp_${sc_from}.log e_gdsc_ctrp_${sc_from}.log
    mkdir -p data_gdsc_ctrp/
    mkdir -p data_gdsc_ctrp/${sc_from}
    mkdir -p data_gdsc_ctrp/${sc_from}/molecular/
    mkdir -p data_gdsc_ctrp/${sc_from}/single_cell/
    if [ "${model}" == "pancancer" ] ; then
        memory="10G"
    else
        memory="150G"
    fi
    for type_smile_VAE in "old" ; do #"old" "fp" "new"
        if [ "${type_smile_VAE}" == "new" ] ; then
            bsub -M $memory -e e_gdsc_ctrp_${sc_from}.log -o o_gdsc_ctrp_${sc_from}.log -J ${sc_from} "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_gdsc_ctrp.py $sc_from"
        elif [ "${type_smile_VAE}" == "fp" ] ; then
            bsub -M 5G -e e_gdsc_ctrp_${sc_from}_fp.log -o o_gdsc_ctrp_${sc_from}_fp.log -J ${sc_from} "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_gdsc_ctrp_fp.py $sc_from"
        else
            rm *_gdsc_ctrp_${sc_from}_old.log
            bsub -M 5G -e e_gdsc_ctrp_${sc_from}_old.log -o o_gdsc_ctrp_${sc_from}_old.log -J ${sc_from} "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_gdsc_old_smilesVAE.py $sc_from"
        fi
    done
done