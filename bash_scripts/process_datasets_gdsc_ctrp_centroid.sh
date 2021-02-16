#!/bin/bash

#run before: activar_rdkit
#rm -r data_gdsc_ctrp

for sc_from in "integrated_centroids" ; do #"pancancer_centroids" ; do #"pancancer" "integrated"
    if [ "${sc_from}" == "pancancer_centroids" ] ; then
        memory="60G"
        rm -r data_gdsc_ctrp/pancancer_related/${sc_from}
        mkdir -p data_gdsc_ctrp/pancancer_related/${sc_from}
    elif [ "${sc_from}" == "integrated_centroids" ] ; then
        memory="150G"
        rm -r data_gdsc_ctrp/integrated/${sc_from}
        mkdir -p data_gdsc_ctrp/integrated/${sc_from}
    fi
    rm *_gdsc_ctrp_${sc_from}.log
    bsub -M $memory -e e_gdsc_ctrp_${sc_from}.log -o o_gdsc_ctrp_${sc_from}.log -J ${sc_from} "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_gdsc_ctrp_centroid.py $sc_from"
done
