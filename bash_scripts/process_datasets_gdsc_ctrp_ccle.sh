#!/bin/bash

#run before: activar_rdkit

sc_from="ccle"
rm o_gdsc_ctrp_${sc_from}.log e_gdsc_ctrp_${sc_from}.log
rm -r data_gdsc_ctrp/pancancer_related/${sc_from}
mkdir -p data_gdsc_ctrp/pancancer_related/${sc_from}
memory="5G"
bsub -M 5G -e e_gdsc_ctrp_${sc_from}.log -o o_gdsc_ctrp_${sc_from}.log -J ${sc_from} "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_gdsc_ctrp_ccle.py"