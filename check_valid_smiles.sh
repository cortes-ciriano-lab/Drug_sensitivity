#!/bin/bash

rm e2.log o2.log 
rm -r data/10times
mkdir -p data/10times/
bsub -q research-rh74 -P gpu -gpu - -M 50G -e e2.log -o o2.log -J check "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/check_valid_smiles.py"
