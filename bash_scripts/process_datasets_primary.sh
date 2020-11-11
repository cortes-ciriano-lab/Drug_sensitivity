#!/bin/bash

rm e_prim.log o_prim.log invalid_smiles.txt
#rm -r data
#rm -r data/molecular/run_500/ data/molecular/run_10000/

mkdir -p data_primary/
mkdir -p data_primary/molecular/
mkdir -p data_primary/molecular/run_once/
mkdir -p data_primary/molecular/run_once/pkl_files
mkdir -p data_primary/molecular/run_once/valid_smiles
mkdir -p data_primary/single_cell/
mkdir -p data_primary/single_cell/pkl_files
mkdir -p data_primary/single_cell/csv_files
mkdir -p data_primary/pkl_files/
mkdir -p data_primary/prism_pancancer
mkdir -p data_primary/prism_pancancer/csv_files
mkdir -p data_primary/prism_pancancer/csv_files/once


bsub -P gpu -gpu - -M 40G -e e_prim.log -o o_prim.log -J prism_prim "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets_primary.py pancancer"
