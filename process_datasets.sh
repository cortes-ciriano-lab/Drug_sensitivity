#!/bin/bash

rm e.log o.log invalid_smiles.txt
#rm -r data/molecular/run_500/ data/molecular/run_10000/

mkdir -p data/
mkdir -p data/molecular/
mkdir -p data/molecular/run_once/
#mkdir -p data/molecular/run_500/
#mkdir -p data/molecular/run_10000/
mkdir -p data/molecular/run_once/pkl_files
mkdir -p data/molecular/run_once/valid_smiles
#mkdir -p data/molecular/run_500/pkl_files
#mkdir -p data/molecular/run_500/valid_smiles
#mkdir -p data/molecular/run_10000/pkl_files
#mkdir -p data/molecular/run_10000/valid_smiles
mkdir -p data/single_cell/
mkdir -p data/single_cell/pkl_files
mkdir -p data/single_cell/csv_files
mkdir -p data/pkl_files/
mkdir -p data/prism_pancancer


bsub -P gpu -gpu - -M 30G -e e.log -o o.log -J prism_gpu "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets.py pancancer"