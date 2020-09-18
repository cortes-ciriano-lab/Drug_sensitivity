#!/bin/bash

rm e_sec.log o_sec.log invalid_smiles.txt
#rm -r data_secondary

mkdir -p data_secondary/
mkdir -p data_secondary/molecular/
mkdir -p data_secondary/molecular/run_once/
mkdir -p data_secondary/molecular/run_once/pkl_files
mkdir -p data_secondary/molecular/run_once/valid_smiles
mkdir -p data_secondary/single_cell/
mkdir -p data_secondary/single_cell/pkl_files
mkdir -p data_secondary/single_cell/csv_files
mkdir -p data_secondary/pkl_files/
mkdir -p data_secondary/prism_pancancer
mkdir -p data_secondary/prism_pancancer/csv_files
mkdir -p data_secondary/prism_pancancer/csv_files/once


bsub -P gpu -gpu - -M 40G -e e_sec.log -o o_sec.log -J prism_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets_secondary.py pancancer"