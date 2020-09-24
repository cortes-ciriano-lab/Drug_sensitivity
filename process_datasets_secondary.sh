#!/bin/bash

values_from="ic50" # "auc" "ic50"

mkdir -p data_secondary/
mkdir -p data_secondary/${values_from}
mkdir -p data_secondary/${values_from}/molecular/
mkdir -p data_secondary/${values_from}/molecular/run_once/
mkdir -p data_secondary/${values_from}/molecular/run_once/pkl_files
mkdir -p data_secondary/${values_from}/molecular/run_once/valid_smiles
mkdir -p data_secondary/${values_from}/single_cell/
mkdir -p data_secondary/${values_from}/single_cell/pkl_files
mkdir -p data_secondary/${values_from}/single_cell/csv_files
mkdir -p data_secondary/${values_from}/pkl_files/
mkdir -p data_secondary/${values_from}/prism_pancancer
mkdir -p data_secondary/${values_from}/prism_pancancer/csv_files
mkdir -p data_secondary/${values_from}/prism_pancancer/csv_files/once


bsub -P gpu -gpu - -M 40G -e e_sec_${values_from}.log -o o_sec_${values_from}.log -J pri_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets_secondary.py pancancer $values_from"