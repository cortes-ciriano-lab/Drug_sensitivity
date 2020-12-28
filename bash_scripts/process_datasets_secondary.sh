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

#for pathway in "no_pathway" "canonical_pathways" "kegg_pathways" "chemical_genetic_perturbations" ; do
#    for num_genes in "all_genes" "best_7000" ; do #
#        if [ "${num_genes}" == "all_genes" ] ; then
#            if [ "${pathway}" == "no_pathway" ] || [ "${pathway}" == "canonical_pathways" ] || [ "${pathway}" == "chemical_genetic_perturbations" ] ; then
#                bsub -P gpu -gpu - -M 30G -e e_sec_${values_from}_${pathway}_${num_genes}.log -o o_sec_${values_from}_${pathway}_${num_genes}.log -J pri_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets_secondary.py pancancer $values_from $num_genes $pathway"
#            else
#                echo "Done!"
#            fi
#        else
#            echo "Done!"
#        fi
#    done
#done

bsub -M 30G -e e_sec_$values_from.log -o o_sec_$values_from.log -J pri_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/process_datasets_secondary.py pancancer $values_from"