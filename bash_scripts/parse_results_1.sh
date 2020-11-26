#!/bin/bash

cd /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity

type="secondary" #"primary single_cell"
for pathway in "kegg_pathways" ; do #"no_pathway" "canonical_pathways" "kegg_pathways" "chemical_genetic_perturbations" 
	for num_genes in "all_genes" ; do #"best_7000" "all_genes"
        combination="${pathway}_${num_genes}"
        if [ "${combination}" == "all_genes_no_pathway" ] || [ "${combination}" == "all_genes_canonical_pathways" ] || [ "${combination}" == "all_genes_chemical_genetic_perturbations" ] ; then
            echo "Try later"
        else
            for type_data in $type ; do
                for type_network in "RF" ; do #"NNet" "lGBM" "yrandom" "linear" ; do
                    if [ "${type}" == "secondary" ] ;  then
                        data_from="pancancer_ic50"
                    elif [ "${type}" == "single_cell" ] || [ "${type}" == "primary" ] ; then
                        data_from="pancancer"
                    fi
                
                    rm loss_results_${type_data}_${data_from}_${type_network}_${combination}.txt check_cases_${type}_${data_from}_${type_network}_${combination}.txt summary_results_${type}_${data_from}_${type_network}_${combination}.csv
                    
                    for file in `find new_results/$type/ -name output_${type_network}*` ; do
                        echo ${file} >> loss_results_${type}_${data_from}_${type_network}_${combination}.txt
                    done
                    
                    python py_scripts/parse_results.py $type $data_from $type_network $combination
                done
            done
        fi
        
    done
done
