#!/bin/bash

rm e_*.log o_*.log

for pathway in "no_pathway" ; do #"canonical_pathways" "kegg_pathways" "chemical_genetic_perturbations" ; do #
	for num_genes in "all_genes" ; do #"best_7000" "all_genes" ; do #
        combination="${num_genes}_${pathway}"
		type_data="secondary"
		for type_network in "NNet" ; do #"RF" "lGBM" "yrandom" "linear" "NNet" ; do
			data_from="pancancer_ic50"
			rm loss_results_${type_data}_${data_from}_${type_network}_${combination}.txt check_cases_${type_data}_${data_from}_${type_network}_${combination}.txt summary_results_${type_data}_${data_from}_${type_network}_${combination}.csv
			
			for file in `find new_results/$type_data/pancancer_ic50/$combination/$type_network -name "output*"` ; do
				echo $file
				echo ${file} >> loss_results_${type_data}_${data_from}_${type_network}_${combination}.txt
			done
			
			bsub -e e_${type_data}_${data_from}_${type_network}_${combination}.log -o o_${type_data}_${data_from}_${type_network}_${combination}.log "python py_scripts/parse_results.py $type_data $data_from $type_network $combination"
		done
    done
done
