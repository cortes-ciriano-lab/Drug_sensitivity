#!/bin/bash

rm e_*.log o_*.log

#for pathway in "no_pathway" ; do #"canonical_pathways" "kegg_pathways" "chemical_genetic_perturbations" ; do #
#	for num_genes in "all_genes" ; do #"best_7000" "all_genes" ; do #
#        combination="${num_genes}_${pathway}"
#		type_data="secondary"
#		for type_network in "NNet" ; do #"RF" "lGBM" "yrandom" "linear" "NNet" ; do
#			data_from="pancancer_ic50"
#			rm loss_results_${type_data}_${data_from}_${type_network}_${combination}.txt check_cases_${type_data}_${data_from}_${type_network}_${combination}.txt summary_results_${type_data}_${data_from}_${type_network}_${combination}.csv
#			
#			for file in `find new_results/$type_data/pancancer_ic50/$combination/$type_network -name "output*"` ; do
#				echo $file
#				echo ${file} >> loss_results_${type_data}_${data_from}_${type_network}_${combination}.txt
#			done
#			
#			bsub -e e_${type_data}_${data_from}_${type_network}_${combination}.log -o o_${type_data}_${data_from}_${type_network}_${combination}.log "python py_scripts/parse_results.py $type_data $data_from $type_network $combination"
#		done
#    done
#done

for sc_from in "pancancer-centroids-bottlenecks" "pancancer-centroids" ; do
	for type_smile_VAE in "old" "fp" ; do #"old" "fp"
		if [ "${type_smile_VAE}" == "new" ] ; then
			name_type_smile_VAE=""
		else
			name_type_smile_VAE="_${type_smile_VAE}"
		fi
		data_from="gdsc_ctrp_${sc_from}${name_type_smile_VAE}"
		combination="all_genes_no_pathway"
		for type_network in "lGBM" ; do #"RF" "lGBM" "yrandom" "linear" "NNet" ; do
			rm loss_results_${data_from}_${type_network}_${combination}.txt check_cases_${data_from}_${type_network}_${combination}.txt summary_results_${data_from}_${type_network}_${combination}.csv	
			#for file in `find new_results_gdsc_ctrp/$data_from/$combination/$type_network -name output* | grep random7` ; do
			for file in `find new_results_gdsc_ctrp/$data_from/$type_network -name output* | grep random7` ; do
				echo $file
				echo ${file} >> loss_results_${data_from}_${type_network}_${combination}.txt
			done
			python py_scripts/parse_results.py $data_from $type_network $combination
		done
	done
done