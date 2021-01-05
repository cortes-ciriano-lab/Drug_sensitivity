#!/bin/bash


#NNet
run_type="start"
type_data="secondary"
for model in "linear" ; do # "lGBM" "yrandom" "linear" "NNet"
	for drug_from in "gdsc" ; do
		sc_from="pancancer"
		data_from="${drug_from}_${sc_from}"
		job_group="drug_${data_from}_${model}"
		if [ "${model}" == "NNet" ] ;  then
			bgmod -L 100 /$job_group
		else
			bgmod -L 20 /$job_group
		fi
		if [ "${model}" == "NNet" ] ;  then
			perc_train="0.7"
			perc_val="0.15"
			model_info="128_128_64"
			types_learning_rates="cyclical"
			dropout_rates="0.1"
			early_stop_options="yes-80"
			lr_values="0.00001"
			total_epochs="1000"
		elif [ "${model}" == "RF" ] || [ "${model}" == "lGBM" ] || [ "${model}" == "yrandom" ] || [ "${model}" == "linear" ] ; then
			perc_train="0.7"
			perc_val="0.3"
			memory=10G
			types_learning_rates="cyclical"
			dropout_rates="0.0"
			early_stop_options="no"
			total_epochs="0"
			if [ "${model}" == "linear" ] ; then
				lr_values="0.0"
				model_info="0"
			else
				lr_values="0.1"
				model_info="100"
			fi	
		fi
		for lr in $lr_values ; do #
			for size_batch in "64" ; do
				for n_epoch in $total_epochs ; do
					for type_lr in $types_learning_rates ; do 
						
						if [ "${type_lr}" == "cyclical" ] ; then
							epoch_reset="200"
							step="50"
						elif [ "${type_lr}" == "non_cyclical" ] ; then
							epoch_reset="${n_epoch}"
							step="100"
						fi
						for dropout in $dropout_rates ; do #
							for gam in "0.6" ; do
								for seed in "42" ; do
									for network_info in $model_info ; do
										for early_stop in $early_stop_options ; do
											for pathway in "no_pathway" ; do
												for num_genes in "all_genes" ; do
													combination="${num_genes}_${pathway}"
													for type_split in "random" "random7" "leave-one-tumour-out" "leave-one-drug-out" "leave-one-cell-line-out" ; do #  
														if [ "${type_split}" == "leave-one-cell-line-out" ] ; then
															file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc/${sc_from}_${drug_from}_ic50/${drug_from}_${sc_from}_cell_lines.txt"
														elif [ "${type_split}" == "leave-one-tumour-out" ] ; then
															file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc/${sc_from}_${drug_from}_ic50/${drug_from}_${sc_from}_tumours.txt"
														elif [ "${type_split}" == "leave-one-drug-out" ] ; then
															file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc/${sc_from}_${drug_from}_ic50/${drug_from}_${sc_from}_list_drugs_only_one.txt"
														elif [ "${type_split}" == "random" ] ; then
															echo "${perc_val}" > "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc/${sc_from}_${drug_from}_ic50/random_value_split.txt"
															file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc/${sc_from}_${drug_from}_ic50/random_value_split.txt"
														elif [ "${type_split}" == "random7" ] ; then
															echo "1 2 3 4 5 6 7" > "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc/${sc_from}_${drug_from}_ic50/random_list.txt"
															file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc/${sc_from}_${drug_from}_ic50/random_list.txt"
														fi
														for to_test in `cat ${file_lines}` ; do
															FILE="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc/${data_from}/${combination}/${model}/output_${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}_${type_lr}_${early_stop}.txt"
															if [ -f "$FILE" ]; then
																echo "$FILE exists."
															else
																mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc/"
																mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc/${data_from}"
																mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc/${data_from}/${combination}"
																mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc/${data_from}/${combination}/${model}"
																mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc/${data_from}/${combination}/${model}/${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}_${type_lr}_${early_stop}" && cd $_
																mkdir -p pickle model_values plots
																if [ "${model}" == "NNet" ] ;  then
																	memory=10G
																	bsub -g /$job_group -P gpu -gpu - -M $memory -e e.log -o o.log -J drug_gdsc "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity_gdsc.py $network_info $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_split $to_test $type_lr $data_from $model $early_stop $combination $run_type"
																else
																	bsub -g /$job_group -M $memory -e e.log -o o.log -J drug_gdsc "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity_gdsc.py $network_info $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_split $to_test $type_lr $data_from $model $early_stop $combination $run_type"
																fi	
															fi
														done
													done
												done
                                            done
                                        done                                    
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

