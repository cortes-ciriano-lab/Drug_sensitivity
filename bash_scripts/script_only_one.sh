#!/bin/bash

#bgadd -L 15 /drug_secondary
#bgmod -L 1 /drug_secondary
#bgmod -L 1 /drug_resume
#bgmod -L 1 /drug_primary
#bgadd -L 15 /drug_sec_ic50
#bgmod -L 50 /drug_sec_ic50


#NNet
run_type="start"
for model in "lGBM" "NNet" ; do # 
	for drug_from in "gdsc" "prism" ; do #
		sc_from="pancancer"
		data_from="${drug_from}_${sc_from}"
		job_group="drug_${data_from}_${model}"
		if [ "${model}" == "NNet" ] ;  then
			bgmod -L 5 /$job_group
		else
			bgmod -L 5 /$job_group
		fi
		if [ "${drug_from}" == "gdsc" ] ; then
			path_data="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/${sc_from}_${drug_from}_ic50/"
		else
			path_data="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/ic50/${drug_from}_${sc_from}/"
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
		elif [ "${model}" == "lGBM" ] ; then
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
											pathway="no_pathway"
											num_genes="all_genes"
											combination="${num_genes}_${pathway}"
											for type_model in "only-one-cell-test-drugs" "only-one-drug-test-cells" ; do #
												if [ "${type_model}" == "only-one-cell-test-drugs" ] ; then
													file_lines="${path_data}/${drug_from}_${sc_from}_celllines2indexes_only_one.txt"
												elif [ "${type_model}" == "only-one-drug-test-cells" ] ; then
													file_lines="${drug_from}_${sc_from}_tumours.txt"
												fi
												for to_keep in `head -1 ${file_lines}` ; do
													FILE="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_only_one/${data_from}/${combination}/${model}/output_${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_model}_${to_keep}_${type_lr}_${early_stop}.txt"
													if [ -f "$FILE" ]; then
														echo "$FILE exists."
													else
														mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_only_one/"
														mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_only_one/${data_from}"
														mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_only_one/${data_from}/${combination}"
														mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_only_one/${data_from}/${combination}/${model}"
														mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_only_one/${data_from}/${combination}/${model}/${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_model}_${to_keep}_${type_lr}_${early_stop}" && cd $_
														mkdir -p pickle model_values plots
														if [ "${model}" == "NNet" ] ;  then
															memory=15G
															bsub -g /$job_group -P gpu -gpu - -M $memory -e e.log -o o.log -J drug_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity_only_one.py $network_info $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_model $to_keep $type_lr $data_from $model $early_stop $combination $run_type"
														else
															bsub -g /$job_group -M $memory -e e.log -o o.log -J drug_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity_only_one.py $network_info $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_model $to_keep $type_lr $data_from $model $early_stop $combination $run_type"
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

