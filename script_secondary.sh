#!/bin/bash

#bgadd -L 15 /drug_secondary
#bgmod -L 1 /drug_secondary
#bgmod -L 1 /drug_resume
#bgmod -L 1 /drug_primary
#bgadd -L 15 /drug_sec_ic50
#bgmod -L 50 /drug_sec_ic50


#NNet
run_type="start"
type_data="secondary"
for model in "NNet" ; do # "lGBM" ; do #"RF" "NNet" "lGBM"
	if [ "${model}" == "NNet" ] ;  then
		perc_train="0.7"
		perc_val="0.15"
		model_info="10_10" #"128_128_128 256_256_128_64 256_128_64 128_128_128 512_256_128" #"64_32 128_128_64 128_64_32_16 64_32_16"
		memory=10G
		types_learning_rates="non_cyclical" # cyclical"
	elif [ "${model}" == "RF" ] || [ "${model}" == "lGBM" ] ; then
		perc_train="0.7"
		perc_val="0.3"
		model_info="10"
		memory=10G
		types_learning_rates="cyclical"
	fi
	for lr in "0.00005" ; do #"0.0001" ; do #"0.00001" "0.0005" "0.00005" "0.000001" "0.1" "0.5" "0.001" "0.005" "0.01" "0.05" "0.0001"
		for size_batch in "64" ; do
			for n_epoch in "3" ; do # "1000" ; do
				for type_lr in $types_learning_rates ; do 
					if [ "${type_lr}" == "cyclical" ] ; then
						epoch_reset="200"
						step="50"
					elif [ "${type_lr}" == "non_cyclical" ] ; then
						epoch_reset="${n_epoch}"
						step="100"
					fi
					for dropout in "0.5" ; do #"0.3" "0.6" ; do # "0.1"
						for gam in "0.6" ; do
							for seed in "42" ; do
								for network_info in $model_info ; do
									for prism_from in "ic50" ; do #"ic50" "auc"
										if [ "${prism_from}" == "ic50" ] ;  then
											job_group="drug_sec_ic50"
										elif [ "${prism_from}" == "auc" ] ; then
											job_group="drug_secondary"
										fi
										sc_from="pancancer"
										data_from="pancancer_${prism_from}"
										for early_stop in "no" "yes-80" ; do
											for type_split in "random" ; do #"leave-one-cell-line-out" "leave-one-tumour-out" ; do 
												if [ "${sc_from}" == "pancancer" ] && [ "${type_split}" == "leave-one-cell-line-out" ] ; then
													file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/${prism_from}/prism_${sc_from}/prism_pancancer_cell_lines_pancancer.txt"
												elif [ "${sc_from}" == "pancancer" ] && [ "${type_split}" == "leave-one-tumour-out" ] ; then
													file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/${prism_from}/prism_${sc_from}/prism_pancancer_tumours.txt"
												elif [ "${sc_from}" == "pancancer" ] && [ "${type_split}" == "random" ] ; then
													echo "${perc_val}" > "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/${prism_from}/prism_${sc_from}/random_value_split.txt"
													file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/${prism_from}/prism_${sc_from}/random_value_split.txt"
												fi
												for to_test in `cat ${file_lines}` ; do
													mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/"
													mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}"
													mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}/${data_from}"
													mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}/${data_from}/${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}_${type_lr}_${early_stop}" && cd $_
													mkdir -p pickle model_values plots
													bsub -g /$job_group -P gpu -gpu - -M $memory -e e.log -o o.log -J drug_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity.py $type_data-$network_info-$lr-$size_batch-$n_epoch-$perc_train-$perc_val-$dropout-$gam-$step-$seed-$epoch_reset-$type_split-$to_test-$type_lr-$data_from-$model-$run_type"#bsub -P gpu -gpu - -M $memory -e e.log -o o.log -J drug_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity.py $type_data $network_info $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_split $to_test $type_lr $data_from $model $early_stop $run_type"
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

