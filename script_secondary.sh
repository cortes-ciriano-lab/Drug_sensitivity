#!/bin/bash

bgadd -L 15 /drug_secondary

#NNet
run_type="start"
type_data="secondary"
for lr in "0.00001" "0.0005" "0.00005" "0.000001" "0.1" "0.5" "0.001" "0.005" "0.01" "0.05" "0.0001" ; do #"0.00001" "0.0005" "0.00005" "0.000001" "0.1" "0.5" "0.001" "0.005" "0.01" "0.05" "0.0001"
	for size_batch in "64" ; do
		for n_epoch in "400" ; do
			for type_lr in "non_cyclical" ; do # "non_cyclical" "cyclical" 
				if [ "${type_lr}" == "cyclical" ] ; then
					epoch_reset="200"
					step="50"
				elif [ "${type_lr}" == "non_cyclical" ] ; then
					epoch_reset="${n_epoch}"
					step="100"
				fi
				for dropout in "0.1"  ; do
					for gam in "0.6" ; do
						for seed in "42" ; do
							for model in "lGBM"; do # #"RF" "NNet" "lGBM"
								if [ "${model}" == "NNet" ] ;  then
									perc_train="0.7"
									perc_val="0.15"
									model_info="64_32 128_128_64 128_64_32_16 64_32_16"
								elif [ "${model}" == "RF" ] || [ "${model}" == "lGBM" ] ; then
									perc_train="0.7"
									perc_val="0.3"
									model_info="100"
								fi
								for network_info in $model_info ; do
									for data_from in "pancancer" ; do #"mcfarland", "science"
										for type_split in "random" ; do #"leave-one-cell-line-out" "leave-one-tumour-out"
											if [ "${data_from}" == "pancancer" ] && [ "${type_split}" == "leave-one-cell-line-out" ] ; then
												file_lines="/hps/research1/icortes/acunha/python_scripts/Single_cell/data/pancancer_cell_lines.txt"
											elif [ "${data_from}" == "pancancer" ] && [ "${type_split}" == "leave-one-cell-line-out" ] ; then
												file_lines="/hps/research1/icortes/acunha/python_scripts/Single_cell/data/pancancer_tumours.txt"
											elif [ "${data_from}" == "pancancer" ] && [ "${type_split}" == "random" ] ; then
												echo "${perc_val}" > "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/random_value_split.txt"
												file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/random_value_split.txt"
											fi
											for to_test in `cat ${file_lines}` ; do
												mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/"
												mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}"
												mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}/${data_from}"
												mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}/${data_from}/${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}" && cd $_
												mkdir -p pickle model_values plots

												bsub -g /drug_secondary -P gpu -gpu - -M 10G -e e.log -o o.log -J drug_sec "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity.py $type_data $network_info $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_split $to_test $data_from $model $run_type"
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

