#!/bin/bash

#NNet
for type_data in "single_cell" ; do #"bulk"
    for layers in "64_32_16" ; do # 
    	for af in "elu" "hardshrink" "hardsigmoid" "hardtanh" "hardswish" "leakyrelu" "logsigmoid" "multiheadattention" "prelu" "relu" "relu6" "rrelu" "selu" "celu" "gelu" "sigmoid" "silu" "softplus" "softshrink" "softsign" "tanh" "tanhshrink" "threshold" ; do
        	for op in "adagrad" "adam" "adamw" "sparseadam" "adamax" "asgd" "lbfgs" "rmsprop" "rprop" "sgd" ; do
        		network_info="${layers}_${af}_${op}"
		        for lr in "0.01" ; do
		            for size_batch in "64" ; do #"100"
		                for n_epoch in "5" ; do
		                    perc_train="0.7"
		                    perc_val="0.15"
		                    for dropout in "0.1"  ; do #"0.5"
		                        for gam in "0.6" ; do
		                            for seed in "42" ; do
		                                if [ "${n_epoch}" != "100" ] ;  then
		                                    step="10"
		                                    epoch_reset="50"
		                                elif [ "${n_epoch}" == "500" ] ;  then
		                                    step="25"
		                                    epoch_reset="100"
		                                elif [ "${n_epoch}" == "1000" ] ;  then
		                                    step="50"
		                                    epoch_reset="200"
		                                elif [ "${n_epoch}" == "5000" ] ;  then
		                                    step="100"
		                                    epoch_reset="500"
		                                fi
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
		                                            model="NNet"
		                                            mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/"
		                                            mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/${type_data}"
		                                            mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/${type_data}/${data_from}"
		                                            mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/${type_data}/${data_from}/${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}" && cd $_
		                                            mkdir -p pickle model_values plots
		                                                    
		                                            bsub -P gpu -gpu - -M 20G -e e.log -o o.log -J drug "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity.py $type_data $network_info $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_split $to_test $data_from $model"
		                                            echo "output_${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}.txt" >> "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/list_original_parameters_${type_data}_${data_from}.txt"
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


#Random Forest
#for type_data in "single_cell" ; do #"bulk"
#    for layers in "100" ; do
#        for lr in "0.01" ; do
#            for size_batch in "64" ; do #"100"
#                for n_epoch in "500" ; do
#                    perc_train="0.7"
#                    perc_val="0.3"
#                    for dropout in "0.1"  ; do #"0.5"
#                        for gam in "0.6" ; do
#                            for seed in "42" ; do
#                                if [ "${n_epoch}" == "100" ] ;  then
#                                    step="10"
#                                    epoch_reset="50"
#                                elif [ "${n_epoch}" == "500" ] ;  then
#                                    step="25"
#                                    epoch_reset="100"
#                                elif [ "${n_epoch}" == "1000" ] ;  then
#                                    step="50"
#                                    epoch_reset="200"
#                                elif [ "${n_epoch}" == "5000" ] ;  then
#                                    step="100"
#                                    epoch_reset="500"
#                                fi
#                                for data_from in "pancancer" ; do #"mcfarland", "science"
#                                    for type_split in "random" ; do #"leave-one-cell-line-out" "leave-one-tumour-out"
#                                        if [ "${data_from}" == "pancancer" ] && [ "${type_split}" == "leave-one-cell-line-out" ] ; then
#                                            file_lines="/hps/research1/icortes/acunha/python_scripts/Single_cell/data/pancancer_cell_lines.txt"
#                                        elif [ "${data_from}" == "pancancer" ] && [ "${type_split}" == "leave-one-cell-line-out" ] ; then
#                                            file_lines="/hps/research1/icortes/acunha/python_scripts/Single_cell/data/pancancer_tumours.txt"
#                                        elif [ "${data_from}" == "pancancer" ] && [ "${type_split}" == "random" ] ; then 
#                                            echo "${perc_val}" > "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/random_value_split.txt"
#                                            file_lines="/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/random_value_split.txt"
#                                        fi
#                                        for to_test in `cat ${file_lines}` ; do
#                                            model="RF"
#                                            perc_val="0.0"
#                                            mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/"
#                                            mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/${type_data}"
#                                            mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/${type_data}/${data_from}"
#                                            mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/${type_data}/${data_from}/${model}_${layers}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}" && cd $_
#                                            mkdir -p pickle model_values plots
#                                                    
#                                            bsub -P gpu -gpu - -M 20G -e e.log -o o.log -J drug "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity.py $type_data $layers $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_split $to_test $data_from $model"
#                                            echo "output_${model}_${layers}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}.txt" >> "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/list_original_parameters_${type_data}_${data_from}.txt"
#                                        done                                    
#                                    done
#                                done
#                            done
#                        done
#                    done
#                done
#            done
#        done
#    done
#done