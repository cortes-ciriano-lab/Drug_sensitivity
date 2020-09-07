#!/bin/bash

bgadd -L 15 /drug_nnet

#NNet
for type_data in "single_cell" ; do #"bulk"
    for network_info in "64_32" "128_128_64" "128_64_32_16" "128_32" ; do
        for lr in "0.00001" ; do #"0.0005" "0.00005" "0.1" "0.5" "0.001" "0.005" "0.00001" "0.01" "0.05" "0.0001" ; do
            for size_batch in "128" "64" ; do
                for n_epoch in "400" ; do 
                    perc_train="0.7"
                    perc_val="0.15"
                    for epoch_reset in "400" ; do
                        for dropout in "0.1"  ; do 
                            for gam in "0.6" ; do
                                for seed in "42" ; do
                                    if [ "${epoch_reset}" == "500" ] ;  then
                                        step="100"
                                    elif [ "${epoch_reset}" == "400" ] ;  then
                                        step="400"
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
                                                mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/"
                                                mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}"
                                                mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}/${data_from}"
                                                mkdir -p "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results/${type_data}/${data_from}/${model}_${network_info}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_split}_${to_test}" && cd $_
                                                mkdir -p pickle model_values plots
                                                        
                                                bsub -g /drug_nnet -P gpu -gpu - -M 10G -e e.log -o o.log -J drug "python /hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/py_scripts/drug_sensitivity.py $type_data $network_info $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_split $to_test $data_from $model"
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