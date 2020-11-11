#!/bin/bash

output=$(cat list_best_parameters_pancancer_with.txt | sed -n 1p)
folder=$(cat list_best_parameters_pancancer_with.txt | sed -n 2p)
mkdir -p best_results/ best_results/pancancer_0.00005/
cp -r results/pancancer_0.00005/$folder best_results/pancancer_0.00005/
echo $folder
cp results/pancancer_0.00005/"output_${output}.txt" best_results/pancancer_0.00005/

output=$(cat list_best_parameters_pancancer_without.txt | sed -n 1p)
folder=$(cat list_best_parameters_pancancer_without.txt | sed -n 2p)
mkdir -p best_results/ best_results/pancancer_1.0/
cp -r results/pancancer_1.0/$folder best_results/pancancer_1.0/
echo $folder
cp results/pancancer_1.0/"output_${output}.txt" best_results/pancancer_1.0/

echo "done!"
