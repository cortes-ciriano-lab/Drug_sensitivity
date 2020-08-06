# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import gc

from single_cell import Genexp_sc, create_report

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)

# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
def create_pancancer_bottleneck():
    gene_expression_single = pickle.load(open("/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/pancancer_dataset.pkl", "rb"))
    # gene_expression_single = gene_expression_single.iloc[:100]
    metadata_single = pickle.load(open("/hps/research1/icortes/acunha/data/PANCANCER/pancancer_metadata.pkl", "rb"))
    metadata_single = metadata_single.loc[metadata_single.index.isin(list(gene_expression_single.index))]
    
    #initialize the single cell model
    print("Single cell model: started \n ")
    genexpr = Genexp_sc()
    filename = "data/single_cell/genexp_sc_output_with_alpha.txt"
    genexpr.set_filename_report(filename)
    gene_model = genexpr.start_expression(num_genes=gene_expression_single.shape[1],
                                          path_model = "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/pancancer_with_alpha")

    # list_cell_lines = list(metadata_single["Cell_line"].unique())
    # 
    # gene_predictions = []
    # losses = {"Total_loss" : [], "Reconstruction_loss" : [], "KL_loss" : []}
    # for i in range(len(list_cell_lines)):
    #     list_indexes = list(metadata_single.loc[metadata_single["Cell_line"] == list_cell_lines[i]].index)
    #     subset = gene_expression_single.loc[gene_expression_single.index.isin(list_indexes)]
    #     output = genexpr.run_dataset(gene_model, subset.to_numpy(), "Pancancer_line_{}".format(list_cell_lines[i]))
    #     gene_bottlenecks = pd.DataFrame(output[1].cpu().numpy())
    #     gene_bottlenecks.index = list(subset.index)
    #     gene_predictions.append(gene_bottlenecks)
    #     gene_outputs = pd.DataFrame(output[0].cpu().numpy())
    #     gene_outputs.index = list(subset.index)
    #     gene_outputs.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/single_cell/csv_files/pancancer_with_alpha_outputs_{}.csv'.format(list_cell_lines[i]),
    #                                       header=True, index=False)
    #     losses['Total_loss'].append(output[2])
    #     losses["Reconstruction_loss"].append(output[3])
    #     losses["KL_loss"].append(output[4])
    # 
    # free_memory = [gene_model, output, metadata_single, gene_expression_single]
    # for item in free_memory:
    #     del item
    # gc.collect()
    # 
    # for k, v in losses.items():
    #     losses[k] = np.mean(v)
    # 
    # lines = ["* MEAN LOSSES *",
    #             "Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f} \n".format(losses['Total_loss'], losses["Reconstruction_loss"], losses["KL_loss"]),
    #             "\n"] 
    # create_report(filename, lines)
    
    # for i in range(len(gene_predictions)):
    #     if i == 0:
    #         gene_bottlenecks = gene_predictions[i]
    #     else:
    #         gene_bottlenecks = gene_bottlenecks.append(gene_predictions[i])
    
    
    list_indexes = list(gene_expression_single.index)
    output = genexpr.run_dataset(gene_model, gene_expression_single.to_numpy(), "Pancancer")
    
    del gene_expression_single
    gc.collect()
    
    gene_outputs = pd.DataFrame(output[0])
    gene_outputs.index = list_indexes
    gene_outputs.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/single_cell/pancancer_with_alpha_outputs.csv',
                                      header=True, index=False)
    
    print(gene_outputs.shape)
    
    del gene_outputs
    gc.collect()
    
    gene_bottlenecks = pd.DataFrame(output[1])
    gene_bottlenecks.index = list_indexes
    
    cell_lines = []
    for barcode in list(gene_bottlenecks.index):
        cell_lines.append(metadata_single.loc[barcode, 'Cell_line'])
    gene_bottlenecks["Cell_line"] = cell_lines
    
    pickle.dump(gene_bottlenecks, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/single_cell/pancancer_with_alpha_bottlenecks.pkl', 'wb'))
    gene_bottlenecks.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/single_cell/pancancer_with_alpha_bottlenecks.csv', header=True, index=False)

    gene_bottlenecks.set_index(list(gene_bottlenecks.columns)[0])
    print("PANCANCER BOTTLENECK \n", gene_bottlenecks.shape)
    
    return gene_bottlenecks, list_indexes

# _ = create_pancancer_bottleneck()