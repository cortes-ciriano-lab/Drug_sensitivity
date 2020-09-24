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
def create_pancancer_bottleneck(path):
    gene_expression_single = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}/pkl_files/pancancer_dataset.pkl'.format(path), 'rb'))
    # gene_expression_single = gene_expression_single.iloc[:100]
    metadata_single = pickle.load(open('/hps/research1/icortes/acunha/data/PANCANCER/pancancer_metadata.pkl', 'rb'))
    metadata_single = metadata_single.loc[metadata_single.index.isin(list(gene_expression_single.index))]
    
    #initialize the single cell model
    print('Single cell model: started \n ')
    genexpr = Genexp_sc()
    filename = '{}/single_cell/genexp_sc_output_with_alpha.txt'.format(path)
    genexpr.set_filename_report(filename)
    gene_model = genexpr.start_expression(num_genes=gene_expression_single.shape[1],
                                          path_model = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/pancancer_with_alpha')

    list_indexes = list(gene_expression_single.index)
    output = genexpr.run_dataset(gene_model, gene_expression_single.to_numpy(), 'Pancancer')
    
    del gene_expression_single
    gc.collect()
    
    gene_outputs = pd.DataFrame(output[0])
    gene_outputs.index = list_indexes
    gene_outputs.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}/single_cell/pancancer_with_alpha_outputs.csv'.format(path), header=True, index=False)
    
    print(gene_outputs.shape)
    
    del gene_outputs
    gc.collect()
    
    gene_bottlenecks = pd.DataFrame(output[1])
    gene_bottlenecks.index = list_indexes
    
    cell_lines = []
    for barcode in list(gene_bottlenecks.index):
        cell_lines.append(metadata_single.loc[barcode, 'Cell_line'])
    gene_bottlenecks['Cell_line'] = cell_lines
    
    pickle.dump(gene_bottlenecks, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}/single_cell/pancancer_with_alpha_bottlenecks.pkl'.format(path), 'wb'))
    gene_bottlenecks.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}/single_cell/pancancer_with_alpha_bottlenecks.csv'.format(path), header=True, index=False)

    gene_bottlenecks.set_index(list(gene_bottlenecks.columns)[0])
    print('PANCANCER BOTTLENECK \n', gene_bottlenecks.shape)
    
    return gene_bottlenecks, list_indexes

# _ = create_pancancer_bottleneck()