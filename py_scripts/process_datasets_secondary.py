# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import gc
import sys
from standardiser import standardise
from sklearn.utils import shuffle

from molecular import check_valid_smiles, Molecular
from featurizer_SMILES import OneHotFeaturizer
from create_mol_bot import create_prism_bottleneck_run_secondary
from create_sc_bot import create_pancancer_bottleneck

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)

# -------------------------------------------------- DEFINE PATHS --------------------------------------------------


path_data = '/hps/research1/icortes/acunha/data/'
# path_data = 'C:/Users/abeat/Dropbox/data'
#
path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/' #cluster
# path_results = 'C:/Users/abeat/Documents/GitHub/Drug_sensitivity/'

# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
class Process_dataset_pancancer():
    
    def __init__(self, **kwargs):
        self.barcodes_per_cell_line = {}
        self.ohf = OneHotFeaturizer()
    
    # --------------------------------------------------
    
    def load_pancancer(self):
        #gene_expresion :: rows: AAACCTGAGACATAAC-1-18 ; columns: RP11-34P13.7
        pancancer_data = pickle.load(open('{}/PANCANCER/pancancer_data.pkl'.format(path_data), 'rb'))
        print('\n Pancancer dataset (after loading)')
        print(pancancer_data.shape)
        
        #metadata :: rows: AAACCTGAGACATAAC-1-18 ; Cell_line: NCIH2126_LUNG (CCLE_name)
        pancancer_metadata= pickle.load(open('{}/PANCANCER/pancancer_metadata.pkl'.format(path_data), 'rb'))
        print('\n Pancancer metadata (after loading)')
        print(pancancer_metadata.shape)
        
        #filter the cell lines - keep only the ones that have at least 500 genes expressed
        valid_rows = pancancer_data[pancancer_data > 0].count(axis = 1) > 500
        pancancer_data = pancancer_data.loc[valid_rows]
        
        pancancer_metadata = pancancer_metadata.loc[pancancer_metadata.index.isin(list(pancancer_data.index))]
        
        print('\n Pancancer dataset (after filtering the cells that have at least 500 genes expressed)')
        print(pancancer_data.shape)
        print('\n Pancancer metadata (after filtering the cells that have at least 500 genes expressed)')
        print(pancancer_metadata.shape)
        
        return pancancer_data, pancancer_metadata
    
    # --------------------------------------------------
        
    def load_prism(self, maximum_length_smiles):
        #rows: ACH-000001 ; columns: BRD-A00077618-236-07-6::2.5::HTS
        prism_matrix = pd.read_csv('{}/Prism_19Q4_secondary/secondary-screen-dose-response-curve-parameters.csv'.format(path_data), header=0, nrows = 200, usecols= ['broad_id', 'depmap_id', 'ccle_name', 'screen_id', 'auc', 'name', 'moa', 'target', 'smiles', 'passed_str_profiling'])
        print('\n PRISM dataset (after loading)')
        print(prism_matrix.shape)
        
        #filter the smiles - drop nan values (1), that has passed_str_profiling TRUE (2), standardise the smiles (3) and check if the standardised smile is compatible with the molecular VAE (4)
        prism_matrix = prism_matrix.loc[prism_matrix['passed_str_profiling']]
        prism_matrix.dropna(subset=['smiles', 'auc'], inplace=True) # (1) - only keep data with smiles and valid auc values
        
        for i in range(len(prism_matrix['smiles'])):
            smile = prism_matrix['smiles'].iloc[i]
            
            if ',' in smile: #means that exists more than one smile representation of the compound
                if '\'' in smile:
                    smile = smile.strip('\'')
                smiles = smile.split(', ')
            else:
                smiles = [smile]
            
            standard_smiles = [] # (2)
            for s in smiles:
                try:
                    mol = standardise.run(s)
                    standard_smiles.append(mol)
                except standardise.StandardiseException:
                    pass
            
            valid_smiles = check_valid_smiles(standard_smiles, maximum_length_smiles) # (3)
            if valid_smiles:
                for s in valid_smiles:
                    if not self.ohf.check_smile(s):
                        valid_smiles.remove(s)
            
            if not valid_smiles:
                prism_matrix['smiles'].iloc[i] = float('NaN')
            else:
                if len(valid_smiles) > 1:
                    prism_matrix['smiles'].iloc[i] = ', '.join(valid_smiles)
                else:
                    prism_matrix['smiles'].iloc[i] = ''.join(valid_smiles)
        
        prism_matrix.dropna(subset=['smiles'], inplace=True) # (1) - because the non valid smiles were transformed into nan
        
        print('\n PRISM dataset (after filtering the valid smiles)')
        print(prism_matrix.shape)
        
        return prism_matrix
    
    # --------------------------------------------------
    
    def filter_cell_lines(self, single_cell, metadata_sc, drug_data):
        
        list_barcodes = list(single_cell.index) #list of barcodes from single cell data
        metadata_sc = metadata_sc.loc[metadata_sc.index.isin(list_barcodes)] #filter the metadata
        
        #extract the different cell lines from the single cell dataset
        list_cell_lines_sc_ccle = list(metadata_sc['Cell_line'].unique()) #list of the different cell lines - ccle id
        
        #extract the different cell lines from the prism dataset - depmap id
        list_cell_lines_drug_ccle = list(drug_data['ccle_name'].unique())
        
        #find the common cell lines from both datasets (prism and single cell)
        list_commun_cell_lines = list(set(list_cell_lines_drug_ccle).intersection(list_cell_lines_sc_ccle))
        
        return list_commun_cell_lines

    # --------------------------------------------------
    
    def create_integrated_datasets(self, screens_list, prism_dataset, prism_bottlenecks, pancancer_bottlenecks, pancancer_metadata):
        barcode2indexes = {}
        new_indexes_dict = {}

        for ccle in pancancer_metadata['Cell_line'].unique():
            barcodes = list(pancancer_metadata.loc[pancancer_metadata['Cell_line'] == ccle].index)
            barcodes = {x : pancancer_bottlenecks.index.get_loc(x) for x in barcodes} #for each barcode returns its index
            indexes = []
            prism_subset = prism_dataset.loc[prism_dataset['ccle_name'] == ccle]
            prism_subset = prism_subset.loc[prism_subset['board_id'].isin(screens_list)]
            for j in range(prism_subset.shape[0]):
                screen = prism_subset['board_id'].iloc[j]
                screen_i = prism_bottlenecks.index.get_loc(screen)
                index = prism_subset.iloc[j].name
                sens_value = prism_subset['auc'].loc[j]
                new_indexes_dict[index] = ((ccle, barcodes), (screen, screen_i), sens_value)
                indexes.append(index)
            barcode2indexes[ccle] = indexes

        pickle.dump(barcode2indexes, open('{}/data_secondary/prism_pancancer/prism_pancancer_new_indexes_dict.pkl'.format(path_results), 'wb'))
        pickle.dump(new_indexes_dict, open('{}/data_secondary/prism_pancancer/prism_pancancer_new_indexes_newIndex2barcodeScreen_dict.pkl'.format(path_results), 'wb'))

    # --------------------------------------------------
    
    def run(self):
        #initialize the molecular model
        molecules = Molecular()
        molecules.set_filename_report('/data_secondary/molecular/run_once/molecular_output2.txt'.format(path_results))
        _ = molecules.start_molecular()
        maximum_length_smiles = int(molecules.get_maximum_length())
        
        prism_matrix = self.load_prism(maximum_length_smiles)
        pancancer_data, pancancer_metadata = self.load_pancancer()
        
        #get commun cell lines in both datasets
        list_commun_cell_lines_ccle = self.filter_cell_lines(pancancer_data, pancancer_metadata, prism_matrix)
        
        #filter datasets
        prism_matrix = prism_matrix.loc[prism_matrix['ccle_name'].isin(list_commun_cell_lines_ccle)]
        pancancer_metadata = pancancer_metadata.loc[pancancer_metadata['Cell_line'].isin(list_commun_cell_lines_ccle)]
        pancancer_data = pancancer_data.loc[pancancer_data.index.isin(list(pancancer_metadata.index))]
        
        for cell in list_commun_cell_lines_ccle:
            self.barcodes_per_cell_line[cell] = list(pancancer_metadata[pancancer_metadata['Cell_line'] == cell].index)

        print('\n PRISM dataset (after filtering the common cell lines)')
        print(prism_matrix.shape)
        print('\n Pancancer dataset (after filtering the common cell lines)')
        print(pancancer_data.shape)
        print('\n Pancancer metadata (after filtering the common cell lines)')
        print(pancancer_metadata.shape)
        
        list_cell_lines_prism = list(prism_matrix['ccle_name'].unique())
        
        with open('{}/data_secondary/prism_pancancer/prism_pancancer_cell_lines_depmap.txt'.format(path_results), 'w') as f:
            f.write('\n'.join(list_cell_lines_prism))
        with open('{}/data_secondary/prism_pancancer/prism_pancancer_cell_lines_pancancer.txt'.format(path_results), 'w') as f:
            f.write('\n'.join(list(pancancer_metadata['Cell_line'].unique())))
        pancancer_tumours = list(pancancer_metadata['Cancer_type'].unique())
        with open('{}/data_secondary/prism_pancancer/prism_pancancer_tumours.txt'.format(path_results), 'w') as f:
            f.write('\n'.join(pancancer_tumours))
        with open('{}/data_secondary/prism_pancancer/prism_pancancer_barcodes_sc.txt'.format(path_results), 'w') as f:
            f.write('\n'.join(list(pancancer_data.index)))

        list_index = []
        for i in range(prism_matrix.shape[0]):
            list_index.append("{}::{}::{}".format(prism_matrix['ccle_name'].iloc[i], prism_matrix['broad_id'].iloc[i], prism_matrix['screen_id'].iloc[i]))
        prism_matrix.index = list_index

        pickle.dump(pancancer_data, open('{}/data_secondary/pkl_files/pancancer_dataset.pkl'.format(path_results), 'wb'), protocol = 4)
        pickle.dump(prism_matrix, open('{}/data_secondary/pkl_files/prism_dataset.pkl'.format(path_results), 'wb'), protocol = 4)
        pickle.dump(pancancer_metadata, open('{}/data_secondary/pkl_files/pancancer_metadata.pkl'.format(path_results), 'wb'))
        prism_matrix.reset_index().to_csv('{}/data_secondary/prism_dataset.csv'.format(path_results), header=True, index=False)

        del pancancer_data
        gc.collect()

        barcodes_per_tumour = {}
        for i in range(len(pancancer_tumours)):
            tumour = pancancer_tumours[i]
            barcodes_per_tumour[tumour] = list(pancancer_metadata[pancancer_metadata['Cancer_type'] == tumour].index)
        
        ccle_per_barcode = {}
        for k,v in self.barcodes_per_cell_line.items():
            for i in v:
                ccle_per_barcode[i] = k
                
        pickle.dump(barcodes_per_tumour, open('{}/data_secondary/prism_pancancer/barcodes_per_tumour_dict.pkl'.format(path_results), 'wb'))
        pickle.dump(self.barcodes_per_cell_line, open('{}/data_secondary/prism_pancancer/barcodes_per_cell_line_dict.pkl'.format(path_results), 'wb'))
        pickle.dump(ccle_per_barcode, open('{}/data_secondary/prism_pancancer/ccle_per_barcode_dict.pkl'.format(path_results), 'wb'))
        
        # create the bottlenecks
        prism_bottlenecks, list_indexes_prism = create_prism_bottleneck_run_secondary()
        with open('{}/data_secondary/prism_pancancer/prism_pancancer_screens.txt'.format(path_results), 'w') as f:
            f.write('\n'.join(list(list_indexes_prism)))
        
        pancancer_bottlenecks, _ = create_pancancer_bottleneck()

        #create the integrate files
        self.create_integrated_datasets(list_indexes_prism, prism_matrix, prism_bottlenecks, pancancer_bottlenecks, pancancer_metadata)
        
        print('DONE!')

# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    if sc_from == 'pancancer':
        process = Process_dataset_pancancer()
        process.run()

except EOFError:
    print('ERROR!')