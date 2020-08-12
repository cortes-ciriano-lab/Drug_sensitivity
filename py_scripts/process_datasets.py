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
from create_mol_bot import create_prism_bottleneck_run_once, create_prism_bottleneck_only_valids
from create_sc_bot import create_pancancer_bottleneck

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)

# -------------------------------------------------- DEFINE PATHS --------------------------------------------------


path_data = '/hps/research1/icortes/acunha/data/'
# path_data = 'C:/Users/abeat/Dropbox/data'
#
path_results = '/hps/research1/icortes/acunha/python_scripts/single_cell/' #cluster
# path_results = 'C:/Users/abeat/Documents/GitHub/Drug_sensitivity/'

# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
class Process_dataset_pancancer():
    
    def __init__(self, **kwargs):
        self.ccle2depmap = {}
        self.depmap2ccle = {}
        self.barcodes_per_cell_line = {}
        self.ohf = OneHotFeaturizer()
        self.final_ccle_lines = []
        self.final_dep_map_lines = []
        self.final_barcodes = []
        self.prism_screen = kwargs['prism_screen']
    
    # --------------------------------------------------

    def __create_ccle2depmap_and_depmap2ccle(self):
        cell_info = pd.read_csv('/hps/research1/icortes/acunha/data/CCLE/sample_info.csv', index_col=0, header=0)
        for i in range(cell_info.shape[0]):
            self.ccle2depmap[cell_info['CCLE_Name'].iloc[i]] = cell_info.iloc[i, :].name #converter ccle to depmap
            self.depmap2ccle[cell_info.iloc[i, :].name] = cell_info['CCLE_Name'].iloc[i] #converter depmap to ccle
        
        del cell_info
        gc.collect()
    
    # --------------------------------------------------
    
    def load_pancancer(self):
        #gene_expresion :: rows: AAACCTGAGACATAAC-1-18 ; columns: RP11-34P13.7
        pancancer_data = pickle.load(open('/hps/research1/icortes/acunha/data/PANCANCER/pancancer_data.pkl', 'rb'))
        print('\n Pancancer dataset (after loading)')
        print(pancancer_data.shape)
        
        #metadata :: rows: AAACCTGAGACATAAC-1-18 ; Cell_line: NCIH2126_LUNG (CCLE_name)
        pancancer_metadata= pickle.load(open('/hps/research1/icortes/acunha/data/PANCANCER/pancancer_metadata.pkl', 'rb'))
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
        if self.prism_screen == 'primary':
            #rows: ACH-000001 ; columns: BRD-A00077618-236-07-6::2.5::HTS
            prism_matrix = pd.read_csv('/hps/research1/icortes/acunha/data/Drug_Sensitivity_PRISM/primary-screen-replicate-collapsed-logfold-change.csv', sep=',', header=0,
                                        index_col=0)
            # rows: BRD-A00055058-001-01-0::2.325889319::MTS004
            prism_metadata = pd.read_csv(
                '/hps/research1/icortes/acunha/data/Drug_Sensitivity_PRISM/primary-screen-replicate-collapsed-treatment-info.csv',
                sep=',', header=0,
                index_col=0, usecols=['column_name', 'smiles', 'broad_id', 'name', 'dose', 'screen_id', 'moa', 'target'])
        else:
            pass

        print('\n PRISM dataset (after loading)')
        print(prism_matrix.shape)
        
        # drug_sensitivity_matrix = drug_sensitivity_matrix.iloc[:, :100]

        print('\n PRISM metadata (after loading)')
        print(prism_metadata.shape)
        
        #filter the smiles - drop nan values (1), standardise the smiles (2) and check if the standardised smile is compatible with the molecular VAE (3)
        prism_metadata.dropna(subset=['smiles'], inplace=True) # (1)
        
        for i in range(len(prism_metadata['smiles'])):
            smile = prism_metadata['smiles'].iloc[i]
            
            if ',' in smile: #means that exists more than one smile representation of the compound
                if '\'' in smile:
                    smile = smile.strip('\'')
                smiles = smile.split(', ')
            else:
                smiles = [smile]
            
            standard_smiles = [] # (2)
            for s in list(smiles):
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
                prism_metadata['smiles'].iloc[i] = float('NaN')
            else:
                if len(valid_smiles) > 1:
                    prism_metadata['smiles'].iloc[i] = ', '.join(valid_smiles)
                else:
                    prism_metadata['smiles'].iloc[i] = ''.join(valid_smiles)
        
        prism_metadata.dropna(subset=['smiles'], inplace=True) # (1) - because the non valid smiles were transformed into nan
        
        prism_matrix = prism_matrix[np.intersect1d(prism_matrix.columns, list(prism_metadata.index))]
        
        print('\n PRISM dataset (after filtering the valid smiles)')
        print(prism_matrix.shape)
        print('\n PRISM metadata (after filtering the valid smiles)')
        print(prism_metadata.shape)
        
        return prism_matrix, prism_metadata
    
    # --------------------------------------------------
    
    def filter_cell_lines(self, single_cell, metadata_sc, drug_data, drug_metadata):
        
        list_barcodes = list(single_cell.index) #list of barcodes from single cell data
        metadata_sc = metadata_sc.loc[metadata_sc.index.isin(list_barcodes)] #filter the metadata
        
        #extract the different cell lines from the single cell dataset
        list_cell_lines_sc_ccle = list(metadata_sc['Cell_line'].unique()) #list of the different cell lines - ccle id
        list_cell_lines_sc_depmap = [] #list of the different cell lines - depmap id
        for i in list_cell_lines_sc_ccle:
            if i in self.ccle2depmap.keys():
                list_cell_lines_sc_depmap.append(self.ccle2depmap[i])
        
        #extract the different cell lines from the prism dataset - depmap id
        list_cell_lines_drug = list(drug_data.index)
        
        #find the common cell lines from both datasets (prism and single cell)
        list_commun_cell_lines_depmap = list(set(list_cell_lines_drug).intersection(list_cell_lines_sc_depmap))
        list_commun_cell_lines_ccle = []
        for i in list_commun_cell_lines_depmap:
            if i in self.depmap2ccle.keys():
                list_commun_cell_lines_ccle.append(self.depmap2ccle[i])
        
        free_memory = [list_cell_lines_drug, list_cell_lines_sc_depmap, list_cell_lines_sc_ccle, list_barcodes]
        for item in free_memory:
            del item
        gc.collect()
        
        return list_commun_cell_lines_ccle, list_commun_cell_lines_depmap

    # --------------------------------------------------
    
    def create_integrated_datasets(self, screens_list, prism_dataset, prism_bottlenecks, pancancer_bottlenecks, depmap_per_barcode, what_type):
        '''
        This function will create different subpartions (per cell line) of the final dataset.
        '''

        columns_dataset = []
        n_feat_pancancer = int(pancancer_bottlenecks.shape[1]-1)
        columns_dataset.extend(['feature_sc_{}'.format(x+1) for x in range(n_feat_pancancer)])
        n_feat_prism = int(prism_bottlenecks.shape[1])
        columns_dataset.extend(['feature_drug_{}'.format(x + 1) for x in range(n_feat_prism)])
        columns_dataset.extend(['sensitivity_value'])

        barcodes_to_use = []

        for k,v in self.barcodes_per_cell_line.items():
            barcodes = shuffle(v)
            barcodes_to_use.extend(barcodes[:5])

        barcode2indexes = {}
        new_indexes_dict = {}

        for i in range(pancancer_bottlenecks.shape[0]):
            bar = pancancer_bottlenecks.iloc[i].name
            if bar in barcodes_to_use:
                indexes = []
                depmap_id = depmap_per_barcode[bar]
                for j in range(prism_bottlenecks.shape[0]):
                    screen = prism_bottlenecks.iloc[i].name
                    sens_value = prism_dataset.loc[depmap_id, screen.split(':::')[0]]
                    if not np.isnan(sens_value):
                        new_index = '{}::{}'.format(bar, screen)
                        new_indexes_dict[new_index] = ((bar, i), (screen, j), sens_value)
                        indexes.append(new_index)
                barcode2indexes[bar] = indexes

        pickle.dump(barcode2indexes, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_new_indexes_5percell_line_once_dict.pkl', 'wb'))
        pickle.dump(new_indexes_dict, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_new_indexes_5percell_line_once_newIndex2barcodeScreen_dict.pkl', 'wb'))

    # --------------------------------------------------
    
    def run(self):
        #initialize the molecular model
        molecules = Molecular()
        molecules.set_filename_report('data/molecular/run_once/molecular_output2.txt')
        _ = molecules.start_molecular()
        maximum_length_smiles = int(molecules.get_maximum_length())
        
        self.__create_ccle2depmap_and_depmap2ccle()
        prism_matrix, prism_metadata = self.load_prism(maximum_length_smiles)
        pancancer_data, pancancer_metadata = self.load_pancancer()
        
        #get commun cell lines in both datasets
        list_commun_cell_lines_ccle, list_commun_cell_lines_depmap = self.filter_cell_lines(pancancer_data, pancancer_metadata,
                                                                                            prism_matrix, prism_metadata)
        
        #filter datasets
        prism_matrix = prism_matrix.loc[prism_matrix.index.isin(list_commun_cell_lines_depmap)]
        pancancer_metadata = pancancer_metadata.loc[pancancer_metadata['Cell_line'].isin(list_commun_cell_lines_ccle)]
        pancancer_data = pancancer_data.loc[pancancer_data.index.isin(list(pancancer_metadata.index))]
        
        for cell in list_commun_cell_lines_ccle:
            self.barcodes_per_cell_line[cell] = list(pancancer_metadata[pancancer_metadata['Cell_line'] == cell].index)

        print('\n PRISM dataset (after filtering the common cell lines)')
        print(prism_matrix.shape)
        print('\n PRISM metadata (after filtering the common cell lines)')
        print(prism_metadata.shape)
        print('\n Pancancer dataset (after filtering the common cell lines)')
        print(pancancer_data.shape)
        print('\n Pancancer metadata (after filtering the common cell lines)')
        print(pancancer_metadata.shape)
        
        list_cell_lines_prism = list(prism_matrix.index)
        
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_cell_lines_depmap.txt', 'w') as f:
            f.write('\n'.join(list_cell_lines_prism))
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_cell_lines_pancancer.txt', 'w') as f:
            f.write('\n'.join(list(pancancer_metadata['Cell_line'].unique())))
        pancancer_tumours = list(pancancer_metadata['Cancer_type'].unique())
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_tumours.txt', 'w') as f:
            f.write('\n'.join(pancancer_tumours))
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_barcodes_sc.txt', 'w') as f:
            f.write('\n'.join(list(pancancer_data.index)))
        
        pickle.dump(prism_metadata, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/prism_metadata.pkl', 'wb'))
        pickle.dump(pancancer_data, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/pancancer_dataset.pkl', 'wb'), protocol = 4)
        pickle.dump(prism_matrix, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/prism_dataset.pkl', 'wb'), protocol = 4)
        pickle.dump(pancancer_metadata, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/pancancer_metadata.pkl', 'wb'))
        prism_matrix.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_dataset.csv', header=True, index=False)

        free_memory = [prism_metadata, pancancer_data]
        for item in free_memory:
            del item
        gc.collect()

        barcodes_per_tumour = {}
        for i in range(len(pancancer_tumours)):
            tumour = pancancer_tumours[i]
            barcodes_per_tumour[tumour] = list(pancancer_metadata[pancancer_metadata['Cancer_type'] == tumour].index)
        
        ccle_per_barcode = {}
        dep_map_per_barcode = {}
        for k,v in self.barcodes_per_cell_line.items():
            for i in v:
                ccle_per_barcode[i] = k
                dep_map_per_barcode[i] = self.ccle2depmap[k]
                
        pickle.dump(barcodes_per_tumour, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/barcodes_per_tumour_dict.pkl', 'wb'))
        pickle.dump(self.ccle2depmap, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/ccle2depmap_dict.pkl', 'wb'))
        pickle.dump(self.depmap2ccle, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/depmap2ccle_dict.pkl', 'wb'))
        pickle.dump(self.barcodes_per_cell_line, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/barcodes_per_cell_line_dict.pkl', 'wb'))
        pickle.dump(ccle_per_barcode, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/ccle_per_barcode_dict.pkl', 'wb'))
        pickle.dump(dep_map_per_barcode, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/dep_map_per_barcode.pkl', 'wb'))
        
        # just once
        prism_bottlenecks, list_indexes_prism = create_prism_bottleneck_run_once()
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_screens_once.txt', 'w') as f:
            f.write('\n'.join(list(list_indexes_prism)))
        
        pancancer_bottlenecks, _ = create_pancancer_bottleneck()
        #create the integrate files
        self.create_integrated_datasets(list_indexes_prism, prism_matrix, prism_bottlenecks, pancancer_bottlenecks, dep_map_per_barcode, 'once')
        
        # del list_indexes_prism
        # gc.collect()
        
        #500 iterations
        # list_indexes_prism = create_prism_bottleneck_only_valids(500)
        # with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_screens_500iterations.txt', 'w') as f:
        #     f.write('\n'.join(list(list_indexes_prism)))
        # self.create_integrated_datasets(list_cell_lines_prism, list_indexes_prism, prism_matrix, '500iterations')
        
        #10000 iterations
        # list_indexes_prism = create_prism_bottleneck_only_valids(10000)
        # with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_screens_10000iterations.txt', 'w') as f:
        #     f.write('\n'.join(list(list_indexes_prism)))
        # self.create_integrated_datasets(list_cell_lines_prism, list_indexes_prism, prism_matrix, '10000iterations')
        
        print('DONE!')
# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    # drug_from = sys.argv[2]
    drug_from = 'primary'
    if sc_from == 'pancancer':
        process = Process_dataset_pancancer(prism_screen = drug_from)
        process.run()

except EOFError:
    print('ERROR!')