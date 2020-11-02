# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import gc
import sys
from rdkit import Chem
from standardiser import standardise
from rdkit.Chem import AllChem
from sklearn.utils import shuffle
import torch
import torch.nn as nn

from featurizer_SMILES import OneHotFeaturizer
from full_network import VAE_gene_expression_single_cell, VAE_molecular

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- DEFINE PATHS --------------------------------------------------

path_data = '/hps/research1/icortes/acunha/data/'
# path_data = 'C:/Users/abeat/Dropbox/data'

# -------------------------------------------------- DEFINE FUNCTIONS --------------------------------------------------

def check_valid_smiles(data, maximum_length):
    found_invalid = False
    valid_smiles = []
    for i in range(len(data)):
        m = data[i]
        if len(m) <= maximum_length and m not in valid_smiles:
            valid_smiles.append(m)
        else:
            with open('invalid_smiles.txt', 'a') as f:
                f.write(data[i])
                f.write('\n')

    if found_invalid:
        print('WARNING!! \nSome molecules have invalid lengths and will not be considered. Please check the file invalid_smiles.txt for more information. \n')

    return valid_smiles

# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
class Process_dataset_pancancer():
    
    def __init__(self):
        self.barcodes_per_cell_line = {}
        self.ohf = OneHotFeaturizer()
        self.path_results = None
        self.values_from = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        #from the smilesVAE model
        self.molecular_model = None
        self.alpha_m = None
        self.maximum_length_m = None
        self.batch_m = None
        self.dropout_m = None
        self.ohf = None
        
        #from the scVAE model
        self.run_type = None
        self.sc_model = None
        self.batch_sc = None
        self.dropout_sc = None
        self.layers_sc = None
        self.alpha_sc = None
        self.pathway_sc = None
        self.num_genes_sc = None
        
    # --------------------------------------------------
    
    def load_smilesVAE(self):
        # path = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/molecular/'
        path = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model'
        
        self.alpha_m, _, self.maximum_length_m, self.lr_m, self.batch_m, _, _, _, self.dropout_m, _, _, _, _, _, _ = pickle.load(open('{}/list_initial_parameters_smiles.pkl'.format(path), 'rb'))
        # self.alpha_m, self.maximum_length_m, _, self.batch_m, _, _, _, self.dropout_m, _, _, _, _, _ = pickle.load(open('{}/list_initial_parameters_molecular.pkl'.format(path), 'rb'))
        self.batch_m = int(self.batch_m)
        self.dropout_m = float(self.dropout_m)
        self.maximum_length_m = int(self.maximum_length_m)
        self.alpha_m = float(self.alpha_m)
        self.ohf = OneHotFeaturizer()
        
        self.molecular_model = VAE_molecular(number_channels_in=self.maximum_length_m, length_signal_in=len(self.ohf.get_charset()), dropout_prob = self.dropout_m)
        self.molecular_model.to(self.device)
        
        model_parameters = pickle.load(open('{}/molecular_model.pkl'.format(path), 'rb'))
        self.molecular_model.load_state_dict(model_parameters)
    
    # --------------------------------------------------
    
    def get_smiles_bottlenecks(self, dataset, valid_compounds):
        fingerprints_smiles_dicts = {}
        for k,v in dataset.items():
            if k in valid_compounds:
                fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(v), 2, nBits=1024)
                fingerprints_smiles_dicts[k] = {'Morgan_Fingerprint' : '[{}]'.format(','.join([str(x) for x in fp])),
                                                'Smile' : v}
        
        fingerprints_smiles_dicts = pd.DataFrame.from_dict(fingerprints_smiles_dicts, orient = 'index')
        fingerprints_smiles_dicts.to_csv('{}/molecular/prism_indexes_morgan_fingerprints_smiles.csv'.format(self.path_results), header=True, index=True)
    
        mols = self.ohf.featurize(list(fingerprints_smiles_dicts['Smile']), self.maximum_length_m)
        if True in np.isnan(np.array(mols)):
            print('there are nan in the dataset!! \n ')
            sys.exit()
        
        self.molecular_model.eval()
        indexes = list(fingerprints_smiles_dicts.index)
        bottleneck_complete = []
        predictions_complete = []
        valid = 0
        same = 0
        invalid_id = []
        with torch.no_grad():
            for i in range(0, len(indexes), self.batch_m):
                smiles = list(fingerprints_smiles_dicts['Smile'].iloc[i:i + self.batch_m])
                batch = mols[i:i + self.batch_m]
                inputs = torch.tensor(batch).type('torch.FloatTensor').to(self.device)
                predictions = self.molecular_model(inputs)
                predictions_complete.extend(predictions[0].cpu().numpy().tolist())
                bottleneck_complete.extend(predictions[1].cpu().numpy().tolist())
                output = self.ohf.back_to_smile(predictions[0].cpu().numpy().tolist())
                for i in range(len(output)):
                    m = Chem.MolFromSmiles(output[i])
                    if m is not None:
                        valid += 1
                        if m == smiles[i]:
                            same += 1
                    else:
                        invalid_id.append(i)
                    with open('{}/molecular/run_once/valid_smiles.txt'.format(self.path_results), 'a') as f:
                        f.write('\n'.join(['Input: {}'.format(smiles[i]), 'Output: {}'.format(output[i]), '\n']))
                        f.write('\n')
                    
        mol_outputs = {}
        for i in range(len(predictions_complete)):
            mol_outputs[indexes[i]] = predictions_complete[i]
        pickle.dump(mol_outputs, open('{}/molecular/run_once/prism_outputs.pkl'.format(self.path_results), 'wb'), protocol=4)
        
        del predictions_complete
        del mol_outputs
        gc.collect()
        
        mol_bottlenecks = pd.DataFrame(bottleneck_complete)
        mol_bottlenecks.index = indexes
        pickle.dump(mol_bottlenecks, open('{}/molecular/run_once/prism_bottlenecks.pkl'.format(self.path_results), 'wb'), protocol=4)
        mol_bottlenecks.to_csv('{}/molecular/run_once/prism_bottlenecks.csv'.format(self.path_results), header=True, index=False)
  
        del bottleneck_complete
        gc.collect()
        
        return mol_bottlenecks, indexes
    
    # --------------------------------------------------
    
    def load_scVAE(self):
        path = '/hps/research1/icortes/acunha/python_scripts/single_cell/best_model/pancancer_with_alpha/pancancer_{}/'.format(self.run_type)
        
        combinations = [['all_genes'], ['no_pathway']] #, 'best_7000'], ['no_pathway', 'kegg_pathways', 'canonical_pathways', 'chemical_genetic_perturbations']]
        
        if self.run_type == 'all_genes_no_pathway' or self.run_type == 'all_genes_canonical_pathways':
            _, self.batch_sc, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, self.alpha_sc, _, self.pathway_sc, self.num_genes_sc = pickle.load(open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))
        else:
            _, self.batch_sc, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, self.alpha_sc, _, self.pathway_sc, self.num_genes_sc, _ = pickle.load(open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))
        self.batch_sc = int(self.batch_sc)
        self.dropout_sc = float(self.dropout_sc)
        self.layers_sc = self.layers_sc.split('_')
        self.alpha_sc = float(self.alpha_sc)
        
        if self.pathway != 'no_pathway':
            pathways = {'canonical_pathways' : '/hps/research1/icortes/acunha/data/pathways/canonical_pathways/',
                        'chemical_genetic_perturbations' : '/hps/research1/icortes/acunha/data/pathways/chemical_genetic_perturbations/',
                        'kegg_pathways' : '/hps/research1/icortes/acunha/data/pathways/kegg_pathways'}
            list_pathways = pickle.load(open('{}/list_pathways.pkl'.format(pathways[self.pathway]), 'rb'))
            number_pathways = len(list_pathways)
            path_matrix_file = '/hps/research1/icortes/acunha/python_scripts/single_cell/data/pathway_matrices/pancancer_matrix_{}_{}_only_values.csv'.format(self.num_genes, self.pathway)
        else:
            number_pathways = 0
            path_matrix_file = ''
        
        self.sc_model = VAE_gene_expression_single_cell(dropout_prob=self.dropout_sc, n_genes=self.num_genes_sc, layers=self.layers_sc)
        self.sc_model.to(self.device)
        
        model_parameters = pickle.load(open('{}/single_cell_model.pkl'.format(path), 'rb'))
        self.sc_model.load_state_dict(model_parameters)
    
    # --------------------------------------------------
    
    def get_sc_bottlenecks(self, metadata, indexes, genes, pathways):
        dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_data_{}_{}.csv'.format(genes, pathways), header = 0, index_col = 0)
        dataset = dataset.loc[dataset.index.isin(indexes)]
        
        self.sc_model.eval()
        bottleneck_complete = []
        with torch.no_grad():
            for i in range(0, len(indexes), self.batch_sc):
                list_indexes = indexes[i:i + self.batch_sc]
                batch = dataset.iloc[i:i + self.batch_sc]
                inputs = torch.tensor(batch.to_numpy()).type('torch.FloatTensor').to(self.device)
                predictions = self.sc_model(inputs)
                output = predictions[0].cpu().numpy().tolist()
                bottleneck_complete.extend(predictions[1].cpu().numpy().tolist())
                for j in range(len(output)):
                    output[j] = '{},{}\n'.fomat(list_indexes[j], ','.join([str(x) for x in output[j]]))
                with open('{}/single_cell/pancancer_{}_outputs.csv'.format(self.path_results, self.run_type), 'a') as f:
                    f.write('\n'.join(output))
                    
        bottleneck_complete = pd.DataFrame(bottleneck_complete)
        bottleneck_complete.index = indexes
        cell_lines = []
        for barcode in indexes:
            cell_lines.append(sc_metadata.loc[barcode, 'Cell_line'])
        bottleneck_complete['Cell_line'] = cell_lines
        bottleneck_complete.to_csv('{}/single_cell/pancancer_{}_bottlenecks.csv'.format(self.path_results, self.run_type), header=True, index=True)
        pickle.dump(bottleneck_complete, open('{}/single_cell/pancancer_{}_bottlenecks.pkl'.format(self.path_results, self.run_type), 'wb'))

        return bottleneck_complete
    
    # --------------------------------------------------
    
    def load_pancancer(self):
        #gene_expresion :: rows: AAACCTGAGACATAAC-1-18 ; columns: RP11-34P13.7
        pancancer_data = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_data_all_genes_kegg_pathways.csv', header = 0, index_col = 0)
        print('\n Pancancer: number of barcodes (after loading)')
        print(pancancer_data.shape[0])
        
        list_single_cells = list(pancancer_data.index)
        
        del pancancer_data
        gc.collect()
        
        #metadata :: rows: AAACCTGAGACATAAC-1-18 ; Cell_line: NCIH2126_LUNG (CCLE_name)
        pancancer_metadata = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_metadata.csv', header = 0, index_col = 0)
        print('\n Pancancer metadata (after loading)')
        print(pancancer_metadata.shape)
        
        pancancer_metadata = pancancer_metadata.loc[pancancer_metadata.index.isin(list_single_cells)]
        return list_single_cells, pancancer_metadata
    
    # --------------------------------------------------
        
    def load_prism(self):
        #rows: ACH-000001 ; columns: BRD-A00077618-236-07-6::2.5::HTS
        prism_matrix = pd.read_csv('{}/Prism_19Q4_secondary/secondary-screen-dose-response-curve-parameters.csv'.format(path_data), header=0, usecols= ['broad_id', 'depmap_id', 'ccle_name', 'screen_id', 'auc', 'ic50', 'name', 'moa', 'target', 'smiles', 'passed_str_profiling'])
        print('\n PRISM dataset (after loading)')
        print(prism_matrix.shape)
        
        #filter the smiles - drop nan values (1), that has passed_str_profiling TRUE (2), standardise the smiles (3) and check if the standardised smile is compatible with the molecular VAE (4)
        prism_matrix = prism_matrix.loc[prism_matrix['passed_str_profiling']]
        prism_matrix.dropna(subset=['smiles', self.values_from], inplace=True) # (1) - only keep data with smiles and valid auc/ic50 values
        
        new_matrix = {}
        to_keep = []
        subset = prism_matrix.drop_duplicates(subset=['broad_id'])
        for i in range(len(subset['smiles'])):
            smile = subset['smiles'].iloc[i]
            
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
            
            valid_smiles = check_valid_smiles(standard_smiles, self.maximum_length_m) # (3)
            if valid_smiles:
                if len(valid_smiles) > 1:
                    to_keep.append(subset['broad_id'].iloc[i])
                    for j in range(len(valid_smiles)):
                        new_matrix['{}:::{}'.format(subset['broad_id'].iloc[i], j)] = valid_smiles[j]
                elif len(valid_smiles) == 1:
                    new_matrix[subset['broad_id'].iloc[i]] = valid_smiles[0]
                    to_keep.append(subset['broad_id'].iloc[i])
        
        del subset
        gc.collect()

        prism_matrix = prism_matrix.loc[prism_matrix['broad_id'].isin(to_keep)]

        print('\n PRISM dataset (after filtering the valid smiles)')
        print(prism_matrix.shape)
        
        return prism_matrix, new_matrix
    
    # --------------------------------------------------
    
    def filter_cell_lines(self, list_barcodes, metadata_sc, drug_data):
        metadata_sc = metadata_sc.loc[metadata_sc.index.isin(list_barcodes)] #filter the metadata
        
        #extract the different cell lines from the single cell dataset
        list_cell_lines_sc_ccle = list(metadata_sc['Cell_line'].unique()) #list of the different cell lines - ccle id
        
        #extract the different cell lines from the prism dataset - depmap id
        list_cell_lines_drug_ccle = list(drug_data['ccle_name'].unique())
        
        #find the common cell lines from both datasets (prism and single cell)
        list_commun_cell_lines = list(set(list_cell_lines_drug_ccle).intersection(list_cell_lines_sc_ccle))
        
        return list_commun_cell_lines[:1]

    # --------------------------------------------------
    
    def create_integrated_datasets(self, screens_list, prism_dataset, prism_bottlenecks, pancancer_bottlenecks, pancancer_metadata):
        barcode2indexes = {}
        new_indexes_dict = {}
        total = 0
        screens_prism2bottlenecks = {}
        for full_index in screens_list:
            screen = full_index.split(':::')[0]
            if screen not in screens_prism2bottlenecks.keys():
                screens_prism2bottlenecks[screen] = []
            screens_prism2bottlenecks[screen].append(full_index)

        for ccle in pancancer_metadata['Cell_line'].unique():
            barcodes = list(pancancer_metadata.loc[pancancer_metadata['Cell_line'] == ccle].index)
            barcodes = {x : pancancer_bottlenecks.index.get_loc(x) for x in barcodes} #for each barcode returns its index
            indexes = []
            prism_subset = prism_dataset.loc[prism_dataset['ccle_name'] == ccle]
            prism_subset = prism_subset.loc[prism_subset['broad_id'].isin(list(screens_prism2bottlenecks.keys()))]
            for j in range(prism_subset.shape[0]):
                screen = prism_subset.iloc[j].name
                compound_id = prism_subset['broad_id'].iloc[j]
                for full_index in screens_prism2bottlenecks[compound_id]:
                    screen_i = prism_bottlenecks.index.get_loc(full_index)
                    sens_value = prism_subset[self.values_from].iloc[j]
                    new_index = screen.split('::')
                    new_index[1] = full_index
                    new_index = '::'.join(new_index)
                    if self.values_from == 'ic50':
                        if sens_value >= 0.000610352 and sens_value <= 10:
                            sens_value = -np.log10(sens_value * 10**(-6))
                            new_indexes_dict[new_index] = ((ccle, barcodes), (compound_id, screen_i), sens_value)
                            indexes.append(new_index)
                            total += 1
                    else:
                        new_indexes_dict[new_index] = ((ccle, barcodes), (compound_id, screen_i), sens_value)
                        indexes.append(new_index)
                        total += 1
            barcode2indexes[ccle] = indexes
        print('Total indexes: {}'.format(total))
 
        pickle.dump(barcode2indexes, open('{}/prism_pancancer/prism_pancancer_new_indexes_dict.pkl'.format(self.path_results), 'wb'))
        pickle.dump(new_indexes_dict, open('{}/prism_pancancer/prism_pancancer_new_indexes_newIndex2barcodeScreen_dict.pkl'.format(self.path_results), 'wb'))

    # --------------------------------------------------
    
    def define_path_results(self, values_from):
        self.values_from = values_from
        self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary2/{}'.format(values_from) #cluster
        # self.path_results = 'C:/Users/abeat/Documents/GitHub/Drug_sensitivity/data_secondary/{}'.format(values_from)
    
    # --------------------------------------------------
    
    def define_sc_path(self, combination):
        self.run_type = combination
    
    # --------------------------------------------------
    
    def run(self, values_from):
        self.define_path_results(values_from)
        
        #initialize the molecular model
        self.load_smilesVAE()
        
        #load prism dataset
        prism_matrix, only_compounds = self.load_prism()
        
        #load pancancer dataset
        list_single_cells, pancancer_metadata = self.load_pancancer()
        
        #get commun cell lines in both datasets
        list_commun_cell_lines_ccle = self.filter_cell_lines(list_single_cells, pancancer_metadata, prism_matrix)
        
        #filter datasets
        prism_matrix = prism_matrix.loc[prism_matrix['ccle_name'].isin(list_commun_cell_lines_ccle)].iloc[:10]
        pancancer_metadata = pancancer_metadata.loc[pancancer_metadata['Cell_line'].isin(list_commun_cell_lines_ccle)].iloc[:10]
        list_single_cells = list(pancancer_metadata.index)
        
        for cell in list_commun_cell_lines_ccle:
            self.barcodes_per_cell_line[cell] = list(pancancer_metadata[pancancer_metadata['Cell_line'] == cell].index)

        print('\n PRISM dataset (after filtering the common cell lines)')
        print(prism_matrix.shape)
        print('\n Pancancer: number of barcodes (after filtering the common cell lines)')
        print(len(list_single_cells))
        print('\n Pancancer metadata (after filtering the common cell lines)')
        print(pancancer_metadata.shape)
        
        list_cell_lines_prism = list(prism_matrix['depmap_id'].unique())
        list_tumours = list(pancancer_metadata['Cancer_type'].unique())
        
        with open('{}/prism_pancancer/prism_pancancer_cell_lines_depmap.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list_cell_lines_prism))
        with open('{}/prism_pancancer/prism_pancancer_cell_lines_pancancer.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list_commun_cell_lines_ccle))
        with open('{}/prism_pancancer/prism_pancancer_tumours.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list_tumours))
        with open('{}/prism_pancancer/prism_pancancer_barcodes_sc.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list_single_cells))

        list_index = []
        for i in range(prism_matrix.shape[0]):
            list_index.append('{}::{}::{}'.format(prism_matrix['ccle_name'].iloc[i], prism_matrix['broad_id'].iloc[i], prism_matrix['screen_id'].iloc[i]))
        prism_matrix.index = list_index

        pickle.dump(prism_matrix, open('{}/pkl_files/prism_dataset.pkl'.format(self.path_results), 'wb'), protocol = 4)
        pickle.dump(pancancer_metadata, open('{}/pkl_files/pancancer_metadata.pkl'.format(self.path_results), 'wb'))
        prism_matrix.to_csv('{}/prism_dataset.csv'.format(self.path_results), header=True, index=True)

        barcodes_per_tumour = {}
        for i in range(len(list_tumours)):
            tumour = list_tumours[i]
            barcodes_per_tumour[tumour] = list(pancancer_metadata[pancancer_metadata['Cancer_type'] == tumour].index)
        
        pickle.dump(barcodes_per_tumour, open('{}/prism_pancancer/barcodes_per_tumour_dict.pkl'.format(self.path_results), 'wb'))
        
        del barcodes_per_tumour
        
        ccle_per_barcode = {}
        for k,v in self.barcodes_per_cell_line.items():
            for i in v:
                ccle_per_barcode[i] = k
                
        pickle.dump(self.barcodes_per_cell_line, open('{}/prism_pancancer/barcodes_per_cell_line_dict.pkl'.format(self.path_results), 'wb'))
        pickle.dump(ccle_per_barcode, open('{}/prism_pancancer/ccle_per_barcode_dict.pkl'.format(self.path_results), 'wb'))
        
        del ccle_per_barcode
        
        #create the bottlenecks - prism
        list_valid_compounds = list(prism_matrix['broad_id'].unique())
        prism_bottlenecks, list_indexes_prism = self.get_smiles_bottlenecks(only_compounds, list_valid_compounds)
        with open('{}/prism_pancancer/prism_pancancer_screens.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list(list_indexes_prism)))
            
        del only_compounds
        gc.collect()
        
        combinations = [['all_genes'], ['no_pathway']] #, 'best_7000'], ['no_pathway', 'kegg_pathways', 'canonical_pathways', 'chemical_genetic_perturbations']]
        for i in range(len(combinations[0])):
            for j in range(len(combinations[1])):
                combination = '{}_{}'.format(combinations[0][i], combinations[1][j])
                self.define_sc_path(combination)
                
                #initialize the molecular model
                self.load_scVAE()
                
                #create the bottlenecks - pancancer
                pancancer_bottlenecks = self.get_sc_bottlenecks(pancancer_metadata, list_single_cells, combinations[0][i], combinations[1][j])

        #create the integrate files
        self.create_integrated_datasets(list_indexes_prism, prism_matrix, prism_bottlenecks, pancancer_bottlenecks, pancancer_metadata)
        
        print('DONE!')

# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    values_from = sys.argv[2]
    if sc_from == 'pancancer':
        process = Process_dataset_pancancer()
        process.run(values_from)

except EOFError:
    print('ERROR!')