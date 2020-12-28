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
        self.maximum_length_m = None
        self.dropout_m = None
        self.ohf = None
        
        #from the scVAE model
        self.run_type = None
        self.sc_model = None
        self.dropout_sc = None
        self.layers_sc = None
        self.pathway_sc = None
        self.num_genes_sc = None
        
    # --------------------------------------------------
    
    def load_smilesVAE(self):
        # path = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/molecular/'
        path = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model'
        
        _, _, self.maximum_length_m, _, _, _, _, _, self.dropout_m, _, _, _, _, _, _ = pickle.load(open('{}/list_initial_parameters_smiles.pkl'.format(path), 'rb'))
        # self.alpha_m, self.maximum_length_m, _, self.batch_m, _, _, _, self.dropout_m, _, _, _, _, _ = pickle.load(open('{}/list_initial_parameters_molecular.pkl'.format(path), 'rb'))
        self.dropout_m = float(self.dropout_m)
        self.maximum_length_m = int(self.maximum_length_m)
        self.ohf = OneHotFeaturizer()
        
        self.molecular_model = VAE_molecular(number_channels_in=self.maximum_length_m, length_signal_in=len(self.ohf.get_charset()), dropout_prob = self.dropout_m)
        self.molecular_model.load_state_dict(torch.load('{}/molecular_model.pt'.format(path), map_location=self.device))
        self.molecular_model.to(self.device)
    
    # --------------------------------------------------
    
    def get_smiles_bottlenecks(self, dataset):
        fingerprints_smiles_dicts = {}
        for i in range(dataset.shape[0]):
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(dataset['smile'].iloc[i]), 2, nBits=1024)
            fingerprints_smiles_dicts[dataset.iloc[i].name] ={'Morgan_Fingerprint' : '[{}]'.format(','.join([str(x) for x in fp]))}

        pd.DataFrame.from_dict(fingerprints_smiles_dicts, orient = 'index').to_csv('{}/molecular/prism_indexes_morgan_fingerprints.csv'.format(self.path_results), header=True, index=True)
        del fingerprints_smiles_dicts
        
        mols = self.ohf.featurize(list(dataset['smile']), self.maximum_length_m)
        if True in np.isnan(np.array(mols)):
            print('there are nan in the dataset!! \n ')
            sys.exit()
        
        self.molecular_model.eval()
        bottleneck_complete = []
        predictions_complete = []
        valid = 0
        same = 0
        invalid_id = []
        with torch.no_grad():
            for i in range(0, dataset.shape[0], 128):
                batch = mols[i:i + 128]
                inputs = torch.tensor(batch).type('torch.FloatTensor').to(self.device)
                predictions = self.molecular_model(inputs)
                bottleneck_complete.extend(predictions[1].cpu().numpy().tolist())
                output = self.ohf.back_to_smile(predictions[0].cpu().numpy().tolist())
                predictions_complete.extend(output)
        
        with open('{}/molecular/run_once/prism_outputs_total.csv'.format(self.path_results), 'w') as f:
            f.write('Index,Input,Output')
            for i in range(dataset.shape[0]):
                m = Chem.MolFromSmiles(predictions_complete[i])
                if m is not None:
                    valid += 1
                    if m == dataset['smile'].iloc[i]:
                        same += 1
                else:
                    invalid_id.append(i)
                f.write('\n{},{},{}'.format(dataset.iloc[i].name, dataset['smile'].iloc[i], predictions_complete[i]))
        
        del predictions_complete
        
        mol_bottlenecks = pd.DataFrame(bottleneck_complete)
        mol_bottlenecks.index = list(dataset.index)
        pickle.dump(mol_bottlenecks, open('{}/molecular/run_once/prism_bottlenecks.pkl'.format(self.path_results), 'wb'))
        mol_bottlenecks.to_csv('{}/molecular/run_once/prism_bottlenecks.csv'.format(self.path_results), header=True, index=True)
  
        del bottleneck_complete
        del mol_bottlenecks
        gc.collect()
    
    # --------------------------------------------------
    
    def load_scVAE(self, num_genes):
        path = '/hps/research1/icortes/acunha/python_scripts/single_cell/best_model/pancancer_{}'.format(self.run_type)
        
        if self.run_type == 'all_genes_no_pathway' or self.run_type == 'all_genes_canonical_pathways':
            _, _, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, _, _, self.pathway_sc, self.num_genes_sc = pickle.load(open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))
        else:
            _, _, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, _, _, self.pathway_sc, self.num_genes_sc, _ = pickle.load(open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))
        self.dropout_sc = float(self.dropout_sc)
        self.layers_sc = self.layers_sc.split('_')
        
        if self.pathway_sc != 'no_pathway':
            pathways = {'canonical_pathways' : '/hps/research1/icortes/acunha/data/pathways/canonical_pathways/',
                        'chemical_genetic_perturbations' : '/hps/research1/icortes/acunha/data/pathways/chemical_genetic_perturbations/',
                        'kegg_pathways' : '/hps/research1/icortes/acunha/data/pathways/kegg_pathways'}
            list_pathways = pickle.load(open('{}/list_pathways.pkl'.format(pathways[self.pathway_sc]), 'rb'))
            number_pathways = len(list_pathways)
            path_matrix_file = '/hps/research1/icortes/acunha/python_scripts/single_cell/data/pathway_matrices/pancancer_matrix_{}_{}_only_values.csv'.format(self.num_genes_sc, self.pathway_sc)
        else:
            number_pathways = 0
            path_matrix_file = ''
        
        self.sc_model = VAE_gene_expression_single_cell(dropout_prob=self.dropout_sc, n_genes=num_genes, layers=self.layers_sc, n_pathways = number_pathways, path_matrix = path_matrix_file)
        self.sc_model.load_state_dict(torch.load('{}/single_cell_model.pt'.format(path), map_location=self.device))
        self.sc_model.to(self.device)
   
    # --------------------------------------------------
    
    def get_sc_bottlenecks(self, dataset, metadata, indexes):
        self.sc_model.eval()
        bottleneck_complete = []
        with open('{}/single_cell/pancancer_{}_outputs.csv'.format(self.path_results, self.run_type), 'w') as f_o:
            with open('{}/single_cell/pancancer_{}_pathways.csv'.format(self.path_results, self.run_type), 'w') as f_p:
                with torch.no_grad():
                    for i in range(0, len(indexes), 128):
                        list_indexes = indexes[i:i + 128]
                        batch = dataset.iloc[i:i + 128]
                        inputs = torch.tensor(batch.to_numpy()).type('torch.FloatTensor').to(self.device)
                        predictions = self.sc_model(inputs)
                        output = predictions[0].cpu().numpy().tolist()
                        bottleneck_complete.extend(predictions[1].cpu().numpy().tolist())
                        pathway = predictions[-1].cpu().numpy().tolist()
                        for j in range(len(output)):
                            output[j] = '{},{}\n'.format(list_indexes[j], ','.join([str(x) for x in output[j]]))
                        f_o.write('\n'.join(output))
                        for j in range(len(pathway)):
                            pathway[j] = '{},{}\n'.format(list_indexes[j], ','.join([str(x) for x in pathway[j]]))
                        f_p.write('\n'.join(pathway))
                    
        bottleneck_complete = pd.DataFrame(bottleneck_complete)
        bottleneck_complete.index = indexes
        cell_lines = []
        for barcode in indexes:
            cell_lines.append(metadata.loc[barcode, 'Cell_line'])
        bottleneck_complete['Cell_line'] = cell_lines
        bottleneck_complete.to_csv('{}/single_cell/pancancer_{}_bottlenecks.csv'.format(self.path_results, self.run_type), header=True, index=True)
        pickle.dump(bottleneck_complete, open('{}/single_cell/pancancer_{}_bottlenecks.pkl'.format(self.path_results, self.run_type), 'wb'))
        
        del bottleneck_complete
        del output
        del pathway
        gc.collect
    
    # --------------------------------------------------
    
    def load_pancancer(self):
        #metadata :: rows: AAACCTGAGACATAAC-1-18 ; Cell_line: NCIH2126_LUNG (CCLE_name)
        pancancer_metadata = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_metadata.csv', header = 0, index_col = 0)
        print('\n Pancancer metadata (after loading)')
        print(pancancer_metadata.shape)

        return pancancer_metadata
    
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
        subset_prism = prism_matrix.drop_duplicates(subset=['broad_id'])
        for i in range(subset_prism.shape[0]):
            smile = subset_prism['smiles'].iloc[i]
            
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
                    to_keep.append(subset_prism['broad_id'].iloc[i])
                    for j in range(len(valid_smiles)):
                        new_matrix['{}:::{}'.format(subset_prism['broad_id'].iloc[i], j)] = {'drug': subset_prism['broad_id'].iloc[i], 'smile' : valid_smiles[j]}
                elif len(valid_smiles) == 1:
                    new_matrix[subset_prism['broad_id'].iloc[i]] = {'drug': subset_prism['broad_id'].iloc[i], 'smile' : valid_smiles[0]}
                    to_keep.append(subset_prism['broad_id'].iloc[i])
        
        del subset_prism
        gc.collect()

        prism_matrix = prism_matrix.loc[prism_matrix['broad_id'].isin(to_keep)]
        new_matrix = pd.DataFrame.from_dict(new_matrix, orient = 'index')
        
        print('\n PRISM dataset (after filtering the valid smiles)')
        print(prism_matrix.shape)
        
        return prism_matrix, new_matrix
    
    # --------------------------------------------------
    
    def filter_cell_lines(self, metadata_sc, drug_data):
        #extract the different cell lines from the single cell dataset
        list_cell_lines_sc_ccle = list(metadata_sc['Cell_line'].unique()) #list of the different cell lines - ccle id
        
        #extract the different cell lines from the prism dataset - depmap id
        list_cell_lines_drug_ccle = list(drug_data['ccle_name'].unique())
        
        #find the common cell lines from both datasets (prism and single cell)
        list_commun_cell_lines = list(set(list_cell_lines_drug_ccle).intersection(list_cell_lines_sc_ccle))
        
        return list_commun_cell_lines

    # --------------------------------------------------
    
    def create_integrated_datasets(self, screens_list, prism_dataset, list_single_cells, pancancer_metadata):
        barcodes_dict = {}
        for ccle in pancancer_metadata['Cell_line'].unique():
            barcodes = list(pancancer_metadata.loc[pancancer_metadata['Cell_line'] == ccle].index)
            barcodes_dict[ccle] = {x : list_single_cells.index(x) for x in barcodes}
        
        screen_indexes_dict = {}
        for i in range(len(screens_list)):
            drug = screens_list[i].split(':::')[0]
            if drug not in screen_indexes_dict:
                screen_indexes_dict[drug] = []
            screen_indexes_dict[drug].append(screens_list[i])
        
        celllines2indexes = {}
        pre_drug2indexes = {}
        new_indexes_dict = {}
        dict_number = {}
        total = 0
        for i in range(prism_dataset.shape[0]):
            sens_value = prism_dataset[self.values_from].iloc[i]
            if sens_value >= 0.000610352 and sens_value <= 10:
                sens_value = -np.log10(sens_value * 10**(-6))
                ccle = prism_dataset['ccle_name'].iloc[i]
                drug = prism_dataset['broad_id'].iloc[i]
                drugs_list = screen_indexes_dict[drug]
                screen_id = prism_dataset['screen_id'].iloc[i]
                for screen in drugs_list:
                    new_index = '{}::{}::{}'.format(ccle, screen, screen_id)
                    if new_index in new_indexes_dict:
                        print('Check indexes because it created the same index')
                        print(new_index)
                        print(list(prism_dataset['ccle_name']))
                        exit()
                    new_indexes_dict[new_index] = ((ccle, barcodes_dict[ccle]), (screens_list.index(screen), screen, drug), sens_value)
                    if ccle not in celllines2indexes:
                        celllines2indexes[ccle] = []
                    celllines2indexes[ccle].append(new_index)
                    total += 1
                    if drug not in pre_drug2indexes:
                        pre_drug2indexes[drug] = []
                    pre_drug2indexes[drug].append(new_index)
                    
                if drug not in dict_number:
                    dict_number[drug] = []
                if ccle not in dict_number[drug]:
                    dict_number[drug].append(ccle)
                
        print('Total indexes: {}'.format(total))
        print('Confirmation: {}'.format(len(new_indexes_dict)))
        
        '''with open('{}/prism_pancancer/n_cells_per_compound.txt'.format(self.path_results), 'w') as f:
            for k,v in dict_number.items():
                f.write('{} :: number of cell lines ({})\n'.format(k, len(v)))
        
        with open('{}/prism_pancancer/n_compounds_per_cell.txt'.format(self.path_results), 'w') as f:
            for k,v in celllines2indexes.items():
                f.write('{} :: number of compounds ({})\n'.format(k, len(v)))
        
        pickle.dump(celllines2indexes, open('{}/prism_pancancer/prism_pancancer_new_indexes_dict.pkl'.format(self.path_results), 'wb'))
        pickle.dump(new_indexes_dict, open('{}/prism_pancancer/prism_pancancer_new_indexes_newIndex2barcodeScreen_dict.pkl'.format(self.path_results), 'wb'))'''
        
        #for the only one compounds/cells models
        drug2indexes = {}
        for k,v in pre_drug2indexes.items():
            if len(v) >= 50:
                drug2indexes[k] = v
        pickle.dump(drug2indexes, open('{}/prism_pancancer/prism_pancancer_drug2indexes_dict.pkl'.format(self.path_results), 'wb'))
        with open('{}/prism_pancancer/prism_pancancer_list_drugs_only_one.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list(drug2indexes.keys())))
        
        final_celllines = []
        for k,v in celllines2indexes.items():
            if len(v) >= 50:
                final_celllines.append(k)
        with open('{}/prism_pancancer/prism_pancancer_celllines2indexes_only_one.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(final_celllines))

    # --------------------------------------------------
    
    def define_path_results(self, values_from):
        self.values_from = values_from
        self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/{}'.format(values_from) #cluster
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
        prism_matrix, smiles_dataframe = self.load_prism()
        
        #load pancancer dataset
        pancancer_metadata = self.load_pancancer()
        
        #get commun cell lines in both datasets
        list_commun_cell_lines_ccle = self.filter_cell_lines(pancancer_metadata, prism_matrix)
        
        #filter datasets
        prism_matrix = prism_matrix.loc[prism_matrix['ccle_name'].isin(list_commun_cell_lines_ccle)]
        smiles_dataframe = smiles_dataframe.loc[smiles_dataframe['drug'].isin(list(prism_matrix['broad_id'].unique()))]
        pancancer_metadata = pancancer_metadata.loc[pancancer_metadata['Cell_line'].isin(list_commun_cell_lines_ccle)]
        list_single_cells = sorted(list(pancancer_metadata.index))
        
        print('\nPRISM dataset (after filtering the common cell lines) \n{}'.format(prism_matrix.shape))
        print('\nPancancer: number of barcodes (after filtering the common cell lines) \n{}'.format(len(list_single_cells)))
        print('\nPancancer metadata (after filtering the common cell lines) \n{}'.format(pancancer_metadata.shape))
        print('\nNumber of bottlenecks (drug) \n{}'.format(smiles_dataframe.shape[0]))
        
        '''prism_matrix.to_csv('{}/prism_dataset.csv'.format(self.path_results), header=True, index=True)
        smiles_dataframe.to_csv('{}/prism_smiles.csv'.format(self.path_results), header=True, index=True)'''
        
        #create the bottlenecks - prism
        '''self.get_smiles_bottlenecks(smiles_dataframe)'''
        list_indexes_prism = list(smiles_dataframe.index)
        '''with open('{}/prism_pancancer/prism_pancancer_screens.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list_indexes_prism))'''
        
        print('PRISM bottlenecks created')
            
        del self.molecular_model
        gc.collect()
        
        for cell in list_commun_cell_lines_ccle:
            self.barcodes_per_cell_line[cell] = list(pancancer_metadata[pancancer_metadata['Cell_line'] == cell].index)
        
        '''with open('{}/prism_pancancer/prism_pancancer_cell_lines_pancancer.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list_commun_cell_lines_ccle))
        list_tumours = list(pancancer_metadata['Cancer_type'].unique())
        with open('{}/prism_pancancer/prism_pancancer_tumours.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list_tumours))
        with open('{}/prism_pancancer/prism_pancancer_barcodes_sc.txt'.format(self.path_results), 'w') as f:
            f.write('\n'.join(list_single_cells))

        pickle.dump(pancancer_metadata, open('{}/pkl_files/pancancer_metadata.pkl'.format(self.path_results), 'wb'))

        barcodes_per_tumour = {}
        for i in range(len(list_tumours)):
            tumour = list_tumours[i]
            barcodes_per_tumour[tumour] = list(pancancer_metadata[pancancer_metadata['Cancer_type'] == tumour].index)
        
        pickle.dump(barcodes_per_tumour, open('{}/prism_pancancer/barcodes_per_tumour_dict.pkl'.format(self.path_results), 'wb'))
        
        ccle_per_barcode = {}
        for k,v in self.barcodes_per_cell_line.items():
            for i in v:
                ccle_per_barcode[i] = k
                
        pickle.dump(self.barcodes_per_cell_line, open('{}/prism_pancancer/barcodes_per_cell_line_dict.pkl'.format(self.path_results), 'wb'))
        pickle.dump(ccle_per_barcode, open('{}/prism_pancancer/ccle_per_barcode_dict.pkl'.format(self.path_results), 'wb'))'''
        
        #create the integrate files
        self.create_integrated_datasets(list_indexes_prism, prism_matrix, list_single_cells, pancancer_metadata)
        
        '''combinations = [['all_genes', 'best_7000'], ['no_pathway', 'kegg_pathways', 'canonical_pathways', 'chemical_genetic_perturbations']]
        for i in range(len(combinations[0])):
            for j in range(len(combinations[1])):
                combination = '{}_{}'.format(combinations[0][i], combinations[1][j])
                self.define_sc_path(combination)
                
                pancancer_dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_data_{}_{}.csv'.format(combinations[0][i], combinations[1][j]), header = 0, index_col = 0)
                pancancer_dataset = pancancer_dataset.loc[pancancer_dataset.index.isin(list_single_cells)]
                pancancer_dataset.sort_index(axis = 0, inplace = True)
                
                #initialize the molecular model
                self.load_scVAE(pancancer_dataset.shape[1])
                
                #create the bottlenecks - pancancer
                self.get_sc_bottlenecks(pancancer_dataset, pancancer_metadata, list_single_cells)
                
                del pancancer_dataset
                gc.collect()
                
                print('pancancer bottlenecks created - {} :: {}'.format(combinations[0][i], combinations[1][j]))
                
                del self.sc_model
                gc.collect()'''
        
        print('DONE!')

# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    values_from = sys.argv[2]
    '''num_gene = sys.argv[3]
    pathway = sys.argv[4]
    combinations = [[num_gene], [pathway]]'''
    if sc_from == 'pancancer':
        process = Process_dataset_pancancer()
        process.run(values_from)

except EOFError:
    print('ERROR!')