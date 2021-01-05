# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import re
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

path_data = '/hps/research1/icortes/acunha/data'
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

def create_report(path, list_comments):
    with open('{}/process_report.txt'.format(path), 'a') as f:
        f.write('\n'.join(list_comments))

# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
class Process_dataset():
    
    def __init__(self):
        self.barcodes_per_cell_line = {}
        self.ohf = OneHotFeaturizer()
        self.path_results = None
        self.values_from = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sc_from = None
        self.drug_from = None
        
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
    
    def define_sc_data(self, value):
        self.sc_from = value
    
    # --------------------------------------------------
    
    def define_drug_data(self, value):
        self.drug_from = value

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
            fingerprints_smiles_dicts[str(dataset.iloc[i].name)] ={'Morgan_Fingerprint' : '[{}]'.format(','.join([str(x) for x in fp]))}
        pd.DataFrame.from_dict(fingerprints_smiles_dicts, orient = 'index').to_csv('{}/molecular/{}_indexes_morgan_fingerprints.csv'.format(self.path_results, self.drug_from), header=True, index=True)
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
        
        with open('{}/molecular/{}_outputs_total.csv'.format(self.path_results, self.drug_from), 'w') as f:
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
        pickle.dump(mol_bottlenecks, open('{}/molecular/{}_bottlenecks.pkl'.format(self.path_results, self.drug_from), 'wb'))
        mol_bottlenecks.to_csv('{}/molecular/{}_bottlenecks.csv'.format(self.path_results, self.drug_from), header=True, index=True)
  
        del bottleneck_complete
        del mol_bottlenecks
        gc.collect()
    
    # --------------------------------------------------
    
    def load_scVAE(self, num_genes):
        if self.sc_from == 'pancancer':
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
        with open('{}/single_cell/{}_{}_outputs.csv'.format(self.path_results, self.sc_from, self.run_type), 'w') as f_o:
            with open('{}/single_cell/{}_{}_pathways.csv'.format(self.path_results, self.sc_from, self.run_type), 'w') as f_p:
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
        bottleneck_complete.to_csv('{}/single_cell/{}_{}_bottlenecks.csv'.format(self.path_results, self.sc_from, self.run_type), header=True, index=True)
        pickle.dump(bottleneck_complete, open('{}/single_cell/{}_{}_bottlenecks.pkl'.format(self.path_results, self.sc_from, self.run_type), 'wb'))
        
        del bottleneck_complete
        del output
        del pathway
        gc.collect
    
    # --------------------------------------------------
    
    def load_pancancer(self):
        #metadata :: rows: AAACCTGAGACATAAC-1-18 ; Cell_line: NCIH2126_LUNG (CCLE_name)
        pancancer_metadata = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_metadata.csv', header = 0, index_col = 0)
        lines = ['\n Pancancer metadata (after loading)\n{}'.format(pancancer_metadata.shape)]
        create_report(self.path_results, lines)
        print(lines)

        return pancancer_metadata, list(pancancer_metadata.Cell_line.unique())
    
    # --------------------------------------------------
        
    def load_prism(self):
        #rows: ACH-000001 ; columns: BRD-A00077618-236-07-6::2.5::HTS
        prism_matrix = pd.read_csv('{}/Prism_19Q4_secondary/secondary-screen-dose-response-curve-parameters.csv'.format(path_data), header=0, usecols= ['broad_id', 'depmap_id', 'ccle_name', 'screen_id', 'auc', 'ic50', 'name', 'moa', 'target', 'smiles', 'passed_str_profiling'])
        lines = ['\n PRISM dataset (after loading) \n{}'.format(prism_matrix.shape)]
        create_report(self.path_results, lines)
        print(lines)
        
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
        
        lines = ['\n PRISM dataset (after filtering the valid smiles) \n{}'.format(prism_matrix.shape)]
        create_report(self.path_results, lines)
        print(lines)
        
        return prism_matrix, new_matrix, list(prism_matrix.ccle_name.unique())
    
    # --------------------------------------------------
    
    def load_gdsc(self):
        drug_data = pd.read_csv('{}/GDSC/gdsc_log10pIC50.csv'.format(path_data), index_col=0)
        drug_data.sort_index(axis=1, inplace=True)
        
        drug_metadata = pd.read_csv('{}/GDSC/gdsc_metadata.csv'.format(path_data), index_col=0)
        drug_metadata.index = drug_metadata.index.astype(str)
        lines = ['\n GDSC dataset (after loading) \n{}'.format(drug_data.shape)]
        create_report(self.path_results, lines)
        print(lines)
        
        # filter the smiles - the metadata has been already filtered for nan values, so now: standardise the smiles (1) and check if the standardised smile is compatible with the molecular VAE (2)
        new_matrix = {}
        for i in range(drug_metadata.shape[0]):
            smile = drug_metadata.Smiles.iloc[i]
            drug_id = drug_metadata.iloc[i].name
            name = drug_metadata.Name.iloc[i]
            if ',' in smile:  # means that exists more than one smile representation of the compound
                smiles = smile.split(',')
            else:
                smiles = [smile]
            standard_smiles = []  # (2)
            for s in smiles:
                try:
                    mol = standardise.run(s)
                    standard_smiles.append(mol)
                except standardise.StandardiseException:
                    pass
            valid_smiles = check_valid_smiles(standard_smiles, 120)  # (3)
            if valid_smiles:
                if len(valid_smiles) > 1:
                    for j in range(len(valid_smiles)):
                        new_matrix['{}:::{}'.format(drug_id, j)] = {'drug_id':drug_id, 'name':name, 'smile':valid_smiles[j]}
                elif len(valid_smiles) == 1:
                    new_matrix[drug_id] = {'drug_id':drug_id, 'name':name, 'smile':valid_smiles[0]}
        new_matrix = pd.DataFrame.from_dict(new_matrix, orient='index')
        new_matrix.index = new_matrix.index.astype(str)
        lines = ['\n GDSC dataset (after filtering the valid smiles) \n{}'.format(drug_data.shape)]
        create_report(self.path_results, lines)
        print(lines)
        
        return drug_data, new_matrix, list(drug_data.index.unique())
    
    # --------------------------------------------------
    
    def filter_cell_lines(self, list_cells_sc, list_cell_drugs):
        #find the common cell lines from both datasets (prism and single cell)
        if self.drug_from == 'prism':
            list_commun_cell_lines = list(set(list_cell_drugs).intersection(list_cells_sc))
            return list_commun_cell_lines
        elif self.drug_from == 'gdsc':
            list_cells_sc = {re.sub('[^A-Za-z0-9]+', '', x.split('_')[0]).lower():x for x in list_cells_sc}
            list_cell_drugs = {re.sub('[^A-Za-z0-9]+', '', x.split('_')[0]).lower():x for x in list_cell_drugs}
            list_commun_cell_lines = list(set(list_cell_drugs.keys()).intersection(list_cells_sc.keys()))
            return list_commun_cell_lines, list_cells_sc, list_cell_drugs
    
    # --------------------------------------------------
    
    def create_integrated_datasets_prism_pancancer(self, screens_list, prism_dataset, list_single_cells, pancancer_metadata):
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
                
        lines = ['Total indexes: {}'.format(total), 'Confirmation: {}'.format(len(new_indexes_dict))]
        create_report(self.path_results, lines)
        print('\n'.join(lines))
        
        with open('{}/n_cells_per_compound.txt'.format(self.path_results), 'w') as f:
            for k,v in dict_number.items():
                f.write('{} :: number of cell lines ({})\n'.format(k, len(v)))
        
        with open('{}/n_compounds_per_cell.txt'.format(self.path_results), 'w') as f:
            for k,v in celllines2indexes.items():
                f.write('{} :: number of compounds ({})\n'.format(k, len(v)))
        
        pickle.dump(celllines2indexes, open('{}/{}_{}_new_indexes_dict.pkl'.format(self.path_results, self.drug_from, self.sc_from), 'wb'))
        pickle.dump(new_indexes_dict, open('{}/{}_{}_new_indexes_newIndex2barcodeScreen_dict.pkl'.format(self.path_results, self.drug_from, self.sc_from), 'wb'))
        
        #for the only one compounds/cells models
        drug2indexes = {}
        for k,v in pre_drug2indexes.items():
            if len(v) >= 50:
                drug2indexes[k] = v
        pickle.dump(drug2indexes, open('{}/{}_{}_drug2indexes_dict.pkl'.format(self.path_results, self.drug_from, self.sc_from), 'wb'))
        with open('{}/{}_{}_list_drugs_only_one.txt'.format(self.path_results, self.drug_from, self.sc_from), 'w') as f:
            f.write('\n'.join(list(drug2indexes.keys())))
        
        final_celllines = []
        for k,v in celllines2indexes.items():
            if len(v) >= 50:
                final_celllines.append(k)
        with open('{}/{}_{}_celllines2indexes_only_one.txt'.format(self.path_results, self.drug_from, self.sc_from), 'w') as f:
            f.write('\n'.join(final_celllines))

    # --------------------------------------------------
    
    def create_integrated_datasets_gdsc_pancancer(self, screens_list, gdsc_dataset, list_single_cells, pancancer_metadata, dict_cells_sc, dict_cell_drugs):
        dict_cells_sc = {v:k for k,v in dict_cells_sc.items()}
        
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
        for ccle in pancancer_metadata['Cell_line'].unique():
            barcodes = list(pancancer_metadata.loc[pancancer_metadata['Cell_line'] == ccle].index)
            barcodes = {x : list_single_cells.index(x) for x in barcodes}
            
            ccle_drug = dict_cell_drugs[dict_cells_sc[ccle]]
            for drug, screens in screen_indexes_dict.items():
                sens_value = gdsc_dataset.loc[ccle_drug, drug]
                if not np.isnan(sens_value):
                    for screen in screens:
                        new_index = '{}::{}'.format(ccle, screen)
                        if new_index in new_indexes_dict:
                            print('Check indexes because it created the same index')
                            print(new_index)
                            exit()
                        new_indexes_dict[new_index] = ((ccle, barcodes), (screens_list.index(screen), screen, drug), sens_value)
                        
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

        lines = ['Total indexes: {}'.format(total), 'Confirmation: {}'.format(len(new_indexes_dict))]
        create_report(self.path_results, lines)
        print('\n'.join(lines))
        
        with open('{}/n_cells_per_compound.txt'.format(self.path_results), 'w') as f:
            for k,v in dict_number.items():
                f.write('{} :: number of cell lines ({})\n'.format(k, len(v)))
        
        with open('{}/n_compounds_per_cell.txt'.format(self.path_results), 'w') as f:
            for k,v in celllines2indexes.items():
                f.write('{} :: number of compounds ({})\n'.format(k, len(v)))
        
        pickle.dump(celllines2indexes, open('{}/{}_{}_new_indexes_dict.pkl'.format(self.path_results, self.drug_from, self.sc_from), 'wb'))
        pickle.dump(new_indexes_dict, open('{}/{}_{}_new_indexes_newIndex2barcodeScreen_dict.pkl'.format(self.path_results, self.drug_from, self.sc_from), 'wb'))
        
        #for the only one compounds/cells models
        drug2indexes = {}
        for k,v in pre_drug2indexes.items():
            if len(v) >= 50:
                drug2indexes[k] = v
        pickle.dump(drug2indexes, open('{}/{}_{}_drug2indexes_dict.pkl'.format(self.path_results, self.drug_from, self.sc_from), 'wb'))
        with open('{}/{}_{}_list_drugs_only_one.txt'.format(self.path_results, self.drug_from, self.sc_from), 'w') as f:
            f.write('\n'.join(list(drug2indexes.keys())))
        
        final_celllines = []
        for k,v in celllines2indexes.items():
            if len(v) >= 50:
                final_celllines.append(k)
        with open('{}/{}_{}_celllines2indexes_only_one.txt'.format(self.path_results, self.drug_from, self.sc_from), 'w') as f:
            f.write('\n'.join(final_celllines))

    # --------------------------------------------------
    
    def define_path_results(self, values_from):
        self.values_from = values_from
        if self.drug_from == 'prism':
            pass
        elif self.drug_from == 'gdsc':
            self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc/{}_{}_{}'.format(self.sc_from, self.drug_from, values_from) #cluster
    
    # --------------------------------------------------
    
    def define_sc_path(self, combination):
        self.run_type = combination
    
    # --------------------------------------------------
    
    def run(self, sc_from, drug_from, values_from):
        self.define_drug_data(drug_from)
        self.define_sc_data(sc_from)
        self.define_path_results(values_from)
        
        #initialize the molecular model
        self.load_smilesVAE()
        
        if drug_from == 'prism':
            #load prism dataset
            drug_data, smiles_dataframe, list_cell_drugs = self.load_prism()
        
        elif drug_from == 'gdsc':
            drug_data, smiles_dataframe, list_cell_drugs = self.load_gdsc()
        
        if sc_from == 'pancancer':
            #load pancancer dataset
            sc_metadata, list_cells_sc = self.load_pancancer()
        
        #get commun cell lines in both datasets and filter datasets
        if sc_from == 'pancancer':
            if drug_from == 'prism':
                list_commun_cell_lines = self.filter_cell_lines(list_cells_sc, list_cell_drugs)
                drug_data = drug_data.loc[drug_data['ccle_name'].isin(list_commun_cell_lines)]
                smiles_dataframe = smiles_dataframe.loc[smiles_dataframe['drug'].isin(list(drug_data['broad_id'].unique()))]
                lines = ['\nPRISM dataset (after filtering the common cell lines) \n{}'.format(drug_data.shape), '\nNumber of bottlenecks (drug) \n{}'.format(smiles_dataframe.shape[0])]
                create_report(self.path_results, lines)
                print('\n'.join(lines))
                
            elif drug_from == 'gdsc':
                list_commun_cell_lines_short, dict_cells_sc, dict_cell_drugs = self.filter_cell_lines(list_cells_sc, list_cell_drugs)
                list_cell_drugs = [v for k,v in dict_cell_drugs.items() if k in list_commun_cell_lines_short]
                drug_data = drug_data.loc[drug_data.index.isin(list_cell_drugs)]
                lines = ['\nGDSC dataset (after filtering the common cell lines) \n{}'.format(drug_data.shape)]
                create_report(self.path_results, lines)
                print(lines)
                list_commun_cell_lines = [v for k,v in dict_cells_sc.items() if k in list_commun_cell_lines_short]
            
            sc_metadata = sc_metadata.loc[sc_metadata['Cell_line'].isin(list_commun_cell_lines)]
            list_single_cells = sorted(list(sc_metadata.index))
            lines = ['\nPancancer: number of barcodes (after filtering the common cell lines) \n{}'.format(len(list_single_cells)), '\nPancancer metadata (after filtering the common cell lines) \n{}'.format(sc_metadata.shape)]
            create_report(self.path_results, lines)
            print('\n'.join(lines))
         
        drug_data.to_csv('{}/{}_dataset.csv'.format(self.path_results, self.drug_from), header=True, index=True)
        smiles_dataframe.to_csv('{}/{}_smiles.csv'.format(self.path_results, self.drug_from), header=True, index=True)
        
        #create the bottlenecks
        self.get_smiles_bottlenecks(smiles_dataframe)
        list_indexes_drug = list(smiles_dataframe.index)
        with open('{}/{}_{}_screens.txt'.format(self.path_results, self.drug_from, self.sc_from), 'w') as f:
            f.write('\n'.join(list_indexes_drug))
    
        print('Drug bottlenecks created')
            
        del self.molecular_model
        gc.collect()
        
        if self.sc_from == 'pancancer':
            for cell in list_commun_cell_lines:
                self.barcodes_per_cell_line[cell] = list(sc_metadata[sc_metadata['Cell_line'] == cell].index)
        
            with open('{}/{}_{}_cell_lines.txt'.format(self.path_results, self.drug_from, self.sc_from), 'w') as f:
                f.write('\n'.join(list_commun_cell_lines))
            list_tumours = list(sc_metadata['Cancer_type'].unique())
            with open('{}/{}_{}_tumours.txt'.format(self.path_results, self.drug_from, self.sc_from), 'w') as f:
                f.write('\n'.join(list_tumours))
            with open('{}/{}_{}_barcodes_sc.txt'.format(self.path_results, self.drug_from, self.sc_from), 'w') as f:
                f.write('\n'.join(list_single_cells))
            
            ccle_per_barcode = {}
            for k,v in self.barcodes_per_cell_line.items():
                for i in v:
                    ccle_per_barcode[i] = k
                    
            pickle.dump(self.barcodes_per_cell_line, open('{}/barcodes_per_cell_line_dict.pkl'.format(self.path_results), 'wb'))
            pickle.dump(ccle_per_barcode, open('{}/ccle_per_barcode_dict.pkl'.format(self.path_results), 'wb'))
            

        pickle.dump(sc_metadata, open('{}/{}_metadata.pkl'.format(self.path_results,self.sc_from), 'wb'))
        
        #create the integrate files
        if drug_from == 'prism' and sc_from == 'pancancer':
            self.create_integrated_datasets_prism_pancancer(list_indexes_drug, drug_data, list_single_cells, sc_metadata)
        elif drug_from == 'gdsc' and sc_from == 'pancancer':
            self.create_integrated_datasets_gdsc_pancancer(list_indexes_drug, drug_data, list_single_cells, sc_metadata, dict_cells_sc, dict_cell_drugs)
        
        if sc_from == 'pancancer':
            combinations = [['all_genes', 'best_7000'], ['no_pathway', 'kegg_pathways', 'canonical_pathways', 'chemical_genetic_perturbations']]
            for i in range(len(combinations[0])):
                for j in range(len(combinations[1])):
                    combination = '{}_{}'.format(combinations[0][i], combinations[1][j])
                    print(combination)
                    self.define_sc_path(combination)
                    
                    pancancer_dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_data_{}_{}.csv'.format(combinations[0][i], combinations[1][j]), header = 0, index_col = 0)
                    pancancer_dataset = pancancer_dataset.loc[pancancer_dataset.index.isin(list_single_cells)]
                    pancancer_dataset.sort_index(axis = 0, inplace = True)
                    
                    #initialize the molecular model
                    self.load_scVAE(pancancer_dataset.shape[1])
                    
                    #create the bottlenecks - pancancer
                    self.get_sc_bottlenecks(pancancer_dataset, sc_metadata, list_single_cells)
                    
                    del pancancer_dataset
                    gc.collect()
                    
                    print('pancancer bottlenecks created - {} :: {}'.format(combinations[0][i], combinations[1][j]))
                    
                    del self.sc_model
                    gc.collect()
        
        print('DONE!')

# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    drug_from = sys.argv[2]
    values_from = sys.argv[3]
    process = Process_dataset()
    process.run(sc_from, drug_from, values_from)

except EOFError:
    print('ERROR!')