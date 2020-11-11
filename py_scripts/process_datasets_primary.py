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
path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/' #cluster
# path_results = 'C:/Users/abeat/Documents/GitHub/Drug_sensitivity/'

# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
class Process_dataset_pancancer():
    
    def __init__(self, **kwargs):
        self.ccle2depmap = {}
        self.depmap2ccle = {}
        self.barcodes_per_cell_line = {}
        self.ohf = OneHotFeaturizer()
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
                bottleneck_complete.extend(predictions[1].cpu().numpy().tolist())
                output = self.ohf.back_to_smile(predictions[0].cpu().numpy().tolist())
                predictions_complete.extend(output)
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
                    
        
        mol_outputs = pd.DataFrame(predictions_complete)
        mol_outputs.index = indexes
        mol_outputs.to_csv('{}/molecular/run_once/prism_outputs.csv'.format(self.path_results), header=True, index=True)
        
        del predictions_complete
        del mol_outputs
        gc.collect()
        
        mol_bottlenecks = pd.DataFrame(bottleneck_complete)
        mol_bottlenecks.index = indexes
        pickle.dump(mol_bottlenecks, open('{}/molecular/run_once/prism_bottlenecks.pkl'.format(self.path_results), 'wb'))
        mol_bottlenecks.to_csv('{}/molecular/run_once/prism_bottlenecks.csv'.format(self.path_results), header=True, index=True)
  
        del bottleneck_complete
        del mol_bottlenecks
        gc.collect()
        
        return indexes
    
    # --------------------------------------------------
    
    def load_scVAE(self, num_genes):
        path = '/hps/research1/icortes/acunha/python_scripts/single_cell/best_model/pancancer_{}'.format(self.run_type)
        
        if self.run_type == 'all_genes_no_pathway' or self.run_type == 'all_genes_canonical_pathways':
            _, self.batch_sc, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, self.alpha_sc, _, self.pathway_sc, self.num_genes_sc = pickle.load(open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))
        else:
            _, self.batch_sc, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, self.alpha_sc, _, self.pathway_sc, self.num_genes_sc, _ = pickle.load(open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))
        self.batch_sc = int(self.batch_sc)
        self.dropout_sc = float(self.dropout_sc)
        self.layers_sc = self.layers_sc.split('_')
        self.alpha_sc = float(self.alpha_sc)
        
        if self.pathway_sc != 'no_pathway':
            pathways = {'canonical_pathways' : '/hps/research1/icortes/acunha/data/pathways/canonical_pathways/',
                        'chemical_genetic_perturbations' : '/hps/research1/icortes/acunha/data/pathways/chemical_genetic_perturbations/',
                        'kegg_pathways' : '/hps/research1/icortes/acunha/data/pathways/kegg_pathways'}
            list_pathways = pickle.load(open('{}/list_pathways.pkl'.format(pathways[self.pathway]), 'rb'))
            number_pathways = len(list_pathways)
            path_matrix_file = '/hps/research1/icortes/acunha/python_scripts/single_cell/data/pathway_matrices/pancancer_matrix_{}_{}_only_values.csv'.format(self.num_genes, self.pathway)
        else:
            number_pathways = 0
            path_matrix_file = ''
        
        self.sc_model = VAE_gene_expression_single_cell(dropout_prob=self.dropout_sc, n_genes=num_genes, layers=self.layers_sc, n_pathways = number_pathways, path_matrix = path_matrix_file)
        self.sc_model.to(self.device)
        
        model_parameters = pickle.load(open('{}/single_cell_model.pkl'.format(path), 'rb'))
        self.sc_model.load_state_dict(model_parameters)
    
    # --------------------------------------------------
    
    def get_sc_bottlenecks(self, dataset, metadata, indexes):
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
                pathway = predictions[-1].cpu().numpy().tolist()
                for j in range(len(output)):
                    output[j] = '{},{}\n'.format(list_indexes[j], ','.join([str(x) for x in output[j]]))
                with open('{}/single_cell/pancancer_{}_outputs.csv'.format(self.path_results, self.run_type), 'a') as f:
                    f.write('\n'.join(output))
                for j in range(len(pathway)):
                    pathway[j] = '{},{}\n'.format(list_indexes[j], ','.join([str(x) for x in pathway[j]]))
                with open('{}/single_cell/pancancer_{}_pathways.csv'.format(self.path_results, self.run_type), 'a') as f:
                    f.write('\n'.join(pathway))
                    
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

    def __create_ccle2depmap_and_depmap2ccle(self):
        cell_info = pd.read_csv('{}/CCLE/sample_info.csv'.format(path_data), index_col=0, header=0)
        for i in range(cell_info.shape[0]):
            self.ccle2depmap[cell_info['CCLE_Name'].iloc[i]] = cell_info.iloc[i, :].name #converter ccle to depmap
            self.depmap2ccle[cell_info.iloc[i, :].name] = cell_info['CCLE_Name'].iloc[i] #converter depmap to ccle
        
        del cell_info
        gc.collect()
    
    # --------------------------------------------------
    
    def load_pancancer(self):
        #metadata :: rows: AAACCTGAGACATAAC-1-18 ; Cell_line: NCIH2126_LUNG (CCLE_name)
        pancancer_metadata = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_metadata.csv', header = 0, index_col = 0)
        print('\n Pancancer metadata (after loading)')
        print(pancancer_metadata.shape)

        return pancancer_metadata
    
    # --------------------------------------------------
        
    def load_prism(self, maximum_length_smiles):
        if self.prism_screen == 'primary':
            #rows: ACH-000001 ; columns: BRD-A00077618-236-07-6::2.5::HTS
            prism_matrix = pd.read_csv('{}/Drug_Sensitivity_PRISM/primary-screen-replicate-collapsed-logfold-change.csv'.format(path_data), sep=',', header=0,
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
    
    def create_integrated_datasets(self, screens_list, prism_dataset, prism_bottlenecks, pancancer_bottlenecks, pancancer_metadata, what_type):
        barcode2indexes = {}
        new_indexes_dict = {}
        # ccle_to_barcode = {}

        for ccle in pancancer_metadata['Cell_line'].unique():
            barcodes = list(pancancer_metadata.loc[pancancer_metadata['Cell_line'] == ccle].index)
            barcodes = {x : pancancer_bottlenecks.index.get_loc(x) for x in barcodes}
            # ccle_to_barcode[ccle] = list(barcodes.keys())
            depmap = self.ccle2depmap[ccle]
            indexes = []
            for j in range(prism_bottlenecks.shape[0]):
                screen = prism_bottlenecks.iloc[j].name
                sens_value = prism_dataset.loc[depmap, screen.split(':::')[0]]
                if not np.isnan(sens_value):
                    new_index = '{}::{}'.format(ccle, screen)
                    new_indexes_dict[new_index] = ((ccle, barcodes), (screen, j), sens_value)
                    indexes.append(new_index)
            barcode2indexes[ccle] = indexes

        pickle.dump(barcode2indexes, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_new_indexes_{}_dict.pkl'.format(what_type), 'wb'))
        pickle.dump(new_indexes_dict, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_new_indexes_{}_newIndex2barcodeScreen_dict.pkl'.format(what_type), 'wb'))
        # pickle.dump(ccle_to_barcode, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_new_indexes_{}_ccle2barcode_dict.pkl'.format(what_type), 'wb'))

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
        
        pancancer_bottlenecks, _ = create_pancancer_bottleneck('data')
        #create the integrate files

        self.create_integrated_datasets(list_indexes_prism, prism_matrix, prism_bottlenecks, pancancer_bottlenecks, pancancer_metadata, 'once')
        
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