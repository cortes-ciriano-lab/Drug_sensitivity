# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import gc
import sys
from rdkit import Chem
from standardiser import standardise
from rdkit.Chem import AllChem
import torch

from featurizer_SMILES import OneHotFeaturizer
from full_network import VAE_gene_expression_single_cell, VAE_molecular

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

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
        print(
            'WARNING!! \nSome molecules have invalid lengths and will not be considered. Please check the file invalid_smiles.txt for more information. \n')

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

        # from the smilesVAE model
        self.molecular_model = None
        self.maximum_length_m = None
        self.dropout_m = None
        self.ohf = None

        # from the scVAE model
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

    def load_smilesVAE(self):
        # path = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/molecular/'
        path = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model'
        _, _, self.maximum_length_m, _, _, _, _, _, self.dropout_m, _, _, _, _, _, _ = pickle.load(
            open('{}/list_initial_parameters_smiles.pkl'.format(path), 'rb'))
        self.dropout_m = float(self.dropout_m)
        self.maximum_length_m = int(self.maximum_length_m)
        self.ohf = OneHotFeaturizer()

        self.molecular_model = VAE_molecular(number_channels_in=self.maximum_length_m,
                                             length_signal_in=len(self.ohf.get_charset()), dropout_prob=self.dropout_m)
        self.molecular_model.load_state_dict(torch.load('{}/molecular_model.pt'.format(path), map_location=self.device))
        self.molecular_model.to(self.device)

    # --------------------------------------------------

    def get_smiles_bottlenecks(self, dataset):
        mols = self.ohf.featurize(list(dataset['smile']), self.maximum_length_m)
        if True in np.isnan(np.array(mols)):
            print('there are nan in the dataset!! \n ')
            sys.exit()

        self.molecular_model.eval()
        valid = 0
        same = 0
        invalid_id = []
        with torch.no_grad():
            with open('{}/molecular/gdsc_ctrp_outputs_total.csv'.format(self.path_results), 'w') as f_o:
                f_o.write('Index;Input;Output\n')
                with open('{}/molecular/gdsc_ctrp_bottlenecks.csv'.format(self.path_results), 'w') as f_b:
                    for i in range(0, dataset.shape[0], 128):
                        batch = mols[i:i + 128]
                        dataset_subset = dataset.iloc[i:i + 128]
                        inputs = torch.tensor(batch).type('torch.FloatTensor').to(self.device)
                        output, bottleneck, _, _ = self.molecular_model(inputs)
                        output = self.ohf.back_to_smile(output)
                        bottleneck = bottleneck.cpu().numpy().tolist()
                        for j in range(dataset_subset.shape[0]):
                            bottleneck[j] = '{};{}'.format(dataset_subset.iloc[j].name, ';'.join([str(x) for x in bottleneck[j]]))
                        f_b.write('\n'.join(bottleneck))
                        f_b.write('\n')
                        for j in range(dataset_subset.shape[0]):
                            m = Chem.MolFromSmiles(output[j])
                            if m is not None:
                                valid += 1
                                if m == dataset_subset['smile'].iloc[j]:
                                    same += 1
                            else:
                                invalid_id.append(i)
                            output[j] = '{};{};{}'.format(dataset_subset.iloc[j].name, dataset_subset['smile'].iloc[j], output[j])
                        f_o.write('\n'.join(output))
                        f_o.write('\n')
        lines = ['\nNumber of valid molecules :: {}'.format(valid),
                 'Number of molecules equal to input :: {}'.format(same)]
        create_report(self.path_results, lines)
    # --------------------------------------------------

    def load_scVAE(self, num_genes):
        if self.sc_from == 'pancancer':
            path = '/hps/research1/icortes/acunha/python_scripts/single_cell/best_model/pancancer_all_genes_no_pathway'
            _, _, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, _, _, self.pathway_sc, self.num_genes_sc = pickle.load(
                open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))
        elif self.sc_from == 'integrated':
            path = '/hps/research1/icortes/acunha/python_scripts/sc_integrated/best_model'
            _, _, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, _, _, self.pathway_sc, self.num_genes_sc, _ = pickle.load(
                open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))

        
        self.dropout_sc = float(self.dropout_sc)
        self.layers_sc = self.layers_sc.split('_')

        number_pathways = 0
        path_matrix_file = ''

        self.sc_model = VAE_gene_expression_single_cell(dropout_prob=self.dropout_sc, n_genes=num_genes,
                                                        layers=self.layers_sc, n_pathways=number_pathways,
                                                        path_matrix=path_matrix_file)
        self.sc_model.load_state_dict(torch.load('{}/single_cell_model.pt'.format(path), map_location=self.device))
        self.sc_model.to(self.device)

    # --------------------------------------------------

    def get_sc_bottlenecks(self, dataset, metadata, indexes):
        self.sc_model.eval()
        with open('{}/single_cell/{}_{}_outputs.csv'.format(self.path_results, self.sc_from, self.run_type),
                  'w') as f_o:
            with open('{}/single_cell/{}_{}_bottlenecks.csv'.format(self.path_results, self.sc_from, self.run_type),
                      'w') as f_b:
                with torch.no_grad():
                    for i in range(0, len(indexes), 128):
                        list_indexes = indexes[i:i + 128]
                        batch = dataset.iloc[i:i + 128]
                        inputs = torch.tensor(batch.to_numpy()).type('torch.FloatTensor').to(self.device)
                        output, bottleneck, _, _, _ = self.sc_model(inputs)
                        output = output.cpu().numpy().tolist()
                        bottleneck = bottleneck.cpu().numpy().tolist()
                        for j in range(len(output)):
                            output[j] = '{},{}\n'.format(list_indexes[j], ','.join([str(x) for x in output[j]]))
                        f_o.write('\n'.join(output))
                        f_o.write('\n')
                        for j in range(len(bottleneck)):
                            bottleneck[j] = '{},{},{}\n'.format(list_indexes[j], ','.join([str(x) for x in bottleneck[j]]), metadata.loc[list_indexes[j], 'Cell_line'])
                        f_b.write('\n'.join(bottleneck))
                        f_b.write('\n')

    # --------------------------------------------------

    def load_sc_data(self):
        if self.sc_from == 'pancancer':
            path = '/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer'
            metadata = pd.read_csv('{}/pancancer_metadata.csv'.format(path), header=0, index_col=0)

        elif self.sc_from == 'integrated':
            path = '/hps/research1/icortes/acunha/data/Integrated_dataset_M'
            full_metadata = pd.read_csv('{}/Metadata_Merged.tsv'.format(path), sep='\t', index_col=0)
            full_metadata = full_metadata.loc[full_metadata.batch.isin(['McF', 'pancancer'])]
            full_metadata = full_metadata.loc[:, ['batch', 'Cell_line', 'doublet_CL2', 'Cancer_type']]
            cell_info = pd.read_csv('/hps/research1/icortes/acunha/data/CCLE/sample_info.csv', index_col=0, header=0)
            metadata = {}
            for i in range(full_metadata.shape[0]):
                barcode = full_metadata.iloc[i].name
                paper = full_metadata.batch.iloc[i]
                if paper == 'McF':
                    ccle = full_metadata['doublet_CL2'].iloc[i]
                else:
                    ccle = full_metadata['Cell_line'].iloc[i]
                cancer = full_metadata['Cancer_type'].iloc[i]
                if pd.isnull(cancer):
                    cancer = list(cell_info.loc[cell_info.CCLE_Name == ccle, 'disease'].str.replace(' ', '_').str.replace('/','__'))
                    if len(cancer) != 1:
                        cancer = float('NaN')
                    else:
                        cancer = cancer[0]    
                metadata[barcode] = {'Cell_line':ccle, 'Cancer_type':cancer, 'Batch':paper}
            
            metadata = pd.DataFrame.from_dict(metadata, orient = 'index')

        lines = ['\n Metadata (after loading)\n{}'.format(metadata.shape)]
        create_report(self.path_results, lines)
        print(lines)

        return metadata, list(metadata.Cell_line.unique())

    # --------------------------------------------------

    def load_drug_data(self):
        path = '/hps/research1/icortes/acunha/data'
        dataset = pd.read_csv('{}/GDSC_CTRP/Final_dataset.csv'.format(path), index_col=0)

        new_matrix = {}
        to_keep = []
        for i in range(dataset.shape[0]):
            try:
                mol = standardise.run(dataset.Smile.iloc[i])
                valid_smiles = check_valid_smiles([mol], self.maximum_length_m)  # (3)
                if len(valid_smiles) == 1:
                    row_index = dataset.iloc[i].name
                    to_keep.append(row_index)
                    new_matrix[dataset.Name_compound.iloc[i]] = mol
            except standardise.StandardiseException:
                pass
            

        dataset = dataset.loc[dataset.index.isin(to_keep)]
        indexes = list(dataset.index)
        names = list(dataset.Name_compound)
        indexes = [x.replace(' ', '__') for x in indexes]
        names =  [x.replace(' ', '__') for x in names]
        dataset.index = indexes
        dataset.Name_compound = names
        new_matrix = pd.DataFrame.from_dict(new_matrix, orient= 'index')
        new_matrix.columns = ['smile']
        indexes = list(new_matrix.index)
        indexes = [x.replace(' ', '__') for x in indexes]
        new_matrix.index = indexes

        lines = ['\n Drug dataset (after filtering the valid smiles) \n{}'.format(new_matrix.shape)]
        create_report(self.path_results, lines)
        print(''.join(lines))

        return dataset, new_matrix, list(dataset.Cell_line.unique())

    # --------------------------------------------------

    def filter_cell_lines(self, list_cells_sc, list_cell_drugs):
        # find the common cell lines from both datasets
        list_commun_cell_lines = list(set(list_cell_drugs).intersection(list_cells_sc))
        return list_commun_cell_lines

    # --------------------------------------------------

    def create_integrated_datasets(self, screens_list, drug_dataset, list_single_cells,
                                                  metadata):

        celllines2indexes = {}
        pre_drug2indexes = {}
        new_indexes_dict = {}
        dict_number = {}
        total = 0
        for i in range(drug_dataset.shape[0]):
            row_index = drug_dataset.iloc[i].name
            ccle = drug_dataset.Cell_line.iloc[i]
            barcodes = list(metadata.loc[metadata['Cell_line'] == ccle].index)
            barcodes = {x: list_single_cells.index(x) for x in barcodes}
            drug = drug_dataset.Name_compound.iloc[i]
            auc = drug_dataset.AUC.iloc[i]
            if not np.isnan(auc):
                new_indexes_dict[row_index] = ((ccle, barcodes), (screens_list.index(drug), drug), auc)
                if ccle not in celllines2indexes:
                    celllines2indexes[ccle] = []
                celllines2indexes[ccle].append(row_index)
                total += 1
                if drug not in pre_drug2indexes:
                    pre_drug2indexes[drug] = []
                pre_drug2indexes[drug].append(row_index)
            if drug not in dict_number:
                dict_number[drug] = []
            if ccle not in dict_number[drug]:
                dict_number[drug].append(ccle)

        lines = ['Total indexes: {}'.format(total), 'Confirmation: {}'.format(len(new_indexes_dict))]
        create_report(self.path_results, lines)
        print('\n'.join(lines))

        with open('{}/n_cells_per_compound.txt'.format(self.path_results), 'w') as f:
            for k, v in dict_number.items():
                f.write('{} :: number of cell lines ({})\n'.format(k, len(v)))

        with open('{}/n_compounds_per_cell.txt'.format(self.path_results), 'w') as f:
            for k, v in celllines2indexes.items():
                f.write('{} :: number of compounds ({})\n'.format(k, len(v)))

        pickle.dump(celllines2indexes,
                    open('{}/gdsc_ctrp_{}_new_indexes_dict.pkl'.format(self.path_results, self.sc_from), 'wb'))
        pickle.dump(new_indexes_dict, open(
            '{}/gdsc_ctrp_{}_new_indexes_newIndex2barcodeScreen_dict.pkl'.format(self.path_results,
                                                                          self.sc_from), 'wb'))

        # for the only one compounds/cells models
        drug2indexes = {}
        for k, v in pre_drug2indexes.items():
            if len(v) >= 50:
                drug2indexes[k] = v
        pickle.dump(drug2indexes,
                    open('{}/gdsc_ctrp_{}_drug2indexes_dict.pkl'.format(self.path_results, self.sc_from),
                         'wb'))
        with open('{}/gdsc_ctrp_{}_list_drugs_only_one.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(list(drug2indexes.keys())))

        final_celllines = []
        for k, v in celllines2indexes.items():
            if len(v) >= 50:
                final_celllines.append(k)
        with open('{}/gdsc_ctrp_{}_celllines2indexes_only_one.txt'.format(self.path_results, self.sc_from),
                  'w') as f:
            f.write('\n'.join(final_celllines))

    # --------------------------------------------------

    def define_path_results(self):
        self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/{}'.format(self.sc_from)  # cluster
    # --------------------------------------------------

    def define_sc_path(self, combination):
        self.run_type = combination

    # --------------------------------------------------

    def run(self, sc_from):
        self.define_sc_data(sc_from)
        self.define_path_results()

        # initialize the molecular model
        self.load_smilesVAE()

        #load drug data
        drug_data, smiles_dataframe, list_cell_drugs = self.load_drug_data()

        #load single cell data
        sc_metadata, list_cells_sc = self.load_sc_data()

        combinations = [['all_genes'], ['no_pathway']]
        combination = '{}_{}'.format(combinations[0][0], combinations[1][0])
        print(combination)
        self.define_sc_path(combination)

        if sc_from == 'pancancer':
            path = '/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer'
            sc_dataset = pd.read_csv('{}/pancancer_data_{}_{}.csv'.format(path, combinations[0][0], combinations[1][0]),
                                     header=0, index_col=0)
        else:
            sc_dataset = pd.read_csv('/hps/research1/icortes/mkalyva/SCintegration/results/Reconstructed_matrix_fastmnn.tsv', sep='\t', index_col=0)
        sc_dataset = sc_dataset.loc[sc_dataset.index.isin(list(sc_metadata.index))]
        sc_metadata = sc_metadata.loc[sc_metadata.index.isin(list(sc_dataset.index))]

        # get commun cell lines in both datasets and filter datasets
        list_commun_cell_lines = self.filter_cell_lines(list_cells_sc, list_cell_drugs)
        '''drug_data = drug_data.loc[drug_data['Cell_line'].isin(list_commun_cell_lines)]
        smiles_dataframe = smiles_dataframe.loc[smiles_dataframe.index.isin(list(drug_data['Name_compound'].unique()))]
        lines = ['\nDrug dataset (after filtering the common cell lines) \n{}'.format(drug_data.shape),
                 '\nNumber of bottlenecks (drug) \n{}'.format(smiles_dataframe.shape[0])]
        # create_report(self.path_results, lines)
        print('\n'.join(lines))'''

        sc_metadata = sc_metadata.loc[sc_metadata['Cell_line'].isin(list_commun_cell_lines)]
        list_single_cells = sorted(list(sc_metadata.index))
        '''sc_dataset = sc_dataset.loc[sc_dataset.index.isin(list_single_cells)]
        lines = ['\nSc Dataset: number of barcodes (after filtering the common cell lines) \n{}'.format(
            len(list_single_cells)),
                 '\nSc metadata (after filtering the common cell lines) \n{}'.format(sc_metadata.shape)]
        create_report(self.path_results, lines)
        print('\n'.join(lines))'''

        '''drug_data.to_csv('{}/gdsc_ctrp_dataset.csv'.format(self.path_results), header=True, index=True)
        smiles_dataframe.to_csv('{}/gdsc_ctrp_smiles.csv'.format(self.path_results), header=True, index=True)
        
        
        # create the bottlenecks
        self.get_smiles_bottlenecks(smiles_dataframe)
        list_indexes_drug = list(smiles_dataframe.index)
        with open('{}/gdsc_ctrp_{}_screens.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(list_indexes_drug))'''

        '''print('Drug bottlenecks created')

        del self.molecular_model
        gc.collect()

        for cell in list_commun_cell_lines:
            self.barcodes_per_cell_line[cell] = list(sc_metadata[sc_metadata['Cell_line'] == cell].index)

        with open('{}/gdsc_ctrp_{}_cell_lines.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(list_commun_cell_lines))
        list_tumours = list(sc_metadata['Cancer_type'].unique())
        list_tumours = [x for x in list_tumours if not pd.isnull(x)]
        with open('{}/gdsc_ctrp_{}_tumours.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(list_tumours))
        with open('{}/gdsc_ctrp_{}_barcodes_sc.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(list_single_cells))

        ccle_per_barcode = {}
        for k, v in self.barcodes_per_cell_line.items():
            for i in v:
                ccle_per_barcode[i] = k

        pickle.dump(self.barcodes_per_cell_line,
                    open('{}/barcodes_per_cell_line_dict.pkl'.format(self.path_results), 'wb'))
        pickle.dump(ccle_per_barcode, open('{}/ccle_per_barcode_dict.pkl'.format(self.path_results), 'wb'))'''

        pickle.dump(sc_metadata, open('{}/{}_metadata.pkl'.format(self.path_results, self.sc_from), 'wb'))

        # create the integrate files
        # self.create_integrated_datasets(list_indexes_drug, drug_data, list_single_cells,sc_metadata)

        '''sc_dataset = sc_dataset.loc[sc_dataset.index.isin(list_single_cells)]
        sc_dataset.sort_index(axis=0, inplace=True)

        # initialize the molecular model
        self.load_scVAE(sc_dataset.shape[1])

        # create the bottlenecks
        self.get_sc_bottlenecks(sc_dataset, sc_metadata, list_single_cells)

        print('Sc bottlenecks created - {} :: {}'.format(combinations[0][0], combinations[1][0]))

        del self.sc_model
        gc.collect()'''

        print('DONE!')


# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    process = Process_dataset()
    process.run(sc_from)

except EOFError:
    print('ERROR!')