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
        print('WARNING!! \nSome molecules have invalid lengths and will not be considered. Please check the file invalid_smiles.txt for more information. \n')

    return valid_smiles

def create_report(path, list_comments):
    with open('{}/process_report.txt'.format(path), 'a') as f:
        f.write('\n'.join(list_comments))


# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
class Process_dataset():

    def __init__(self):
        self.ohf = OneHotFeaturizer()
        self.path_results = None
        self.sc_from = 'ccle'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # from the smilesVAE model
        self.molecular_model = None
        self.maximum_length_m = 120
        self.dropout_m = None
        self.ohf = None

    # --------------------------------------------------

    def load_smilesVAE(self):
        # path = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/molecular/'
        # path = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model'
        # _, _, self.maximum_length_m, _, _, _, _, _, self.dropout_m, _, _, _, _, _, _ = pickle.load(open('{}/list_initial_parameters_smiles.pkl'.format(path), 'rb'))

        # path = '/Users/acunha/Desktop/best_smiles_olds' #the old version
        path = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model_old' #the old version
        _, self.maximum_length_m, _, _, _, _, _, self.dropout_m, _, _, _, _, _ = pickle.load(open('{}/list_initial_parameters_smiles.pkl'.format(path), 'rb')) #old smilesVAE

        self.dropout_m = float(self.dropout_m)
        self.maximum_length_m = int(self.maximum_length_m)
        self.ohf = OneHotFeaturizer()

        self.molecular_model = VAE_molecular(number_channels_in=self.maximum_length_m, length_signal_in=len(self.ohf.get_charset()), dropout_prob=self.dropout_m)
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
                f_o.write('Index\tInput\tOutput\n')
                with open('{}/molecular/gdsc_ctrp_bottlenecks.csv'.format(self.path_results), 'w') as f_b:
                    for i in range(0, dataset.shape[0], 128):
                        batch = mols[i:i + 128]
                        dataset_subset = dataset.iloc[i:i + 128]
                        inputs = torch.tensor(batch).type('torch.FloatTensor').to(self.device)
                        output, bottleneck, _, _ = self.molecular_model(inputs)
                        output = self.ohf.back_to_smile(output)
                        bottleneck = bottleneck.cpu().numpy().tolist()
                        for j in range(dataset_subset.shape[0]):
                            bottleneck[j] = '{}\t{}'.format(dataset_subset.iloc[j].name, '\t'.join([str(x) for x in bottleneck[j]]))
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
                            output[j] = '{}\t{}\t{}'.format(dataset_subset.iloc[j].name, dataset_subset['smile'].iloc[j], output[j])
                        f_o.write('\n'.join(output))
                        f_o.write('\n')
        lines = ['Number of valid molecules :: {}'.format(valid),
                 'Number of molecules equal to input :: {}\n'.format(same)]
        create_report(self.path_results, lines)

    # --------------------------------------------------

    def get_smiles_fp(self, dataset):
        new_data = {}
        for i in range(dataset.shape[0]):
            smile = dataset.iloc[i, 0]
            index = dataset.iloc[i].name
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=1024))
            new_data[index] = fp

        new_data = pd.DataFrame.from_dict(new_data, orient='index')
        new_data.to_csv('{}/molecular/gdsc_ctrp_fp.csv'.format(self.path_results), header=True, index=True)

    # --------------------------------------------------

    def load_sc_data(self):
        # path = '/Users/acunha/Desktop/CCLE'
        path = '/hps/research1/icortes/acunha/data/CCLE'
        d = pd.read_csv('{}/CCLE_expression.csv'.format(path), index_col=0)
        conversion = pd.read_csv('{}/sample_info.csv'.format(path), index_col = 0)
        d = d.loc[d.index.isin(list(conversion.index))]
        list_indexes = []
        for i in range(d.shape[0]):
            list_indexes.append(conversion.loc[d.iloc[i].name, 'CCLE_Name'])
        d.index = list_indexes

        del conversion
        del list_indexes
        gc.collect()

        return d, list(d.index)

    # --------------------------------------------------

    def load_drug_data(self):
        path = '/hps/research1/icortes/acunha/data'
        # path = '/Users/acunha/Desktop'
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

        lines = ['Drug dataset (after filtering the valid smiles) :: {}\n'.format(new_matrix.shape)]
        create_report(self.path_results, lines)
        print(''.join(lines))

        return dataset, new_matrix, list(dataset.Cell_line.unique())

    # --------------------------------------------------

    def filter_cell_lines(self, list_cells_sc, list_cell_drugs):
        # find the common cell lines from both datasets
        list_commun_cell_lines = list(set(list_cell_drugs).intersection(list_cells_sc))
        return list_commun_cell_lines

    # --------------------------------------------------

    def create_integrated_datasets(self, screens_list, drug_dataset, list_ccle_index):
        celllines2indexes = {}
        pre_drug2indexes = {}
        new_indexes_dict = {}
        dict_number = {}
        total = 0
        for i in range(drug_dataset.shape[0]):
            row_index = drug_dataset.iloc[i].name
            ccle = drug_dataset.Cell_line.iloc[i]
            drug = drug_dataset.Name_compound.iloc[i]
            auc = drug_dataset.AUC.iloc[i]
            if not np.isnan(auc):
                new_indexes_dict[row_index] = ((list_ccle_index.index(ccle), ccle), (screens_list.index(drug), drug), auc)
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

        lines = ['Total indexes: {}'.format(total), 'Confirmation: {}\n'.format(len(new_indexes_dict))]
        create_report(self.path_results, lines)
        print('\n'.join(lines))

        with open('{}/n_cells_per_compound.txt'.format(self.path_results), 'w') as f:
            for k, v in dict_number.items():
                f.write('{} :: number of cell lines ({})\n'.format(k, len(v)))

        with open('{}/n_compounds_per_cell.txt'.format(self.path_results), 'w') as f:
            for k, v in celllines2indexes.items():
                f.write('{} :: number of compounds ({})\n'.format(k, len(v)))

        pickle.dump(celllines2indexes, open('{}/gdsc_ctrp_{}_new_indexes_dict.pkl'.format(self.path_results, self.sc_from), 'wb'))
        pickle.dump(new_indexes_dict, open('{}/gdsc_ctrp_{}_new_indexes_newIndex2barcodeScreen_dict.pkl'.format(self.path_results, self.sc_from), 'wb'))

    # --------------------------------------------------

    def define_path_results(self):
        self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/{}'.format(self.sc_from)  # cluster
        # self.path_results = '/Users/acunha/Desktop/results'

    # --------------------------------------------------

    def run(self):
        self.define_path_results()

        # initialize the molecular model
        self.load_smilesVAE()

        #load drug data
        drug_data, smiles_dataframe, list_cell_drugs = self.load_drug_data()

        #load single cell data
        cell_data, list_cells = self.load_sc_data()

        # get commun cell lines in both datasets and filter datasets
        list_commun_cell_lines = self.filter_cell_lines(list_cells, list_cell_drugs)
        drug_data = drug_data.loc[drug_data['Cell_line'].isin(list_commun_cell_lines)]
        smiles_dataframe = smiles_dataframe.loc[smiles_dataframe.index.isin(list(drug_data['Name_compound'].unique()))]
        lines = ['Drug dataset (after filtering the common cell lines) :: {}'.format(drug_data.shape),
                 'Number of bottlenecks (drug) :: {}\n'.format(smiles_dataframe.shape[0])]
        create_report(self.path_results, lines)
        print('\n'.join(lines))

        cell_data = cell_data.loc[cell_data.index.isin(list_commun_cell_lines)]
        lines = ['CCLE Dataset: number of cell lines (after filtering the common cell lines) :: {}\n'.format(len(list_commun_cell_lines))]
        create_report(self.path_results, lines)
        print('\n'.join(lines))

        drug_data.to_csv('{}/gdsc_ctrp_dataset.csv'.format(self.path_results), header=True, index=True)
        smiles_dataframe.to_csv('{}/gdsc_ctrp_smiles.csv'.format(self.path_results), header=True, index=True)

        # create the bottlenecks
        self.get_smiles_bottlenecks(smiles_dataframe)
        print('Drug bottlenecks created')

        del self.molecular_model
        gc.collect()

        self.get_smiles_fp(smiles_dataframe)

        list_indexes_drug = list(smiles_dataframe.index)
        with open('{}/gdsc_ctrp_{}_screens.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(list_indexes_drug))

        with open('{}/gdsc_ctrp_{}_cell_lines.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(list_commun_cell_lines))

        cell_data.to_csv('{}/ccle_dataset.csv'.format(self.path_results), index = True, header = True)

        # create the integrate files
        self.create_integrated_datasets(list_indexes_drug, drug_data, list(cell_data.index))

        print('DONE!')


# -------------------------------------------------- INPUT --------------------------------------------------

try:
    process = Process_dataset()
    process.run()

except EOFError:
    print('ERROR!')