# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import gc
import sys

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)

# -------------------------------------------------- DEFINE FUNCTIONS --------------------------------------------------

def create_report(path, list_comments):
    with open('{}/process_report.txt'.format(path), 'a') as f:
        f.write('\n'.join(list_comments))


# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
class Process_dataset():

    def __init__(self):
        self.path_results = None
        self.sc_from = 'ccle'

    # --------------------------------------------------

    def load_sc_data(self, list_commun_cell_lines):
        # path = '/Users/acunha/Desktop/CCLE'
        path = '/hps/research1/icortes/acunha/data/CCLE'
        d = pd.read_csv('{}/CCLE_expression.csv'.format(path), index_col=0)
        conversion = pd.read_csv('{}/sample_info.csv'.format(path), index_col = 0)
        conversion.CCLE_Name = conversion.CCLE_Name.str.replace('-', '')
        conversion = conversion.loc[conversion.CCLE_Name.isin(list_commun_cell_lines)]
        d = d.loc[d.index.isin(list(conversion.index))]
        list_indexes = []
        for i in range(d.shape[0]):
            list_indexes.append(conversion.loc[d.iloc[i].name, 'CCLE_Name'])
        d.index = list_indexes

        del conversion
        del list_indexes
        gc.collect()
        
        lines = ['These lines are not present: {}'.format(set(list_commun_cell_lines).difference(list(d.index)))]
        create_report(self.path_results, lines)
        
        return d, list(d.index)

    # --------------------------------------------------

    def load_drug_data(self, list_commun_cell_lines):        
        dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/pancancer_related/gdsc_ctrp_dataset.csv', index_col = 0)
        dataset = dataset.loc[dataset.Cell_line.isin(list_commun_cell_lines)]
        
        smiles = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/pancancer_related/gdsc_ctrp_smiles.csv', index_col = 0)
        smiles = smiles.loc[smiles.index.isin(list(dataset.Name_compound.unique()))]

        return dataset, smiles

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
        
        # for the only one compounds/cells models
        drug2indexes = {}
        for k, v in pre_drug2indexes.items():
            if len(v) >= 50:
                drug2indexes[k] = v
        pickle.dump(drug2indexes, open('{}/gdsc_ctrp_{}_drug2indexes_dict.pkl'.format(self.path_results, self.sc_from), 'wb'))
        with open('{}/gdsc_ctrp_{}_list_drugs_only_one.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(["".join(filter(str.isalnum, x)) for x in list(drug2indexes.keys())]))

        final_celllines = []
        for k, v in celllines2indexes.items():
            if len(v) >= 50:
                final_celllines.append(k)
        with open('{}/gdsc_ctrp_{}_celllines2indexes_only_one.txt'.format(self.path_results, self.sc_from), 'w') as f:
            f.write('\n'.join(final_celllines))

    # --------------------------------------------------

    def define_path_results(self):
        self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/pancancer_related/{}'.format(self.sc_from)  # cluster
        # self.path_results = '/Users/acunha/Desktop/results'

    # --------------------------------------------------

    def run(self):
        self.define_path_results()
        
        #cell lines from pancancer
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/pancancer_related/pancancer/gdsc_ctrp_pancancer_cell_lines.txt', 'r') as f:
            list_commun_cell_lines = f.readlines()
            list_commun_cell_lines = [x.strip('\n') for x in list_commun_cell_lines]

        #load single cell data
        cell_data, list_commun_cell_lines = self.load_sc_data(list_commun_cell_lines)

        #load drug data
        drug_data, smiles_dataframe = self.load_drug_data(list_commun_cell_lines)

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


process = Process_dataset()
process.run()