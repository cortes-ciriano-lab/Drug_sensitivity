# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import gc
import sys
import torch

from full_network import VAE_gene_expression_single_cell

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- DEFINE FUNCTIONS --------------------------------------------------

def create_report(path, list_comments):
    with open('{}/process_report.txt'.format(path), 'a') as f:
        f.write('\n'.join(list_comments))


# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------
class Process_dataset():

    def __init__(self):
        self.barcodes_per_cell_line = {}
        self.path_results = None
        self.values_from = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sc_from = None


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

    def load_scVAE(self, num_genes):
        if self.sc_from == 'pancancer_centroids':
            path = '/hps/research1/icortes/acunha/python_scripts/single_cell/best_model/pancancer_all_genes_no_pathway'
            _, _, _, _, _, self.dropout_sc, _, _, _, _, _, _, _, self.layers_sc, _, _, self.pathway_sc, self.num_genes_sc = pickle.load(
                open('{}/list_initial_parameters_single_cell.pkl'.format(path), 'rb'))
        else:
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

    def get_sc_bottlenecks(self, dataset):
        self.sc_model.eval()
        with open('{}/{}_outputs.csv'.format(self.path_results, self.sc_from), 'w') as f_o:
            with open('{}/{}_bottlenecks.csv'.format(self.path_results, self.sc_from), 'w') as f_b:
                with torch.no_grad():
                    for i in range(0, dataset.shape[0], 128):
                        batch = dataset.iloc[i:i + 128]
                        list_indexes = list(batch.index)
                        inputs = torch.tensor(batch.to_numpy()).type('torch.FloatTensor').to(self.device)
                        output, bottleneck, _, _, _ = self.sc_model(inputs)
                        output = output.cpu().numpy().tolist()
                        bottleneck = bottleneck.cpu().numpy().tolist()
                        for j in range(len(output)):
                            output[j] = '{},{}\n'.format(list_indexes[j], ','.join([str(x) for x in output[j]]))
                        f_o.write('\n'.join(output))
                        f_o.write('\n')
                        for j in range(len(bottleneck)):
                            bottleneck[j] = '{},{}\n'.format(list_indexes[j], ','.join([str(x) for x in bottleneck[j]]))
                        f_b.write('\n'.join(bottleneck))
                        f_b.write('\n')

    # --------------------------------------------------

    def load_sc_data(self):
        if self.sc_from == 'pancancer_centroids':
            metadata = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/pancancer_related/pancancer/pancancer_metadata.pkl', 'rb'))
        else:
            metadata = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/integrated/integrated_metadata.pkl', 'rb'))

        lines = ['\n Metadata (after loading) \n{}'.format(metadata.shape)]
        create_report(self.path_results, lines)
        print(lines)

        return metadata

    # --------------------------------------------------

    def load_drug_data(self, list_commun_cell_lines):
        if sc_from == 'pancancer_centroids':
            dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/pancancer_related/gdsc_ctrp_dataset.csv', index_col = 0)
        else:
            dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/integrated/gdsc_ctrp_dataset.csv', index_col = 0)
        dataset = dataset.loc[dataset.Cell_line.isin(list_commun_cell_lines)]
        return dataset

    # --------------------------------------------------
    
    def get_centroids(self, sc_dataset, sc_metadata):
        new_dataset = {}
        for cell_line in sc_metadata.Cell_line.unique():
            subset_sc = sc_dataset.loc[sc_dataset.index.isin(list(sc_metadata.loc[sc_metadata.Cell_line == cell_line].index))]
            new_dataset['{}_centroid'.format(cell_line)] = subset_sc.mean().to_dict()
        
        new_dataset = pd.DataFrame.from_dict(new_dataset, orient = 'index')
        
        return new_dataset

    # --------------------------------------------------

    def create_integrated_datasets(self, screens_list, drug_dataset, list_centroids):
        celllines2indexes = {}
        pre_drug2indexes = {}
        new_indexes_dict = {}
        dict_number = {}
        total = 0
        for i in range(drug_dataset.shape[0]):
            row_index = drug_dataset.iloc[i].name
            ccle = drug_dataset.Cell_line.iloc[i]
            position = list_centroids.index('{}_centroid'.format(ccle))
            drug = drug_dataset.Name_compound.iloc[i]
            auc = drug_dataset.AUC.iloc[i]
            if not np.isnan(auc):
                new_indexes_dict[row_index] = ((ccle, position), (screens_list.index(drug), drug), auc)
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
        if self.sc_from == 'pancancer_centroids':
            self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/pancancer_related/{}'.format(self.sc_from)
        else:
            self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/integrated/{}'.format(self.sc_from)
    
    # --------------------------------------------------

    def define_sc_path(self, combination):
        self.run_type = combination

    # --------------------------------------------------

    def run(self, sc_from):
        self.define_sc_data(sc_from)
        self.define_path_results()
        
        if sc_from == 'pancancer_centroids':
            path_data = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/pancancer_related/pancancer'
            with open('{}/gdsc_ctrp_pancancer_cell_lines.txt'.format(path_data), 'r') as f:
                list_commun_cell_lines = f.readlines()
                list_commun_cell_lines = [x.strip('\n') for x in list_commun_cell_lines]
            with open('{}/gdsc_ctrp_pancancer_screens.txt'.format(path_data), 'r') as f:
                list_indexes_drug = f.readlines()
                list_indexes_drug = [x.strip('\n') for x in list_indexes_drug]
        else:
            path_data = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/integrated'
            with open('{}/gdsc_ctrp_integrated_cell_lines.txt'.format(path_data), 'r') as f:
                list_commun_cell_lines = f.readlines()
                list_commun_cell_lines = [x.strip('\n') for x in list_commun_cell_lines]
            with open('{}/gdsc_ctrp_integrated_screens.txt'.format(path_data), 'r') as f:
                list_indexes_drug = f.readlines()
                list_indexes_drug = [x.strip('\n') for x in list_indexes_drug]
        
        #load drug data
        drug_data = self.load_drug_data(list_commun_cell_lines)

        #load single cell data
        sc_metadata = self.load_sc_data()

        if sc_from == 'pancancer_centroids':
            combinations = [['all_genes'], ['no_pathway']]
            combination = '{}_{}'.format(combinations[0][0], combinations[1][0])
            print(combination)
            self.define_sc_path(combination)
            sc_dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/single_cell/data/pancancer/pancancer_data_{}_{}.csv'.format(combinations[0][0], combinations[1][0]), header=0, index_col=0)
        else:
            sc_dataset = pd.read_csv('/hps/research1/icortes/mkalyva/SCintegration/results/Reconstructed_matrix_fastmnn.tsv', sep='\t', index_col=0)
        sc_dataset = sc_dataset.loc[sc_dataset.index.isin(list(sc_metadata.index))]

        sc_dataset = self.get_centroids(sc_dataset, sc_metadata)
        sc_dataset.to_csv('{}/{}_dataset.csv'.format(self.path_results, sc_from), header=True, index=True)
        
        del sc_metadata
        gc.collect()

        # create the integrate files
        self.create_integrated_datasets(list_indexes_drug, drug_data, list(sc_dataset.index))

        # initialize the molecular model
        self.load_scVAE(sc_dataset.shape[1])

        # create the bottlenecks
        self.get_sc_bottlenecks(sc_dataset)

        print('Sc bottlenecks created')

        del self.sc_model
        gc.collect()

        print('DONE!')


# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    process = Process_dataset()
    process.run(sc_from)

except EOFError:
    print('ERROR!')
