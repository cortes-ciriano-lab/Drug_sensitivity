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

    # --------------------------------------------------

    def define_sc_data(self, value):
        self.sc_from = value

    # --------------------------------------------------

    def load_smilesVAE(self):
        # path = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/molecular/'
        path = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model_old'
        _, self.maximum_length_m, _, _, _, _, _, self.dropout_m, _, _, _, _, _ = pickle.load(open('{}/list_initial_parameters_smiles.pkl'.format(path), 'rb'))
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
            with open('{}/molecular/gdsc_ctrp_outputs_total_old.csv'.format(self.path_results), 'w') as f_o:
                f_o.write('Index\tInput\tOutput\n')
                with open('{}/molecular/gdsc_ctrp_bottlenecks_old.csv'.format(self.path_results), 'w') as f_b:
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
        lines = ['\nNumber of valid molecules :: {}'.format(valid),
                 'Number of molecules equal to input :: {}'.format(same)]
        create_report(self.path_results, lines)

    # --------------------------------------------------

    def define_path_results(self):
        self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/{}'.format(self.sc_from)  # cluster

    # --------------------------------------------------

    def run(self, sc_from):
        self.define_sc_data(sc_from)
        self.define_path_results()

        new_data = {}
        smiles_dataframe = pd.read_csv('{}/gdsc_ctrp_smiles.csv'.format(self.path_results), index_col=0)
        
        # initialize the molecular model
        self.load_smilesVAE()
        
        # create the bottlenecks
        self.get_smiles_bottlenecks(smiles_dataframe)

        print('DONE!')


# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    process = Process_dataset()
    process.run(sc_from)

except EOFError:
    print('ERROR!')