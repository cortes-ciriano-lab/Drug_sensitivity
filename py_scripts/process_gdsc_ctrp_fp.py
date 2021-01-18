# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import gc
import sys
from rdkit import Chem
from standardiser import standardise
from rdkit.Chem import AllChem

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
        self.sc_from = None

    # --------------------------------------------------

    def define_sc_data(self, value):
        self.sc_from = value

    # --------------------------------------------------

    def define_path_results(self):
        self.path_results = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/{}'.format(self.sc_from)  # cluster

    # --------------------------------------------------

    def run(self, sc_from):
        self.define_sc_data(sc_from)
        self.define_path_results()

        new_data = {}
        smiles_dataframe = pd.read_csv('{}/gdsc_ctrp_smiles.csv'.format(self.path_results), index_col=0)
        for i in range(smiles_dataframe.shape[0]):
            smile = smiles_dataframe.iloc[i, 0]
            index = smiles_dataframe.iloc[i].name
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=1024))
            new_data[index] = fp

        new_data = pd.DataFrame.from_dict(new_data, orient='index')
        new_data.to_csv('{}/molecular/gdsc_ctrp_fp.csv'.format(self.path_results), header=True, index=True)

        print('DONE!')


# -------------------------------------------------- INPUT --------------------------------------------------

try:
    sc_from = sys.argv[1]
    process = Process_dataset()
    process.run(sc_from)

except EOFError:
    print('ERROR!')