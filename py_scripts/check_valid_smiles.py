# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import copy
import pickle
from sklearn.utils import shuffle
import gc
import sys
import re
from rdkit import Chem
from standardiser import standardise


from molecular import check_valid_smiles, Molecular
from single_cell import Genexp_sc
from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- FUNCTIONS --------------------------------------------------
ohf = OneHotFeaturizer()
molecules = Molecular()
molecules.set_filename_report('data/10times/molecular_output_10times.txt')
mol_model = molecules.start_molecular()
maximum_length_smiles = int(molecules.get_maximum_length())

drug_sensitivity_metadata = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/prism_metadata.pkl', 'rb'))

with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_screens_once.txt', 'r') as f:
    valid_smiles = f.readlines()

list_smiles = list(drug_sensitivity_metadata['smiles'])
list_smiles = list(set(list_smiles).difference(valid_smiles))
list_smiles = shuffle(list_smiles)
list_new_smiles = []
i = 0
while len(list_new_smiles) < 10:
    if ',' in list_smiles[i]:
        pass
    else:
        list_new_smiles.append(list_smiles[i])
    i += 1

del list_smiles
gc.collect()

list_new_smiles = ohf.featurize(list_new_smiles, maximum_length_smiles)
molecules.run_new_latent_space(mol_model, list_new_smiles, 1000)