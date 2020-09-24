# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import gc
import sys
from rdkit import Chem
from standardiser import standardise
from rdkit.Chem import AllChem

from molecular import check_valid_smiles, Molecular
from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------
seed = 42
np.random.seed(seed)

# -------------------------------------------------- PROCESS DATASETS  --------------------------------------------------
def create_prism_bottleneck_run_once():
    ohf = OneHotFeaturizer()
    
    #initialize the single cell model
    print('Molecular model: started \n ')
    molecules = Molecular()
    molecules.set_filename_report('data/molecular/run_once/molecular_output.txt')
    mol_model = molecules.start_molecular()
    maximum_length_smiles = int(molecules.get_maximum_length())
    
    drug_sensitivity_metadata = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/prism_metadata.pkl', 'rb'))
    
    mols = []
    mol_indexes = []
    fingerprints = []
    for i in range(len(list(drug_sensitivity_metadata['smiles']))):
        s = drug_sensitivity_metadata['smiles'].iloc[i]
        if ',' in s: #means that exists more than one smile representation of the compound
            if '\"' in s:
                s = s.strip('\"')
            s = s.split(', ')
        else:
            s = [s]
        for j in range(len(s)):
            mols.append(s[j])
            m = Chem.MolFromSmiles(s[j])
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits = 1024)
            fingerprints.append(np.array(fp))
            if len(s) > 1:
                mol_indexes.append('{}:::{}'.format(drug_sensitivity_metadata.iloc[i].name, j))
            else:
                mol_indexes.append(drug_sensitivity_metadata.iloc[i].name)
    
    del drug_sensitivity_metadata
    gc.collect()
    
    for i in range(len(mol_indexes)):
        index = mol_indexes[i]
        smile = mols[i]
        fp = fingerprints[i]
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/prism_indexes_smiles.txt', 'a') as f:
            f.write('{}, {}\n'.format(index, smile))
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/prism_indexes_morgan_fingerprints.txt', 'a') as f:
            f.write('{}, {}\n'.format(index, fp))
            
    
    mols = ohf.featurize(mols, maximum_length_smiles)
    if True in np.isnan(np.array(mols)):
        print('there are nan in the dataset!! \n ')
        sys.exit()
    mol_predictions = molecules.run_dataset(mol_model, mols)
    
    free_memory = [mol_model, mols]
    for item in free_memory:
        del item
    gc.collect()
    
    mol_outputs = {}
    for i in range(len(mol_predictions[0])):
        mol_outputs[mol_indexes[i]] = mol_predictions[0][i]
    pickle.dump(mol_outputs, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_once/pkl_files/prism_outputs.pkl', 'wb'), protocol=4)
     
    del mol_outputs
    gc.collect()
    
    mol_bottlenecks = pd.DataFrame(mol_predictions[1])
    mol_bottlenecks.index = list(mol_indexes)
    # mol_bottlenecks = mol_bottlenecks.drop(mol_bottlenecks.index[mol_predictions[2]])
    list_indexes = list(mol_bottlenecks.index)
    
    print('PRISM BOTTLENECK \n', mol_bottlenecks.shape)
    
    pickle.dump(mol_bottlenecks, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_once/pkl_files/prism_bottlenecks.pkl', 'wb'), protocol=4)
    mol_bottlenecks.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_once/prism_bottlenecks.csv', header=True, index=False)    

    mol_bottlenecks.set_index(list(mol_bottlenecks.columns)[0])
    return mol_bottlenecks, list_indexes

def create_prism_bottleneck_run_secondary(values_from):
    ohf = OneHotFeaturizer()

    # initialize the single cell model
    print('Molecular model: started \n ')
    molecules = Molecular()
    molecules.set_filename_report('/data_secondary/{}/molecular/run_once/molecular_output_secondary.txt'.format(values_from))
    mol_model = molecules.start_molecular()
    maximum_length_smiles = int(molecules.get_maximum_length())

    drug_sensitivity = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/{}/pkl_files/prism_dataset.pkl'.format(values_from), 'rb'))

    mols = []
    mol_indexes = []
    fingerprints = []
    for i in range(len(list(drug_sensitivity['smiles']))):
        s = drug_sensitivity['smiles'].iloc[i]
        if ',' in s:  # means that exists more than one smile representation of the compound
            if '\"' in s:
                s = s.strip('\"')
            s = s.split(', ')
        else:
            s = [s]
        for j in range(len(s)):
            mols.append(s[j])
            m = Chem.MolFromSmiles(s[j])
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            fingerprints.append(np.array(fp))
            if len(s) > 1:
                mol_indexes.append('{}:::{}'.format(drug_sensitivity.iloc[i].name, j))
            else:
                mol_indexes.append(drug_sensitivity.iloc[i].name)

    del drug_sensitivity
    gc.collect()

    for i in range(len(mol_indexes)):
        index = mol_indexes[i]
        smile = mols[i]
        fp = fingerprints[i]
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/{}/molecular/prism_indexes_smiles.txt'.format(values_from), 'a') as f:
            f.write('{}, {}\n'.format(index, smile))
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/{}/molecular/prism_indexes_morgan_fingerprints.txt'.format(values_from), 'a') as f:
            f.write('{}, {}\n'.format(index, fp))

    mols = ohf.featurize(mols, maximum_length_smiles)
    if True in np.isnan(np.array(mols)):
        print('there are nan in the dataset!! \n ')
        sys.exit()
    mol_predictions = molecules.run_dataset(mol_model, mols)

    free_memory = [mol_model, mols]
    for item in free_memory:
        del item
    gc.collect()

    mol_outputs = {}
    for i in range(len(mol_predictions[0])):
        mol_outputs[mol_indexes[i]] = mol_predictions[0][i]
    pickle.dump(mol_outputs, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/{}/molecular/run_once/pkl_files/prism_outputs.pkl'.format(values_from), 'wb'), protocol=4)

    del mol_outputs
    gc.collect()

    mol_bottlenecks = pd.DataFrame(mol_predictions[1])
    mol_bottlenecks.index = list(mol_indexes)

    print('PRISM BOTTLENECK \n', mol_bottlenecks.shape)

    pickle.dump(mol_bottlenecks, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/{}/molecular/run_once/pkl_files/prism_bottlenecks.pkl'.format(values_from), 'wb'), protocol=4)
    mol_bottlenecks.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/{}/molecular/run_once/prism_bottlenecks.csv'.format(values_from), header=True, index=False)

    mol_bottlenecks.set_index(list(mol_bottlenecks.columns)[0])
    return mol_bottlenecks, mol_indexes

def create_prism_bottleneck_only_valids(times):
    ohf = OneHotFeaturizer()
    
    #initialize the single cell model
    print('Molecular model: started \n ')
    molecules = Molecular()
    molecules.set_filename_report('/data/molecular/run_{}/molecular_output.txt'.format(times))
    mol_model = molecules.start_molecular()
    maximum_length_smiles = int(molecules.get_maximum_length())
    
    drug_sensitivity_metadata = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/prism_metadata.pkl', 'rb'))
    
    mols = []
    mol_indexes = []
    for i in range(len(list(drug_sensitivity_metadata['smiles']))):
        s = drug_sensitivity_metadata['smiles'].iloc[i]
        if ',' in s: #means that exists more than one smile representation of the compound
            if '\"' in s:
                s = s.strip('\"')
            s = s.split(', ')
        else:
            s = [s]
        for j in range(len(s)):
            mols.append(s[j])
            if len(s) > 1:
                mol_indexes.append('{}:::{}'.format(drug_sensitivity_metadata.iloc[i].name, j))
            else:
                mol_indexes.append(drug_sensitivity_metadata.iloc[i].name)
    
    del drug_sensitivity_metadata
    gc.collect()
    
    mols = ohf.featurize(mols, maximum_length_smiles)
    if True in np.isnan(np.array(mols)):
        print('there are nan in the dataset!! \n ')
        sys.exit()
    mol_predictions = molecules.run_only_valids(mol_model, mols, times, mol_indexes)
    
    free_memory = [mol_model, mols]
    for item in free_memory:
        del item
    gc.collect()
    
    mol_outputs = {}
    for i in range(len(mol_predictions[0])):
        if str(mol_predictions[0][i]) != 'nan':
            mol_outputs[mol_indexes[i]] = mol_predictions[0][i]
    pickle.dump(mol_outputs, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_{}/pkl_files/prism_outputs.pkl'.format(times), 'wb'), protocol=4)
     
    del mol_outputs
    gc.collect()
    
    mol_bottlenecks = pd.DataFrame(mol_predictions[1])
    mol_bottlenecks.index = mol_indexes
    mol_bottlenecks = mol_bottlenecks.loc[mol_bottlenecks.index.isin(mol_predictions[2])]
    list_indexes = list(mol_bottlenecks.index)
    
    print('PRISM BOTTLENECK \n', mol_bottlenecks.shape)
    
    pickle.dump(mol_bottlenecks, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_{}/pkl_files/prism_bottlenecks.pkl'.format(times), 'wb'), protocol=4)
    mol_bottlenecks.reset_index().to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_{}/prism_bottlenecks.csv'.format(times), header=True, index=False)    

    mol_bottlenecks.set_index(list(mol_bottlenecks.columns)[0])
    return mol_bottlenecks, list_indexes

# _ = create_prism_bottleneck()

# _ = create_prism_bottleneck()