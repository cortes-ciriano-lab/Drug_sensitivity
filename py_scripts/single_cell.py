# -------------------------------------------------- IMPORTS --------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import pandas as pd
import pickle
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torch.utils.data
import sys
import gc
from rdkit import Chem
from sklearn.utils import shuffle 

from full_network import VAE_gene_expression_single_cell

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- ANOTHER FUNCTIONS --------------------------------------------------

def create_report(filename, list_comments):
    with open("/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}".format(filename), 'a') as f:
        f.write('\n'.join(list_comments))

# -------------------------------------------------- SINGLE CELL --------------------------------------------------

class Genexp_sc():

    def __init__(self):
        self.learning_rate = None
        self.size_batch = None
        self.n_epochs = None
        self.dropout = None
        self.gamma = None
        self.step_size = None
        self.epoch_reset = None
        self.seed = None
        self.layers = None
        self.alpha = None

        self.device = None
        
        self.filename_report = None
        
        self.path = None

    # --------------------------------------------------

    def __set_parameters(self, list_parameters):
        self.learning_rate = float(list_parameters[0])
        self.size_batch = int(list_parameters[1])
        self.n_epochs = int(list_parameters[2])
        self.dropout = float(list_parameters[5])
        self.gamma = float(list_parameters[6])
        self.step_size = int(list_parameters[7])
        self.seed = int(list_parameters[8])
        self.epoch_reset = int(list_parameters[9])
        self.layers = list_parameters[13]
        self.alpha = float(list_parameters[14])
    
    # --------------------------------------------------
        
    def __load_initial_parameters(self):
        list_parameters = pickle.load(open('{}/list_initial_parameters_single_cell.pkl'.format(self.path), 'rb'))
        self.__set_parameters(list_parameters)
        self.device = list_parameters[-1]
        
        lines = ["** REPORT - GENE EXPRESSION - SINGLE CELL DATA **\n",
                "* Parameters",
                "Learning rate: {} ; Size batch: {} ; Number of epochs: {} ; Dropout: {} ; Gamma: {} ;".format(self.learning_rate, self.size_batch, self.n_epochs,self.dropout, self.gamma),
                "Step size: {} ; Seed: {} ; Epoch to reset: {}".format(self.step_size, self.seed, self.epoch_reset)]
        
        lines.extend(["*About the network", "Layers: {}".format(self.layers), "Runs on: {}".format(self.device), "\n"])

        create_report(self.filename_report, lines)

        global seed
        if seed != self.seed:
            seed = self.seed
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
    # --------------------------------------------------

    def __initialize_model(self, num_genes):
        model = VAE_gene_expression_single_cell(dropout_prob=self.dropout, n_genes=num_genes, layers=self.layers)
        model.to(self.device)
        
        return model

    # --------------------------------------------------

    def __loss_function(self, x_input, x_output, z_mu, z_var):
        criterion = nn.MSELoss()
        reconstruction_loss = criterion(x_output, x_input)  # current loss value
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu.pow(2) - 1.0 - z_var)
        return torch.mean(reconstruction_loss + (self.alpha * kl_loss)), reconstruction_loss, kl_loss
    
    # --------------------------------------------------
 
    def __load_model(self, model):
        model_parameters = pickle.load(open("{}/single_cell_model.pkl".format(self.path), 'rb'))
        model.load_state_dict(model_parameters)
        return model
    
    # --------------------------------------------------
    
    def start_expression(self, num_genes, path_model):
        self.path = path_model
        self.__load_initial_parameters()
        model = self.__initialize_model(num_genes)
        model = self.__load_model(model)
        return model
    
    # --------------------------------------------------
    
    def run_dataset(self, model_trained, dataset, type_data):
        dataset_torch = torch.tensor(dataset).type('torch.FloatTensor')
        data_loader = torch.utils.data.DataLoader(dataset_torch, batch_size=self.size_batch, shuffle=False)
        
        lines = ["* {} *".format(type_data),
                "Rows: {} ; Columns: {} ".format(dataset_torch.shape[0], dataset_torch.shape[1]),
                "\n"] 
        create_report(self.filename_report, lines)
        
        del dataset_torch
        gc.collect()

        total_loss = 0.0
        reconstruction_loss = 0.0
        kl_loss = 0.0
        
        predictions_complete, bottleneck_complete = [], []
        model_trained.eval()
        with torch.no_grad():
            for data_batch in data_loader:
                data_batch = data_batch.to(self.device)
                data_predictions = model_trained(data_batch)  # output predicted by the model
                current_loss = self.__loss_function(data_batch, data_predictions[0], data_predictions[2], data_predictions[3])
                total_loss += current_loss[0].item()
                reconstruction_loss += current_loss[1].item()
                kl_loss += current_loss[2].item()
                predictions_complete.extend(list(data_predictions[0].cpu().numpy()))
                bottleneck_complete.extend(list(data_predictions[1].cpu().numpy()))
                
        
        lines = ["Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f} \n".format(total_loss, reconstruction_loss, kl_loss),
                "\n"] 
        create_report(self.filename_report, lines)
        
        return predictions_complete, bottleneck_complete, total_loss, reconstruction_loss, kl_loss
    
    # --------------------------------------------------
    
    def set_filename_report(self, filename):
        self.filename_report = filename