# -*- coding: utf-8 -*-

# -------------------------------------------------- IMPORTS --------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------- VARIATIONAL AUTOENCODER --------------------------------------------------

class Single_cell_VAE(nn.Module):
    def __init__(self, **kwargs):
        super(Single_cell_VAE, self).__init__()

        layer1_input = int(kwargs['n_genes'])
        layers = kwargs['layers']
        layers = layers.split("_")
        layer2_hidden = int(layers[0]) 
        layer3_hidden = int(layers[1])
        layer4_bottleneck = int(layers[2])
        self.dropout_prob = float(kwargs['dropout_prob'])

            
        '''Definition of the different layers'''
        self.fc1 = nn.Linear(layer1_input, layer2_hidden) 
        self.fc2 = nn.Linear(layer2_hidden, layer3_hidden)
        self.fc3_mu = nn.Linear(layer3_hidden, layer4_bottleneck)
        self.fc3_var = nn.Linear(layer3_hidden, layer4_bottleneck) 
        self.fc4 = nn.Linear(layer4_bottleneck, layer3_hidden) 
        self.fc5 = nn.Linear(layer3_hidden, layer2_hidden) 
        self.fc6 = nn.Linear(layer2_hidden, layer1_input) 
    
    def encoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_prob)
        x = F.relu(self.fc2(x))
        z_mu = self.fc3_mu(x)
        z_var = self.fc3_var(x)
        return z_mu, z_var
    
    def reparametrize(self, z_mu, z_var):
        if self.training:
            std = torch.exp(z_var/2)
            eps = torch.randn_like(std) * 1e-2
            x_sample = eps.mul(std).add_(z_mu)
            return x_sample
        else:
            return z_mu
    
    def decoder(self, z):
        z = F.relu(self.fc4(z))
        z = F.dropout(z, self.dropout_prob)
        z = F.relu(self.fc5(z))
        z = F.dropout(z, self.dropout_prob)
        z = self.fc6(z)
        return z
    
    def forward(self, x):
        z_mu, z_var = self.encoder(x)
        x_sample = self.reparametrize(z_mu, z_var)
        output = self.decoder(x_sample)
        return output, x_sample, z_mu, z_var