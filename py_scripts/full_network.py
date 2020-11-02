# -*- coding: utf-8 -*-

# -------------------------------------------------- IMPORTS --------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
torch.manual_seed(seed)
    
# -------------------------------------------------- FULL NETWORK --------------------------------------------------

class NN_drug_sensitivity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        #Defining the sizes of the layers
        self.size_input = int(kwargs['input_size']) #size of the input
        layers = kwargs['layers']
        layers.insert(0, self.size_input)
        self.layers = []
        i = 0
        while (i+1) < len(layers):
            self.layers.append([layers[i],layers[i+1]])
            i += 1
        self.dropout_prob = float(kwargs['dropout_prob'])

        #Definition of the network
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.layers)):
            self.fc_layers.append(nn.Linear(int(self.layers[i][0]), int(self.layers[i][1])))

    def forward(self, x):
        for i, l in enumerate(self.fc_layers):
            x = l(x)
            if i != len(self.layers)-1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_prob, inplace = True)
        return x

#-------------------------------------------------- VAE - GENE EXPRESSION - SINGLE CELL --------------------------------------------------

class VAE_gene_expression_single_cell(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

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
        std = torch.exp(z_var/2)
        eps = torch.randn_like(std) * 1e-2
        x_sample = eps.mul(std).add_(z_mu)
        return x_sample
    
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

# -------------------------------------------------- VAE - DRUGS --------------------------------------------------

class VAE_molecular(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        '''Defining the sizes of the different outputs
        Convolution layers
        Input: (N batch size, Cin number of channels, Lin length of signal sequence)
        Output: (N, Cout, Lout)
        where Lout = [ [ Lin + 2 * padding{in this case 0} - dilation{in this case 1} * (kernel_size - 1) - 1 ] / stride{in this case 1} ] + 1 
        '''
        self.cin = int(kwargs['number_channels_in'])
        self.lin = float(kwargs['length_signal_in'])
        self.dropout_prob = float(kwargs['dropout_prob'])
        
            
        '''Definition of the different layers'''
        self.conv1 = nn.Conv1d(self.cin, 9, kernel_size = 9)
        lout = ((self.lin + 2.0 * 0.0 - 1.0 * (9.0 - 1.0) - 1.0 ) / 1.0 ) + 1.0
        
        self.conv2 = nn.Conv1d(9, 9, kernel_size=9)
        lout = ((lout + 2.0 * 0.0 - 1.0 * (9.0 - 1.0) - 1.0 ) / 1.0 ) + 1.0
        
        self.conv3 = nn.Conv1d(9, 10, kernel_size=11)
        lout = ((lout + 2.0 * 0.0 - 1.0 * (11.0 - 1.0) - 1.0 ) / 1.0 ) + 1.0
        
        self.fc1 = nn.Linear(10 * int(lout), 425) #the input is the channels from the previous layers * lout
        self.fc2_mu = nn.Linear(425, 292)
        self.fc2_var = nn.Linear(425, 292)
        self.fc3 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first = True)
        self.fc4 = nn.Linear(501, int(self.lin))
    
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))
        x = F.dropout(x, self.dropout_prob)
        z_mu = self.fc2_mu(x)
        z_var = self.fc2_var(x)
        return z_mu, z_var
    
    def reparametrize(self, z_mu, z_var):
        std = torch.exp(z_var/2)
        eps = torch.randn_like(std) * 1e-2
        x_sample = eps.mul(std).add_(z_mu)
        return x_sample
    
    def decoder(self, z):
        z = F.selu(self.fc3(z))
        z = F.dropout(z, self.dropout_prob)
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.cin, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc4(out_reshape), dim = 1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y
    
    def forward(self, x):
        z_mu, z_var = self.encoder(x)
        x_sample = self.reparametrize(z_mu, z_var)
        output = self.decoder(x_sample)
        return output, x_sample, z_mu, z_var
