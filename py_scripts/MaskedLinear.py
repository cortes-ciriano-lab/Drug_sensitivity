# -------------------------------------------------- IMPORTS --------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# -------------------------------------------------- DEFINE LAYER --------------------------------------------------

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, relation_file, bias = True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        mask = self.read_relation_from_file(relation_file)
        self.register_buffer('mask', mask)

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def read_relation_from_file(self, relation_file):
        mask = pd.read_csv(relation_file, header = None).T
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(mask.to_numpy()).to(device)