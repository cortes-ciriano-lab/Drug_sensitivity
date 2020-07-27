import pandas as pd
import pickle
import numpy as np

single_cell = pd.read_csv("/hps/research1/icortes/DATA/CCLE/single_cell/CPM_data.txt", sep = "\t",header = 0, index_col = 0)
print(single_cell.head())
single_cell = single_cell.transpose()
print(single_cell.head())
#single_cell.index.names = ["CELLS_ID"]
single_cell = np.log2(single_cell + 1)
print(single_cell.head())
#pickle.dump(single_cell, open("/hps/research1/icortes/acunha/data/PANCANCER_PICKLE/single_cell_data.pkl", "wb"), protocol=4)

