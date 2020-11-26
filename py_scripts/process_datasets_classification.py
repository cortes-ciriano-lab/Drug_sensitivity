# -------------------------------------------------- IMPORTS --------------------------------------------------

import pickle

# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------

new_indexes_dict = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/ic50/prism_pancancer/prism_pancancer_new_indexes_newIndex2barcodeScreen_dict.pkl', 'rb'))

new_dict = {}
for k in new_indexes_dict.keys():
    cells_info, compound_info, sens_value = new_indexes_dict[k]
    if sens_value <= 7:
        label = 'Inactive'
    else:
        label = 'Active'
    new_dict[k] = (cells_info, compound_info, label)

pickle.dump(new_dict, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_secondary/ic50/prism_pancancer/prism_pancancer_new_indexes_newIndex2barcodeScreen_dict_classification.pkl', 'wb'))