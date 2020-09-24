import pandas as pd
import numpy as np
import sys
import pickle

seed = 42
np.random.seed(seed)

validation_loss_total = []
test_loss_total = []
train_loss_total = []
validation_corr_total = []
test_corr_total = []
train_corr_total = []
loss_params = []

check = []
type_data = sys.argv[1]
data_from = sys.argv[2]
files = open('loss_results_{}_{}.txt'.format(type_data, data_from),'r')
files = files.readlines()
for file in files:
    values = open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}'.format(file.strip('\n')), 'r')
    values = values.readlines()
    i = 0 #the results start in line 20
    while i < len(values):
        line = values[i].strip('\n').split(': ')
        
        if line[0] == 'Training loss':
            training_loss = float(line[-1])
        
        elif line[0] == 'Validation loss':
            validation_loss = float(line[-1])
        
        elif line[0] == 'Testing loss':
            test_loss = float(line[-1])
        
        elif line[0] == 'Training correlation':
            train_corr = float(line[-1])
        
        elif line[0] == 'Validation correlation':
            validation_corr = float(line[-1])
        
        elif line[0] == 'Testing correlation':
            test_corr = float(line[-1])
        
        i+=1
    
    try:
            validation_loss_total.append(validation_loss)
            train_loss_total.append(training_loss)
            test_loss_total.append(test_loss)
            
            validation_corr_total.append(validation_corr)
            train_corr_total.append(train_corr)
            test_corr_total.append(test_corr)
            
            loss_params.append(file.split('/')[-1])
        
    except:
        check.append(file)

with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/check_cases_{}_{}.txt'.format(type_data, data_from), 'w') as f:
    f.write('\n'.join(check))

d = pd.DataFrame(validation_loss_total, columns = ['Val_loss_total'])
d['Train_loss_total'] = train_loss_total
d['Test_loss_total'] = test_loss_total
d['Train_corr_total'] = train_corr_total
d['Val_corr_total'] = validation_corr_total
d['Test_corr_total'] = test_corr_total
d['Difference'] = np.abs(d['Train_loss_total'] - d['Val_loss_total'])
d['Parameters'] = loss_params
d.dropna(inplace=True)
d = d.sort_values(['Val_loss_total'])
d.to_csv('summary_results_{}_{}.csv'.format(type_data, data_from), header=True, index=False)

indexes_to_keep = []
for i in list(d.index):
    # if d['Val_loss_total'].loc[i] < 0.58 or '0.00001' in d['Parameters'].loc[i]:
    if d['Val_loss_total'].loc[i] <= 0.05:
        indexes_to_keep.append(i)
best_parameters = d.loc[indexes_to_keep]
print(best_parameters)

jobs_already_resumed = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/jobs_resumed.pkl', 'rb'))
jobs_to_resume = []
for file in list(best_parameters['Parameters']):
    new_file = file.strip('output_')[:-5]
    # new_file = new_file.strip('.txt')
    print(new_file)
    new_file = 'new_results/{}/{}/{}'.format(type_data, data_from, new_file)
    if new_file not in jobs_already_resumed:
        jobs_to_resume.append(new_file)
        jobs_already_resumed.append(new_file)

with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/jobs_to_resume.txt', 'w') as f:
    f.write('\n'.join(jobs_to_resume))
pickle.dump(jobs_already_resumed, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/jobs_resumed.pkl', 'wb'))