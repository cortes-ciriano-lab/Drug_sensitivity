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
test_f1_total = []
train_f1_total = []
loss_params = []

check = []
data_from = sys.argv[1]
type_network = sys.argv[2]
combination = sys.argv[3]
files = open('loss_results_{}_{}_{}.txt'.format(data_from, type_network, combination),'r')
files = files.readlines()

if type_network == "NNet":
    for file in files:
        print(file)
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
            assert training_loss
            assert validation_loss
            assert test_loss
            assert train_corr
            assert validation_corr
            assert test_corr
            
            validation_loss_total.append(validation_loss)
            validation_corr_total.append(validation_corr)
            train_loss_total.append(training_loss)
            test_loss_total.append(test_loss)
            
            train_corr_total.append(train_corr)
            test_corr_total.append(test_corr)
            
            loss_params.append(file.split('/')[-1])
            
            del training_loss
            del validation_loss
            del test_loss
            del train_corr
            del validation_corr
            del test_corr
            
        except:
            check.append(file)
        
    
    d = pd.DataFrame(validation_loss_total, columns = ['Val_loss_total'])
    d['Train_loss_total'] = train_loss_total
    d['Test_loss_total'] = test_loss_total
    d['Train_corr_total'] = train_corr_total
    d['Val_corr_total'] = validation_corr_total
    d['Test_corr_total'] = test_corr_total
    d['Difference'] = np.abs(d['Train_loss_total'] - d['Val_loss_total'])
    d['Parameters'] = loss_params
    d.dropna(inplace=True)
    d = d.sort_values(['Val_loss_total', 'Val_corr_total'], ascending=[True, False])

elif type_network == 'RF':
    for file in files:
        print(file)
        values = open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}'.format(file.strip('\n')), 'r')
        values = values.readlines()
        i = 0
        while i < len(values):
            line = values[i].strip('\n').split(': ')
            if line[0] == 'Number of trees':
                train_line = values[i+2].strip('\n').split(': ')
                assert train_line[0] == 'Precision'
                train_precision = float(train_line[-1])
                
                train_line = values[i+3].strip('\n').split(': ')
                assert train_line[0] == 'Recall'
                train_recall = float(train_line[-1])
                
                train_line = values[i+4].strip('\n').split(': ')
                assert train_line[0] == 'F1 score'
                train_f1 = float(train_line[-1])
                
                test_line = values[i+8].strip('\n').split(': ')
                assert test_line[0] == 'Precision'
                test_precision = float(test_line[-1])
                
                test_line = values[i+9].strip('\n').split(': ')
                assert test_line[0] == 'Recall'
                test_recall = float(test_line[-1])
                
                test_line = values[i+10].strip('\n').split(': ')
                assert test_line[0] == 'F1 score'
                test_f1 = float(test_line[-1])
                break
            
            else:
                i+=1
        
        try:
            assert train_precision
            assert train_recall
            assert train_f1
            assert test_precision
            assert test_recall
            assert test_f1
            
            train_loss_total.append(train_precision)
            train_corr_total.append(train_recall)
            train_f1_total.appemd(train_f1)
            test_loss_total.append(test_precision)
            test_corr_total.append(test_recall)
            test_f1_total.append(test_f1)
            loss_params.append(file.split('/')[-1])
            
            del train_precision
            del train_recall
            del train_f1
            del test_precision
            del test_recall
            del test_f1
            
        except:
            check.append(file)
    d = pd.DataFrame(test_corr_total, columns = ['Test_recall'])
    d['Test_precision'] = test_loss_total
    d['Test_F1'] = test_f1_total
    d['Train_recall'] = train_corr_total
    d['Train_precision'] = train_loss_total
    d['Train_F1'] = train_f1_total
    d['Parameters'] = loss_params
    d.dropna(inplace=True)
    d = d.sort_values(['Test_F1', 'Test_recall', 'Test_precision'], ascending = False)
else:
    for file in files:
        print(file)
        values = open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}'.format(file.strip('\n')), 'r')
        values = values.readlines()
        i = 0 #the results start in line 20
        while i < len(values):
            line = values[i].strip('\n').split(': ')
            
            if line[0] == 'Testing loss':
                test_loss = float(line[-1])
            
            elif line[0] == 'Training correlation':
                train_corr = float(line[-1])
            
            elif line[0] == 'Testing correlation':
                test_corr = float(line[-1])
            
            i+=1
        
        try:
            assert test_loss
            assert train_corr
            assert test_corr
            
            test_loss_total.append(test_loss)
            train_corr_total.append(train_corr)
            test_corr_total.append(test_corr)
            loss_params.append(file.split('/')[-1])
            
            del test_loss
            del train_corr
            del test_corr
            
        except:
            check.append(file)
    
    d = pd.DataFrame(test_loss_total, columns = ['Test_loss_total'])
    d['Train_corr_total'] = train_corr_total
    d['Test_corr_total'] = test_corr_total
    d['Parameters'] = loss_params
    d.dropna(inplace=True)
    d = d.sort_values(['Test_loss_total', 'Test_corr_total'])
    
if len(check) >= 1:
    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/check_cases_{}_{}_{}.txt'.format(data_from, type_network, combination), 'w') as f:
        f.write('\n'.join(check))

print(d)
d.to_csv('summary_results_{}_{}_{}.csv'.format(data_from, type_network, combination), header=True, index=False)