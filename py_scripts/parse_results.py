import pandas as pd
import numpy as np

seed = 42
np.random.seed(seed)

files = ["loss_results_pancancer"]
for f in files:
    print(" ".join(f.upper().split("_")[-2:]))
    file_path = "/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}.txt".format(f)
    file = open(file_path,"r")
    file = file.readlines()
    
    validation_loss_total = []
    test_loss_total = []
    train_loss_total = []
    validation_corr_total = []
    test_corr_total = []
    train_corr_total = []
    train_real_range = []
    train_predicted_range = []
    validation_real_range = []
    validation_predicted_range = []
    test_real_range = []
    test_predicted_range = []
    loss_params = []
    
    i = 0
    while i < len(file):
        if "Training loss" in file[i] and "Validation loss" in file[i+1] and "Testing loss" in file[i+2] and "Training correlation" in file[i+3] and "Validation correlation" in file[i+4] and "Testing correlation" in file[i+5]:
            train_loss_line = file[i].strip("\n").split(': ')
            validation_loss_line = file[i+1].strip("\n").split(': ')
            test_loss_line = file[i+2].strip("\n").split(': ')
            train_corr_line = file[i+3].strip("\n").split(': ')
            validation_corr_line = file[i+4].strip("\n").split(': ')
            test_corr_line = file[i+5].strip("\n").split(': ')
            train_real = '{} - {}'.format(file[i+7].strip("\n").split(': '), file[i+6].strip("\n").split(': '))
            train_predicted = '{} - {}'.format(file[i+9].strip("\n").split(': '), file[i+8].strip("\n").split(': '))
            validation_real = '{} - {}'.format(file[i+11].strip("\n").split(': '), file[i+10].strip("\n").split(': '))
            validation_predicted = '{} - {}'.format(file[i+13].strip("\n").split(': '), file[i+12].strip("\n").split(': '))
            test_real = '{} - {}'.format(file[i+15].strip("\n").split(': '), file[i+14].strip("\n").split(': '))
            test_predicted = '{} - {}'.format(file[i+17].strip("\n").split(': '), file[i+16].strip("\n").split(': '))
            parameters = file[i+18].split("put_")[-1].strip(".txt\n").split("_")
            
            validation_loss_total.append(float(validation_loss_line[-1]))
            train_loss_total.append(float(train_loss_line[-1]))
            test_loss_total.append(float(test_loss_line[-1]))
            
            validation_corr_total.append(float(validation_corr_line[-1]))
            train_corr_total.append(float(train_corr_line[-1]))
            test_corr_total.append(float(test_corr_line[-1]))
            
            validation_real_range.append(validation_real)
            validation_predicted_range.append(validation_predicted)
            train_real_range.append(train_real)
            train_predicted_range.append(train_predicted)
            test_real_range.append(test_real_range)
            test_predicted_range.append(test_predicted)
            
            
            loss_params.append("_".join(parameters))
            
            i += 19
        
        else:
            with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/check_cases.txt', 'a') as f:
                parameters = file[i].split("put_")[-1].strip(".txt\n")
                f.write("_".join(parameters))
                f.write('\n')
            i += 1
    
    d = pd.DataFrame(validation_loss_total, columns = ['Val_loss_total'])
    d['Train_loss_total'] = train_loss_total
    d['Test_loss_total'] = test_loss_total
    d['Train_corr_total'] = train_corr_total
    d['Val_corr_total'] = validation_corr_total
    d['Test_corr_total'] = test_corr_total
    d['Difference'] = np.abs(d['Train_loss_total'] - d['Val_loss_total'])
    d['Train_real_range'] = train_real_range
    d['Test_pred_range'] = train_predicted_range
    d['Val_real_range'] = validation_real_range
    d['Val_pred_range'] = validation_predicted_range
    d['Test_real_range'] = test_real_range
    d['Test_pred_range'] = test_predicted_range
    d['Parameters'] = loss_params
    d = d.sort_values(['Val_corr_total', 'Val_loss_total', "Difference"])
    d.dropna(subset=['Train_corr_total'], inplace=True)
    d.dropna(subset=['Val_corr_total'], inplace=True)
    d.dropna(subset=['Test_corr_total'], inplace=True)
    
    d.to_csv('summary_results.csv', header=True, index=False)
    
    best_parameters = d.head(20)
    # best_parameters.to_csv("/hps/research1/icortes/acunha/python_scripts/single_cell/best_parameters_pancancer_losses.txt")
    # best_parameters = best_parameters.sort_values('Difference')
    print(best_parameters.head(20))
    print(list(best_parameters['Parameters'].head(20)))
    
    new_file = 'pancancer'
    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/list_best_parameters_{}.txt'.format(new_file), 'w') as f:
        f.write(list(best_parameters['Parameters'].head(1))[0])
        f.write('\n')
        f.write(list(best_parameters['Parameters'].head(1))[0])
