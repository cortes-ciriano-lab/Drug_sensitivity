# -------------------------------------------------- IMPORTS --------------------------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import pickle
import time
from sklearn.utils import shuffle
import gc
import sys
import re
from rdkit import Chem
import torch.utils.data
import matplotlib.pyplot as plt
from standardiser import standardise
import seaborn as sns
from torch.utils.data import TensorDataset
from scipy.stats import pearsonr
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error

from full_network import NN_drug_sensitivity, VAE_molecular, VAE_gene_expression_single_cell, AE_gene_expression_bulk
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# -------------------------------------------------- ANOTHER FUNCTIONS --------------------------------------------------

def create_report(filename, list_comments):
    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/{}'.format(filename), 'a') as f:
        f.write('\n'.join(list_comments))

# --------------------------------------------------

def make_correlation_plot(real_data, predicted_data, indexes, type_data):
    d = {'Real_data' : real_data, 'Predicted_data' : predicted_data}
    data = pd.DataFrame.from_dict(d)
    data.index = indexes
    
    plt.figure(figsize=(15, 15))
    data.plot(x='Real_data', y='Predicted_data', style=['o'])
    plt.title('Real values vs Predicted values - {}'.format(type_data), fontsize=14, fontweight='bold')
    plt.xlabel('Real_data')
    plt.ylabel('Predicted_data')
    plt.savefig('plots/Scatter_real_vs_predicted_values_{}.png'.format(type_data))

# --------------------------------------------------

dict_optimisers = {'adagrad':optim.Adagrad(model.parameters(), lr=self.learning_rate),
                   'adam':optim.Adam(model.parameters(), lr=self.learning_rate),
                   'adamw':optim.AdamW(model.parameters(), lr=self.learning_rate),
                   'sparseadam':optim.SparseAdam(model.parameters(), lr=self.learning_rate),
                   'adamax':optim.Adamax(model.parameters(), lr=self.learning_rate),
                   'asgd':optim.ASGD(model.parameters(), lr=self.learning_rate),
                   'lbfgs':optim.LBFGS(model.parameters(), lr=self.learning_rate),
                   'rmsprop':optim.RMSprop(model.parameters(), lr=self.learning_rate),
                   'rprop':optim.Rprop(model.parameters(), lr=self.learning_rate),
                   'sgd':optim.SGD(model.parameters(), lr=self.learning_rate)}

# -------------------------------------------------- DRUG SENSITIVITY --------------------------------------------------

class Drug_sensitivity_single_cell:
    def __init__(self):
        #gene expression
        self.data_from = None
        self.type_data = None
        
        #drug sensitivity
        self.model_architecture = None

        # if NNet
        self.optimiser = None
        self.activation_function = None
        self.layers_info = None

        # if RF
        self.number_trees = None
        
        self.learning_rate = None
        self.size_batch = None
        self.n_epochs = None
        self.dropout = None
        self.gamma = None
        self.step_size = None
        self.epoch_reset = None
        self.seed = None
        self.perc_train = None
        self.perc_val = None
        self.device = None
        self.type_of_split = None
        self.to_test = None
        
        self.filename_report = None

    # --------------------------------------------------

    def set_parameters(self, list_parameters):
        self.type_data = list_parameters[0]
        self.model_architecture = list_parameters[15]

        if self.model_architecture == 'NNet':
            network_info = list_parameters[1].split('_')
            self.optimiser = dict_optimisers[network_info[-1]]
            network_info.pop()
            self.activation_function = network_info[-1]
            network_info.pop()
            layers = {}
            for i in range(len(network_info)):
                if i == 0:
                    layers[str(i+1)] = [network_info[i]]
                elif i == len(network_info) - 1:
                    layers[str(i+2)] = [network_info[i], 1]
                else:
                    layers[str(i + 1)] = [network_info[i-1], network_info[i]]
            self.layers_info = layers

        else:
            self.number_trees = int(list_parameters[1])

        self.learning_rate = float(list_parameters[2])
        self.size_batch = int(list_parameters[3])
        self.n_epochs = int(list_parameters[4])
        self.perc_train = float(list_parameters[5])
        self.perc_val = float(list_parameters[6])
        self.dropout = float(list_parameters[7])
        self.gamma = float(list_parameters[8])
        self.step_size = int(list_parameters[9])
        self.seed = int(list_parameters[10])
        self.epoch_reset = int(list_parameters[11])
        self.type_of_split = list_parameters[12]
        self.to_test = list_parameters[13]
        self.data_from = list_parameters[14]
        self.network_info = list_parameters[16]
        
        #add information to report
        lines = ['** REPORT - DRUG SENSITIVITY **\n',
                '* Parameters',
                'Type of data: {}'.format(self.type_data),
                'Learning rate: {} ; Size batch: {} ; Number of epochs: {} ; Dropout: {} ; Gamma: {} ;'.format(self.learning_rate, self.size_batch, self.n_epochs,self.dropout, self.gamma),
                'Step size: {} ; Seed: {} ; Epoch to reset: {}'.format(self.step_size, self.seed, self.epoch_reset),
                'Data from: {}'.format(self.data_from),
                'Type of split: {}'.format(self.type_of_split)]

        if self.type_of_split == 'random':
            self.to_test = float(self.to_test)
            lines.extend(['Perc. of train: {}% ; Perc of validation: {}% ; Perc of test: {}% \n'.format(int(self.perc_train*100), int(self.perc_val*100), int(self.to_test*100))])
        else:
            lines.extend(['What to test: {} \n'.format(self.to_test)])
        create_report(self.filename_report, lines)

    # --------------------------------------------------
    
    def load_datasets(self):
        if self.data_from == 'pancancer':
            with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_new_indexes_once.txt', 'r') as f:
                list_new_indexes = f.readlines()
                        
            if self.type_of_split == 'random':
                barcodes_per_tumour = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/barcodes_per_tumour_dict.pkl', 'rb'))
                
                #tumours to use ->  Lung_Cancer and Breast_Cancer
                list_tumours = ['Lung_Cancer']#, 'Breast_Cancer']
                list_barcodes = []
                for i in list_tumours:
                    list_barcodes.extend(barcodes_per_tumour[i])
                print(len(list_barcodes))
                list_final_indexes = []
                for i in range(len(list_new_indexes)):
                    barcode = list_new_indexes[i].strip('\n').split('::')[0]
                    if barcode in list_barcodes:
                        list_final_indexes.append(list_new_indexes[i])
                
                free_memory = [list_new_indexes, list_barcodes]
                for item in free_memory:
                    del item
                gc.collect()
                
                list_final_indexes = shuffle(list_final_indexes)
                
                if len(list_final_indexes) > 50000:
                    list_final_indexes = list_final_indexes[:50000]
                
                # list_final_indexes = list_final_indexes[:1000]
                
                
                train_number = int(self.perc_train * len(list_final_indexes))
                validation_number = int(self.perc_val * len(list_final_indexes))
                
                train_set = list_final_indexes[:train_number]
                validation_set = list_final_indexes[train_number:train_number+validation_number]
                test_set = list_final_indexes[train_number+validation_number:]
        
            elif self.type_of_split == 'leave-one-cell-line-out':
                with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_cell_lines_pancancer.txt', 'r') as f:
                    list_cell_lines = f.readlines()
                list_cell_lines.remove(self.to_test)
                
                cell_line_barcode = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/cell_line_barcode_dict.pkl', 'rb'))
                
                train_cells = list_cell_lines[:-1]
                validation_cell = list_cell_lines[-1]
                
                del cell_line_barcode
                gc.collect()
                
                train_set = []
                validation_set = []
                test_set = []
                for i in range(len(list_new_indexes)):
                    barcode = list_new_indexes[i].strip('\n').split('::')[0]
                    cell_line = cell_line_barcode[barcode]
                    if cell_line in train_cells:
                        train_set.append(list_new_indexes[i])
                    elif cell_line in validation_cell:
                        validation_set.append(list_new_indexes[i])
                    elif cell_line == self.to_test:
                        test_set.append(list_new_indexes[i])
                        
                free_memory = [train_cells, validation_cells, cell_line_barcode]
                for item in free_memory:
                    del item
                gc.collect()
                    

            elif self.type_of_split == 'leave-one-tumour-out':
                with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_tumours.txt', 'r') as f:
                    list_tumours = f.readlines()
                list_tumours.remove(self.to_test)
                
                metadata_single = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/pancancer_metadata.pkl', 'wb'))
                
                train_tumours = list_tumours[:-1]
                validation_tumour = list_tumours[-1]
                
                del list_tumours
                gc.collect()
                
                train_set = []
                validation_set = []
                test_set = []
                for i in range(len(list_new_indexes)):
                    barcode = list_new_indexes[i].strip('\n').split('::')[0]
                    tumour = metadata_single.loc[barcode, 'Cancer_type']
                    if tumour in train_tumours:
                        train_set.append(list_new_indexes[i])
                    elif tumour in validation_tumour:
                        validation_set.append(list_new_indexes[i])
                    elif tumour == self.to_test:
                        test_set.append(list_new_indexes[i])
                
                free_memory = [metadata_single, train_tumours, validation_tumour]
                for item in free_memory:
                    del item
                gc.collect()
        
        pickle.dump(train_set, open('pickle/train_set_index.pkl', 'wb'))
        pickle.dump(validation_set, open('pickle/validation_set_index.pkl', 'wb'))
        pickle.dump(test_set, open('pickle/test_set_index.pkl', 'wb'))
        print(len(train_set), len(validation_set), len(test_set))
        lines = ['\n** DATASETS **',
                 'Training set: {}'.format(len(train_set)),
                 'Validation set: {}'.format(len(validation_set)),
                 'Test set: {}'.format(len(test_set)),
                 '\n']
        create_report(self.filename_report, lines)
        
        del list_new_indexes
        gc.collect()
            
        return train_set, validation_set, test_set

    # --------------------------------------------------

    def initialize_model(self, size_input):
        np.random.seed(self.seed)
        if self.model_architecture == 'NNet':
            torch.manual_seed(self.seed)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.first_layer = int(size_input)
            model = NN_drug_sensitivity(input_size=self.first_layer,
                                        layers=self.layers_info,
                                        activation_function=self.activation_function,
                                        dropout_prob=self.dropout)
            model.to(self.device)
            # lines = ['\n*About the network',
            #      'Layers: {} {} {}'.format(self.first_layer, self.second_layer, self.third_layer),
            #     'Runs on: {} \n'.format(self.device)]
            lines = ['\n*About the network',
                 '{}'.format(str(model.modules())),
                'Runs on: {} \n'.format(self.device)]
        else:
            self.device = 'cpu'
            model = RandomForestRegressor(n_estimators = self.number_trees, criterion = 'mse', random_state = self.seed)
            lines = ['\n*About the network',
                 'Number of trees: {}'.format(self.number_trees),
                'Runs on: {} \n'.format(self.device)]
             
        #save parameters as a pkl file
        self.save_parameters()
       
        create_report(self.filename_report, lines)
        
        return model

    # --------------------------------------------------

    def __train_validation_nnet(self, model, train_set, validation_set):
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.size_batch, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.size_batch, shuffle=False)
        
        free_memory = [train_set, validation_set]
        for item in free_memory:
            del item
        gc.collect()

        epoch_stop = int(2.3 * self.epoch_reset)
        got_better = False
        n_epochs_not_getting_better = 0

        optimizer = copy.copy(self.optimiser)
        best_model = copy.deepcopy(model.state_dict())  # save the best model yet with the best accuracy and lower loss value
        decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        # Save the loss values (plot) + learning rates of each epoch (plot)
        loss_values_training = {}  # different loss values in different epochs (training)
        loss_values_validation = {}  # different loss values in different epoch (validation)
        learning_rates = {}  # learning rates values per epoch
        times_training = {}  # time spent per epoch (training)
        times_validation = {}  # time spent per epoch (validation)

        # Training and Validation
        for epoch in range(self.n_epochs):
            train_loss_epoch = 0.0
            validation_loss_epoch = 0.0

            if (epoch + 1) % self.epoch_reset == 0 and epoch != (self.n_epochs - 1):
                print('-' * 10)
                print('Epoch: {} of {}'.format(epoch + 1, self.n_epochs))
                if epoch != 0:
                    optimizer = copy.copy(self.optimiser)
                    decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

            # epoch learning rate value
            learning_rates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']

            # TRAINING
            start_train_time = time.time()
            train_predictions_complete = []
            model.train()  # set model for training
            for i, data in enumerate(train_loader):
                inputs, real_values = data
                inputs = inputs.to(self.device)
                real_values = real_values.to(self.device)
                optimizer.zero_grad()  # set the gradients of all parameters to zero
                train_predictions = model(inputs)  # output predicted by the model
                train_current_loss = self.__loss_function(real_values, train_predictions)
                train_current_loss.backward()  # backpropagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                train_predictions_complete.extend(x[0] for x in train_predictions.detach().cpu().numpy().tolist())
                train_loss_epoch += train_current_loss.item()
            
            train_loss_epoch = train_loss_epoch / len(train_loader)
            end_train_model = time.time()
            loss_values_training[epoch] = train_loss_epoch
            times_training[epoch] = end_train_model - start_train_time

            # VALIDATION
            start_validation_time = time.time()
            model.eval()
            validation_predictions_complete = []
            with torch.no_grad():
                for i, data in enumerate(validation_loader):
                    inputs, real_values = data
                    inputs = inputs.to(self.device)
                    real_values = real_values.to(self.device)
                    validation_predictions = model(inputs)  # output predicted by the model
                    validation_current_loss = self.__loss_function(real_values, validation_predictions)
                    validation_loss_epoch += validation_current_loss.item()
                    validation_predictions_complete.extend(x[0] for x in validation_predictions.cpu().numpy().tolist())
            
            validation_loss_epoch = validation_loss_epoch / len(validation_loader)
            end_validation_time = time.time()
            loss_values_validation[epoch] = validation_loss_epoch 
            times_validation[epoch] = end_validation_time - start_validation_time

            if epoch == 0 or validation_loss_epoch < best_loss[1]:  # means that this model is best one yet
                best_loss = (train_loss_epoch, validation_loss_epoch)
                best_model = copy.deepcopy(model.state_dict())
                print(validation_predictions_complete[:10])
                with open('pickle/validation_output.txt', 'w') as f:
                    f.write('\n'.join(['{:f}'.format(x) for x in validation_predictions_complete]))
                with open('pickle/train_output.txt', 'w') as f:
                    f.write('\n'.join(['{:f}'.format(x) for x in train_predictions_complete]))
                got_better = True
                n_epochs_not_getting_better = 0
                pickle.dump(best_model, open('pickle/best_model_parameters.pkl', 'wb'))
            else:
                got_better = False
                n_epochs_not_getting_better += 1

            if (epoch + 1) % 200 == 0:
                model_parameters = copy.deepcopy(model.state_dict())
                pickle.dump(model_parameters, open('pickle/model_parameters_{}.pkl'.format(epoch), 'wb'))

            free_memory = [train_predictions_complete, validation_predictions_complete]
            for item in free_memory:
                del item
            gc.collect()

            with open('model_values/loss_value_while_running.txt', 'a') as f:
                f.write('Epoch: {} \n'.format(epoch))
                f.write('Training loss : {:.2f} \n'.format(train_loss_epoch))
                f.write('Validation loss : {:.2f} \n'.format(validation_loss_epoch))
                f.write('Best loss: {:.2f} \n'.format(best_loss[1]))
                f.write('Time (training): {:.2f} \n'.format(end_train_model - start_train_time))
                f.write('Time (validation): {:.2f} \n'.format(end_validation_time - start_validation_time))
                f.write('Not getting better for {} epochs. \n'.format(n_epochs_not_getting_better))
                f.write('\n'.format(n_epochs_not_getting_better))

            # decay the learning rate
            decay_learning_rate.step()

            if n_epochs_not_getting_better == epoch_stop:
                break

            if (epoch + 1) % 10 == 0:
                results = pd.DataFrame.from_dict((loss_values_training, loss_values_validation,
                                                  learning_rates, times_training, times_validation)).T
                results.columns = ['Loss_Training', 'Loss_Validation', 'Learning_rates', 'Duration_Training', 'Duration_Validation']
                results.reset_index().to_csv('Training_Validation_results.txt', header=True, index=False)
                
                del results
                gc.collect()

        free_memory = [train_loader, validation_loader]
        for item in free_memory:
            del item
        gc.collect()

        # Saving the results
        results = pd.DataFrame.from_dict((loss_values_training, loss_values_validation, learning_rates, times_training, times_validation)).T
        results.columns = ['Loss_Training', 'Loss_Validation', 'Learning_rates', 'Duration_Training', 'Duration_Validation']
        results.reset_index().to_csv('Training_Validation_results.txt', header=True, index=False)

        del results
        gc.collect()

        model.load_state_dict(best_model)

        print('Training: Done!')
        lines = ['\nTraining loss: {:.2f}'.format(best_loss[0]),
                'Validation loss: {:.2f}'.format(best_loss[1]),
                'Number of epochs: {:.0f} of {:.0f} \n'.format(epoch + 1, self.n_epochs)]
        create_report(self.filename_report, lines)

        return model
    
    # --------------------------------------------------

    def __train_validation_rf(self, model, train_set):
        X_train = train_set[0].to_numpy()
        y_real = train_set[1]
        train_set_index = train_set[2]
        
        model.fit(X_train, y_real)
        
        y_pred = model.predict(X_train)
        
        with open('pickle/train_output.txt', 'w') as f:
            f.write('\n'.join(['{:f}'.format(x) for x in y_pred.tolist()]))

        return model

    # --------------------------------------------------

    def train_model(self, model, train_set, validation_set):
        start_training = time.time()
        if self.model_architecture == 'NNet':
            model = self.__train_validation_nnet(model, train_set, validation_set)
        else:
            model = self.__train_validation_rf(model, train_set)
        end_training = time.time()
        create_report(self.filename_report, ['Duration: {:.2f} \n'.format(end_training - start_training)])
        self.__save_model(model)

        return model

    # --------------------------------------------------

    def __loss_function(self, x_input, x_output):
        criterion = nn.MSELoss()
        reconstruction_loss = criterion(x_output, x_input)
        return reconstruction_loss

    # --------------------------------------------------

    def __save_model(self, model):
        if self.model_architecture == 'NNet':
            model_parameters = copy.deepcopy(model.state_dict())
            pickle.dump(model_parameters, open('pickle/drug_sensitivity_model.pkl', 'wb'))
        else:
            pickle.dump(model, open('pickle/drug_sensitivity_model.pkl', 'wb'))
    
    # --------------------------------------------------
 
    def load_model(self, model):
        if self.model_architecture == 'NNet':
            model_parameters = pickle.load(open('pickle/drug_sensitivity_model.pkl', 'rb'))
            model.load_state_dict(model_parameters)
        else:
            model = pickle.load(open('pickle/drug_sensitivity_model.pkl', 'rb'))
        
        return model
    
    # --------------------------------------------------

    def __run_test_set_nnet(self, model, test_set):
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.size_batch, shuffle=False)
        test_loss = 0.0
        test_predictions_complete = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, real_values = data
                inputs = inputs.to(self.device)
                real_values = real_values.to(self.device)
                test_predictions = model(inputs)  # output predicted by the model
                current_loss = self.__loss_function(real_values, test_predictions)
                test_loss += current_loss.item()
                test_predictions_complete.extend(x[0] for x in test_predictions.cpu().numpy().tolist())
        
        loss = test_loss / len(test_loader)
        
        print('Test loss: {:.2f} \n'.format(loss))
        create_report(self.filename_report, ['Testing loss: {:.2f}'.format(loss)])
        with open('pickle/test_output.txt', 'w') as f:
            f.write('\n'.join(['{:f}'.format(x) for x in test_predictions_complete]))
        return test_predictions_complete
    
    # --------------------------------------------------
    
    def __run_test_set_rf(self, model, test_set):
        X_test = test_set[0].to_numpy()
        y_real = test_set[1]
        test_set_index = test_set[2]
        
        y_pred = model.predict(X_test)
        
        del X_test
        gc.collect()
        
        with open('pickle/test_output.txt', 'w') as f:
            f.write('\n'.join(['{:f}'.format(x) for x in y_pred.tolist()]))
        
        mse = mean_squared_error(y_real, y_pred)
        print('Mean squared error: {:.2f}'.format(mse))
        
        corr_value, _ = pearsonr(y_real, y_pred)
        make_correlation_plot(y_real, y_pred, test_set_index, 'Test_set')
        
        
        lines = ['\n \nTesting loss: {:.2f}'.format(mse),
                 'Testing correlation: {:.2f}'.format(corr_value),
                 '\n']
        create_report(self.filename_report, lines)
        
        return y_pred.tolist()
    
    # --------------------------------------------------

    def run_test_set(self, model, test_set):
        if self.model_architecture == 'NNet':
            output = self.__run_test_set_nnet(model, test_set)
        else:
            output = self.__run_test_set_rf(model, test_set)

        return output

    # --------------------------------------------------

    def plot_loss_lr(self, x, loss_training, loss_validation, learning_rates):

        minimum_loss = min(min(loss_training), min(loss_validation))
        maximum_loss = max(max(loss_training), max(loss_validation))

        fig = plt.figure(figsize=(12, 16))
        (ax1, ax3) = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(x, loss_training, color='r', label='Loss (training)')
        ax1.set_ylabel('Loss')
        ax1.plot(x, loss_validation, color='g', label='Loss (validation)')
        ax1.set_ylim(minimum_loss, maximum_loss)
        # 
        ax3.set_xlabel('Number of epochs')
        ax3.set_ylabel('Learning rate')
        ax3.plot(x, learning_rates, color='b', label='Learning rates')
        fig.legend(loc=1)
        fig.tight_layout()
        plt.savefig('plots/Values_per_epoch.png')
   
   # --------------------------------------------------

    def create_filename(self, list_parameters):
        filename_output = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(list_parameters[15], list_parameters[1], list_parameters[2], list_parameters[3],
                                                                          list_parameters[4], list_parameters[5], list_parameters[6],
                                                                          list_parameters[7], list_parameters[8], list_parameters[9],
                                                                          list_parameters[10], list_parameters[11], list_parameters[12],
                                                                          list_parameters[13])
        self.filename_report = '{}/{}/output_{}.txt'.format(list_parameters[0], list_parameters[14], filename_output)
        return self.filename_report
        
    # --------------------------------------------------

    def get_model_architecture(self):
        return self.model_architecture
    
    # --------------------------------------------------
    
    def save_parameters(self):
        if self.model_architecture == 'NNet':
            for k,v in dict_optimisers.items():
                if k == self.optimiser:
                    optmiser = v
                    break
            network_info = '{}_{}_{}'.format('_'.join(list(self.layers_info.values()), self.activation_function, optmiser))
            pickle.dump([network_info, self.learning_rate, self.size_batch, self.n_epochs, self.perc_train, self.perc_val,
             self.dropout, self.gamma, self.step_size, self.seed, self.epoch_reset, self.type_of_split,
             self.to_test, self.data_from, self.model_architecture, self.device], open('pickle/list_initial_parameters_single_cell.pkl', 'wb'))
        
        else:
            pickle.dump([self.number_trees, self.learning_rate, self.size_batch, self.n_epochs, self.perc_train, self.perc_val,
             self.dropout, self.gamma, self.step_size, self.seed, self.epoch_reset, self.type_of_split,
             self.to_test, self.data_from, self.model_architecture, self.device], open('pickle/list_initial_parameters_single_cell.pkl', 'wb'))   
    
    # --------------------------------------------------

# -------------------------------------------------- RUN --------------------------------------------------

def run_drug_prediction(list_parameters):
    drug_sens = Drug_sensitivity_single_cell()
    pancancer_bottlenecks = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/single_cell/pancancer_with_alpha_bottlenecks.pkl', 'rb'))
    pancancer_metadata = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/pancancer_metadata.pkl', 'rb'))
    prism_bottlenecks = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_once/pkl_files/prism_bottlenecks.pkl', 'rb'))
    prism_dataset = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/prism_dataset.pkl', 'rb'))
    ccle2depmap_dict = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/ccle2depmap_dict.pkl', 'rb'))
    
    #filename for the reports
    filename = drug_sens.create_filename(list_parameters)
    drug_sens.set_parameters(list_parameters)
    model_architecture = drug_sens.get_model_architecture()
    
    #load and process the datasets
    train_set_index, validation_set_index, test_set_index = drug_sens.load_datasets()
    
    #Training set
    train_set = {}
    train_set_sensitivity = []
    train_set_total = {}
    train_set_gene = {}
    for i in train_set_index:
        new_i = i.strip('\n').split('::')
        index = [new_i[0], '::'.join(new_i[1:])]
        if ':::' in index[1]:
            screen = index[1].split(':::')[0]
        else:
            screen = index[1]
        new_index = i.strip('\n')
        data = list(pancancer_bottlenecks.loc[index[0]].iloc[:-1])
        cell_line_dep_map = ccle2depmap_dict[pancancer_bottlenecks.loc[index[0]].iloc[-1]]
        data.extend(list(prism_bottlenecks.loc[index[1]]))
        train_set_sensitivity.append(prism_dataset.loc[cell_line_dep_map, screen])
        train_set[new_index] = np.array(data)
        train_set_total[new_index] = [index[0], pancancer_metadata.loc[index[0], 'Cell_line'],
                            screen, prism_dataset.loc[cell_line_dep_map, screen]]
        train_set_gene[new_index] = np.array(list(pancancer_bottlenecks.loc[index[0]].iloc[:-1]))
    
    train_set_index = list(train_set.keys())
    with open('pickle/train_set_real_values.txt', 'w') as f:
        f.write('\n'.join(['{:f}'.format(x) for x in train_set_sensitivity]))
    if model_architecture == 'NNet':
        train_set = torch.tensor(list(train_set.values())).type('torch.FloatTensor')
        train_set_sensitivity = torch.tensor(train_set_sensitivity).type('torch.FloatTensor')
        train_data = TensorDataset(train_set, train_set_sensitivity)
    else:
        train_set_sensitivity = np.array(train_set_sensitivity)
        train_set = pd.DataFrame.from_dict(train_set, orient='index')
        train_data = (train_set, train_set_sensitivity, train_set_index)    
    
    #start the Drug Sensitivity model
    if model_architecture == 'NNet':
        model = drug_sens.initialize_model(size_input=train_set.shape[1])
    else:
        model = drug_sens.initialize_model(size_input=[])
    
    free_memory = [train_set, train_set_sensitivity]
    for item in free_memory:
        del item
    gc.collect()
    
    #Validation set
    if model_architecture == 'NNet':
        validation_set = {}
        validation_set_sensitivity = []
        validation_set_total = {}
        validation_set_gene = {}
        for i in validation_set_index:
            new_i = i.strip('\n').split('::')
            index = [new_i[0], '::'.join(new_i[1:])]
            if ':::' in index[1]:
                screen = index[1].split(':::')[0]
            else:
                screen = index[1]
            new_index = i.strip('\n')
            data = list(pancancer_bottlenecks.loc[index[0]].iloc[:-1])
            cell_line_dep_map = ccle2depmap_dict[pancancer_bottlenecks.loc[index[0]].iloc[-1]]
            data.extend(list(prism_bottlenecks.loc[index[1]]))
            validation_set_sensitivity.append(prism_dataset.loc[cell_line_dep_map, screen])
            validation_set[new_index] = np.array(data)
            validation_set_total[new_index] = [index[0], pancancer_metadata.loc[index[0], 'Cell_line'],
                            screen, prism_dataset.loc[cell_line_dep_map, screen]]
            validation_set_gene[new_index] = np.array(list(pancancer_bottlenecks.loc[index[0]].iloc[:-1]))
        
        validation_set_index = list(validation_set.keys())
        with open('pickle/validation_set_real_values.txt', 'w') as f:
            f.write('\n'.join(['{:f}'.format(x) for x in validation_set_sensitivity]))
        validation_set = torch.tensor(list(validation_set.values())).type('torch.FloatTensor')
        validation_set_sensitivity =  torch.tensor(validation_set_sensitivity).type('torch.FloatTensor')
        validation_data = TensorDataset(validation_set, validation_set_sensitivity)
        
        free_memory = [validation_set, validation_set_sensitivity]
        for item in free_memory:
            del item
        gc.collect()
    else:
        validation_data = []
    
    #train the model
    model_trained = drug_sens.train_model(model, train_data, validation_data)
    
    if model_architecture == 'NNEt':
        with open('pickle/validation_output.txt', 'r') as f:
            validation_output = f.readlines()
        for i in range(len(validation_output)):
            index = validation_set_index[i]
            value = validation_output[i]
            validation_set_total[index].append(value)
        validation_set_total = pd.DataFrame.from_dict(validation_set_total, orient='index', columns = ['barcode', 'Cell_line', 'Screen_id', 'Real_sensitivity', 'Predicted_sensitivity'])
        validation_set_gene = pd.DataFrame.from_dict(validation_set_gene, orient='index')
        validation_set_total = pd.concat([validation_set_total, validation_set_gene], axis=1)
        validation_set_total.reset_index().to_csv('validation_set_total.csv', header=True, index=False)
        free_memory = [validation_output, validation_set_gene, validation_set_total]
        for item in free_memory:
            del item
        gc.collect()

    with open('pickle/train_output.txt', 'r') as f:
        train_output = f.readlines()
    for i in range(len(train_output)):
        index = train_set_index[i]
        value = train_output[i]
        train_set_total[index].append(value)
    train_set_total = pd.DataFrame.from_dict(train_set_total, orient='index', columns = ['barcode', 'Cell_line', 'Screen_id', 'Real_sensitivity', 'Predicted_sensitivity'])
    train_set_gene = pd.DataFrame.from_dict(train_set_gene, orient='index')
    train_set_total = pd.concat([train_set_total, train_set_gene], axis=1)
    train_set_total.reset_index().to_csv('train_set_total.csv', header=True, index=False)
    free_memory = [train_output, train_set_gene, train_set_total]
    for item in free_memory:
        del item
    gc.collect()
    
    
    #Test set
    test_set = {}
    test_set_sensitivity = []
    test_set_total = {}
    test_set_gene = {}
    for i in test_set_index:
        new_i = i.strip('\n').split('::')
        index = [new_i[0], '::'.join(new_i[1:])]
        if ':::' in index[1]:
            screen = index[1].split(':::')[0]
        else:
            screen = index[1]
        new_index = i.strip('\n')
        data = list(pancancer_bottlenecks.loc[index[0]].iloc[:-1])
        cell_line_dep_map = ccle2depmap_dict[pancancer_bottlenecks.loc[index[0]].iloc[-1]]
        data.extend(list(prism_bottlenecks.loc[index[1]]))
        test_set_sensitivity.append(prism_dataset.loc[cell_line_dep_map, screen])
        test_set[new_index] = np.array(data)
        test_set_total[new_index] = [index[0], pancancer_metadata.loc[index[0], 'Cell_line'],
                            screen, prism_dataset.loc[cell_line_dep_map, screen]]
        test_set_gene[new_index] = np.array(list(pancancer_bottlenecks.loc[index[0]].iloc[:-1]))
    
    
    free_memory = [prism_dataset, pancancer_bottlenecks, prism_bottlenecks, ccle2depmap_dict]
    for item in free_memory:
        del item
    gc.collect()

    test_set_index = list(test_set.keys())
    with open('pickle/test_set_real_values.txt', 'w') as f:
        f.write('\n'.join(['{:f}'.format(x) for x in test_set_sensitivity]))
    if model_architecture == 'NNet':
        test_set = torch.tensor(list(test_set.values())).type('torch.FloatTensor')
        test_set_sensitivity = torch.tensor(test_set_sensitivity).type('torch.FloatTensor')
        test_data = TensorDataset(test_set, test_set_sensitivity)
    else:
        test_set = pd.DataFrame.from_dict(test_set, orient='index')
        test_set_sensitivity = np.array(test_set_sensitivity)
        test_data = (test_set, test_set_sensitivity, test_set_index)
    
    del test_set
    gc.collect()
    
    test_output = drug_sens.run_test_set(model_trained, test_data)
        
    free_memory = [model_trained, test_data]
    for item in free_memory:
        del item
    gc.collect()
    
    for i in range(len(test_output)):
        index = test_set_index[i]
        value = test_output[i]
        test_set_total[index].append(value)
    
    test_set_total = pd.DataFrame.from_dict(test_set_total, orient='index', columns = ['barcode', 'Cell_line', 'Screen_id', 'Real_sensitivity', 'Predicted_sensitivity'])
    test_set_gene = pd.DataFrame.from_dict(test_set_gene, orient='index')
    test_set_total = pd.concat([test_set_total, test_set_gene], axis=1)
    test_set_total.reset_index().to_csv('test_set_total.csv', header=True, index=False)
    
    free_memory = [test_set_gene, test_set_total]
    for item in free_memory:
        del item
    gc.collect()
    
    #plots
    if model_architecture == 'NNet':
        results = pd.read_csv('Training_Validation_results.txt', header = 0, index_col = 0)
        drug_sens.plot_loss_lr(list(list(results.index)), list(results['Loss_Training']), list(results['Loss_Validation']), list(results['Learning_rates']))
        
        results_barplots = results.loc[results.index % 10 == 0]
        results_barplots.loc[:, ['Duration_Training', 'Duration_Validation']].plot(kind='bar', rot=0, subplots=True, figsize=(16, 8))
        plt.savefig('plots/Duration_per_epoch.png', bbox_inches='tight')
        
        free_memory = [results_barplots, results]
        for item in free_memory:
            del item
        gc.collect()
    
    #correlation matrices
    with open('pickle/train_output.txt', 'r') as f:
        train_predictions = f.readlines()
    with open('pickle/train_set_real_values.txt', 'r') as f:
        train_set_sensitivity = f.readlines()
    train_predictions = [x.strip('\n') for x in train_predictions]
    train_set_sensitivity = [x.strip('\n') for x in train_set_sensitivity]
    corr_value, _ = pearsonr(np.array(train_set_sensitivity), np.array(train_predictions))
    make_correlation_plot(train_set_sensitivity, train_predictions, train_set_index, 'Training_set')
    lines = ['\n \n** CORRELATION VALUES **',
             'Training set: {}'.format(corr_value),
             '\n']
    create_report(filename, lines)
    free_memory = [train_predictions, train_set_sensitivity, train_set_index, corr_value, lines]
    for item in free_memory:
        del item
    gc.collect()
    
    if model_architecture == 'NNet':
        with open('pickle/validation_output.txt', 'r') as f:
            validation_predictions = f.readlines()
        with open('pickle/validation_set_real_values.txt', 'r') as f:
            validation_set_sensitivity = f.readlines()
        validation_predictions = [x.strip('\n') for x in validation_predictions]
        validation_set_sensitivity = [x.strip('\n') for x in validation_set_sensitivity]
        corr_value, _ = pearsonr(np.array(validation_set_sensitivity), np.array(validation_predictions))
        make_correlation_plot(validation_set_sensitivity, validation_predictions, validation_set_index, 'Validation_set')
        lines = ['Validation set: {}'.format(corr_value),
                 '\n']
        create_report(filename, lines)
        
        free_memory = [validation_predictions, validation_set_sensitivity, validation_set_index, corr_value, lines]
        for item in free_memory:
            del item
        gc.collect()
    
    test_output = [x.strip('\n') for x in test_output]
    test_set_sensitivity = [x.strip('\n') for x in test_set_sensitivity]
    corr_value, _ = pearsonr(np.array(test_set_sensitivity), np.array(test_output))
    make_correlation_plot(test_set_sensitivity, test_output, test_set_index, 'Test_set')
    lines = ['Test set: {}'.format(corr_value),
             '\n']
    create_report(filename, lines)
    
    free_memory = [test_output, test_set_sensitivity, test_set_index]
    for item in free_memory:
        del item
    gc.collect()
    
    
    
        
    print('Done!')


# -------------------------------------------------- INPUT --------------------------------------------------

try:
    input_values = sys.argv[1:]
    print(input_values)
    run_drug_prediction(input_values)

except EOFError:
    print('ERROR!')