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
from torchvision import models
from torchsummary import summary
import dask.dataframe as dd

from full_network import NN_drug_sensitivity, VAE_molecular, VAE_gene_expression_single_cell, AE_gene_expression_bulk
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- ANOTHER FUNCTIONS --------------------------------------------------

def create_report(filename, list_comments):
    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/{}'.format(filename), 'a') as f:
        f.write('\n'.join(list_comments))

# --------------------------------------------------

def make_correlation_plot(real_data, predicted_data, type_data):
    d = {'Real_data' : real_data, 'Predicted_data' : predicted_data}
    data = pd.DataFrame.from_dict(d)
    
    plt.figure(figsize=(15, 15))
    data.plot(x='Real_data', y='Predicted_data', style=['o'])
    plt.title('Real values vs Predicted values - {}'.format(type_data), fontsize=14, fontweight='bold')
    plt.xlabel('Real_data')
    plt.ylabel('Predicted_data')
    plt.savefig('plots/Scatter_real_vs_predicted_values_{}.png'.format(type_data))

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

        self.ccle2depmap_dict = None
        
        self.dataset_index_dict = {}

    # --------------------------------------------------

    def set_parameters(self, list_parameters):
        self.type_data = list_parameters[0]
        self.model_architecture = list_parameters[15]

        if self.model_architecture == 'NNet':
            network_info = list_parameters[1].split('_')
            self.optimiser = network_info[-1]
            self.activation_function = network_info[-2]
            self.layers_info = [x for x in network_info[:-2]]
            self.layers_info.append('1')

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
        self.ccle2depmap_dict = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/ccle2depmap_dict.pkl','rb'))

        if seed != self.seed:
            seed = self.seed
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
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
            barcode2indexes = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_new_indexes_5percell_line_once_dict.pkl', 'rb'))

            with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_tumours.txt', 'r') as f:
                list_tumours = f.readlines()

            barcodes_per_tumour = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/barcodes_per_tumour_dict.pkl', 'rb'))

            if self.type_of_split == 'random':
                list_tumours = [x.strip('\n') for x in list_tumours]
                list_tumours = shuffle(list_tumours)
                list_barcodes_pancancer = []
                for i in range(4):
                    tumour = list_tumours[i]
                    list_barcodes_pancancer.extend(barcodes_per_tumour[tumour])
                
                list_final_indexes = []
                for bar in list_barcodes_pancancer:
                    try:
                        list_final_indexes.extend(barcode2indexes[bar])
                    except:
                        pass
                # list_final_indexes = shuffle(list_final_indexes[:50000])
                
                train_number = int(self.perc_train * len(list_final_indexes))
                validation_number = int(self.perc_val * len(list_final_indexes))
                
                train_set = list_final_indexes[:train_number]
                validation_set = list_final_indexes[train_number:train_number+validation_number]
                test_set = list_final_indexes[train_number+validation_number:]
        
            elif self.type_of_split == 'leave-one-cell-line-out':
                with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/prism_pancancer_cell_lines_pancancer.txt', 'r') as f:
                    list_cell_lines = f.readlines()
                list_cell_lines.remove(self.to_test)

                barcodes_per_cell_line = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/barcodes_per_cell_line_dict.pkl', 'rb'))
                
                train_cells = list_cell_lines[:-1]
                validation_cell = list_cell_lines[-1]

                train_set = []
                validation_set = []
                test_set = []
                for i in range(len(train_cells)):
                    barcodes = barcodes_per_cell_line[train_cells[i]]
                    for bar in barcodes:
                        train_set.extend(barcode2indexes[bar])
                barcodes = barcodes_per_cell_line[validation_cell[0]]
                for bar in barcodes:
                    validation_set.extend(barcode2indexes[bar])
                barcodes = barcodes_per_cell_line[self.to_test]
                for bar in barcodes:
                    test_set.extend(barcode2indexes[bar])

            elif self.type_of_split == 'leave-one-tumour-out':
                list_tumours.remove(self.to_test)

                train_tumours = list_tumours[:-1]
                validation_tumour = list_tumours[-1]
                
                train_set = []
                validation_set = []
                test_set = []
                for i in range(len(train_tumours)):
                    barcodes = barcodes_per_tumour[train_tumours[i]]
                    for bar in barcodes:
                        train_set.extend(barcode2indexes[bar])
                barcodes = barcodes_per_tumour[validation_tumour[0]]
                for bar in barcodes:
                    validation_set.extend(barcode2indexes[bar])
                barcodes = barcodes_per_tumour[self.to_test]
                for bar in barcodes:
                    test_set.extend(barcode2indexes[bar])
        
        
        train_set_dict = {x : [] for x in train_set}
        validation_set_dict = {x : [] for x in validation_set}
        test_set_dict = {x : [] for x in test_set}
        
        datasets = [train_set_dict, validation_set_dict, test_set_dict]
        for data in datasets:
            for k in data.keys():
                temp = k.strip('\n').split('::')
                temp[1] = '::'.join(temp[1:])
                data[k] = (temp[0], temp[1])
        
        self.dataset_index_dict['Train'] = train_set_dict
        self.dataset_index_dict['Validation'] = validation_set_dict
        self.dataset_index_dict['Test'] = test_set_dict
        
        pickle.dump(train_set, open('pickle/train_set_index_dict.pkl', 'wb'))
        pickle.dump(validation_set, open('pickle/validation_set_index_dict.pkl', 'wb'))
        pickle.dump(test_set, open('pickle/test_set_index_dict.pkl', 'wb'))
        print(len(train_set), len(validation_set), len(test_set))
        lines = ['\n** DATASETS **',
                 'Training set: {}'.format(len(train_set)),
                 'Validation set: {}'.format(len(validation_set)),
                 'Test set: {}'.format(len(test_set)),
                 '\n']
        create_report(self.filename_report, lines)
        return train_set, validation_set, test_set

    # --------------------------------------------------

    def __save_batch(self, data, type_data):
        sc_metadata = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/pkl_files/pancancer_metadata.pkl', 'rb'))
        list_indexes = list(data.index)
        list_sensitivity = list(data.iloc[-1])

        dataset_total = []
        for i in range(len(list_indexes)):
            ids = list_indexes[i].split('::')
            barcode, prism_bot_index = ids[0], '::'.join(ids[1:])
            screen = prism_bot_index.split(':::')[0]
            dataset_total.append('{},{},{},{},{}'.format(list_indexes[i], barcode, sc_metadata.loc[barcode, 'Cell_line'],
                                                             screen, list_sensitivity[i]))

        with open('pickle/{}_set_real_values.txt'.format(type_data), 'a') as f:
            f.write('\n'.join(list_sensitivity))
            f.write('\n')
        with open('pickle/{}_set_total.txt'.format(type_data), 'a') as f:
            f.write('\n'.join(dataset_total))
            f.write('\n')

    # --------------------------------------------------

    def save_dataset(self, type_data):
        with open('pickle/{}_set_total.txt'.format(type_data), 'r') as f:
            data = f.readlines()
        with open('pickle/{}_output.txt'.format(type_data), 'r') as f:
            output = f.readlines()

        if len(data) != len(output):
            print('Error!! \n{}_set: \nData - {} ; Output - {}'.format(type_data, len(data), len(output)))
            exit()

        for i in range(len(data)):
            if i == 0:
                data[i] = '{},{}'.format(data[i], 'predicted_sensitivity')
            else:
                data[i] = '{},{}'.format(data[i], output[i])

        with open('pickle/{}_set_total.txt'.format(type_data), 'w') as f:
            f.write('\n'.join(data))

    # --------------------------------------------------

    def initialize_model(self, size_input):
        if self.model_architecture == 'NNet':
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

    def __train_validation_nnet(self, model, train_indexes, validation_indexes, train_set, validation_set):
        
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

        epoch_stop = int(1.5 * self.epoch_reset)
        got_better = False
        n_epochs_not_getting_better = 0

        optimizer = dict_optimisers[self.optimiser]
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
                    optimizer = dict_optimisers[self.optimiser]
                    decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

            # epoch learning rate value
            learning_rates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']

            # TRAINING
            start_train_time = time.time()
            train_predictions_complete = []
            model.train()  # set model for training
            batches = 0
            for i in range(0, len(train_indexes), self.size_batch):
                list_indexes = train_indexes[i:i+self.size_batch]
                batch = train_set.loc[list_indexes].compute()
                if epoch == 0:
                    self.__save_batch(batch, 'Train')
                inputs = torch.tensor(batch.iloc[:-1].to_numpy()).type('torch.FloatTensor')
                real_values = torch.tensor(batch.iloc[-1].to_numpy()).type('torch.FloatTensor')
                inputs = inputs.to(self.device)
                real_values = real_values.to(self.device)
                optimizer.zero_grad()  # set the gradients of all parameters to zero
                train_predictions = model(inputs)  # output predicted by the model
                train_current_loss = self.__loss_function(real_values, train_predictions)
                train_current_loss.backward()  # backpropagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                train_predictions_complete.extend(train_predictions.detach().cpu().numpy().tolist())
                train_loss_epoch += train_current_loss.item()
                batches += 1
            
            train_loss_epoch = train_loss_epoch / batches
            end_train_model = time.time()
            loss_values_training[epoch] = train_loss_epoch
            times_training[epoch] = end_train_model - start_train_time

            # VALIDATION
            start_validation_time = time.time()
            model.eval()
            validation_predictions_complete = []
            with torch.no_grad():
                batches = 0
                for i in range(0, len(validation_indexes), self.size_batch):
                    list_indexes = validation_indexes[i:i + self.size_batch]
                    batch = validation_set.loc[list_indexes].compute()
                    if epoch == 0:
                        self.__save_batch(batch, 'Validation')
                    inputs = torch.tensor(batch.iloc[:-1].to_numpy()).type('torch.FloatTensor')
                    real_values = torch.tensor(batch.iloc[-1].to_numpy()).type('torch.FloatTensor')
                    inputs = inputs.to(self.device)
                    real_values = real_values.to(self.device)
                    validation_predictions = model(inputs)  # output predicted by the model
                    validation_current_loss = self.__loss_function(real_values, validation_predictions)
                    validation_loss_epoch += validation_current_loss.item()
                    validation_predictions_complete.extend(validation_predictions.cpu().numpy().tolist())
                    batches += 1
            
            validation_loss_epoch = validation_loss_epoch / batches
            end_validation_time = time.time()
            loss_values_validation[epoch] = validation_loss_epoch 
            times_validation[epoch] = end_validation_time - start_validation_time

            if epoch == 0 or validation_loss_epoch < best_loss[1]:  # means that this model is best one yet
                best_loss = (train_loss_epoch, validation_loss_epoch)
                best_model = copy.deepcopy(model.state_dict())
                with open('pickle/Validation_output.txt', 'w') as f:
                    f.write('\n'.join(['{:f}'.format(x[0]) for x in validation_predictions_complete]))
                with open('pickle/Train_output.txt', 'w') as f:
                    f.write('\n'.join(['{:f}'.format(x[0]) for x in train_predictions_complete]))
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

    def train_model(self, model, train_indexes, validation_indexes, sc_bottlenecks, sc_metadata, prism_bottlenecks, prism_values):
        start_training = time.time()
        if self.model_architecture == 'NNet':
            model = self.__train_validation_nnet(model, train_indexes, validation_indexes, sc_bottlenecks, sc_metadata, prism_bottlenecks, prism_values)
        # else:
        #     model = self.__train_validation_rf(model, train_set)
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

    def __run_test_set_nnet(self, model, test_indexes, test_set):
        test_loss = 0.0
        test_predictions_complete = []
        model.eval()
        with torch.no_grad():
            batches = 0
            for i in range(0, len(test_indexes), self.size_batch):
                list_indexes = test_indexes[i:i + self.size_batch]
                batch = test_set.loc[list_indexes].compute()
                if epoch == 0:
                    self.__save_batch(batch, 'Test')
                inputs = torch.tensor(batch.iloc[:-1].to_numpy()).type('torch.FloatTensor')
                real_values = torch.tensor(batch.iloc[-1].to_numpy()).type('torch.FloatTensor')
                inputs = inputs.to(self.device)
                real_values = real_values.to(self.device)
                test_predictions = model(inputs)  # output predicted by the model
                current_loss = self.__loss_function(real_values, test_predictions)
                test_loss += current_loss.item()
                test_predictions_complete.extend(test_predictions.cpu().numpy().tolist())
                batches += 1
        
        loss = test_loss / batches
        
        print('Test loss: {:.2f} \n'.format(loss))
        create_report(self.filename_report, ['Testing loss: {:.2f}'.format(loss)])
        with open('pickle/Test_output.txt', 'w') as f:
            f.write('\n'.join(['{:f}'.format(x[0]) for x in test_predictions_complete]))
        # return test_predictions_complete
    
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
        
        # return y_pred.tolist()
    
    # --------------------------------------------------

    def run_test_set(self, model, test_indexes, sc_bottlenecks, sc_metadata, prism_bottlenecks, prism_values):
        if self.model_architecture == 'NNet':
            self.__run_test_set_nnet(model, test_indexes, sc_bottlenecks, sc_metadata, prism_bottlenecks, prism_values)
        # else:
        #     self.__run_test_set_rf(model, test_set)

    # --------------------------------------------------

    def plot_loss_lr(self, x, loss_training, loss_validation, learning_rates):

        minimum_loss = min(min(loss_training), min(loss_validation))
        maximum_loss = max(max(loss_training), max(loss_validation))
        filename = self.filename_report.split('output_')[-1]

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
        plt.savefig('plots/Values_per_epoch_{}.png'.format(filename))
   
   # --------------------------------------------------

    def create_filename(self, list_parameters):

        filename_output = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(list_parameters[15], list_parameters[1], list_parameters[2],
                                                                          list_parameters[3], list_parameters[4], list_parameters[5],
                                                                          list_parameters[6], list_parameters[7], list_parameters[8],
                                                                          list_parameters[9], list_parameters[10], list_parameters[11],
                                                                          list_parameters[12], list_parameters[13])
        self.filename_report = '{}/{}/output_{}.txt'.format(list_parameters[0], list_parameters[14], filename_output)
        return self.filename_report
        
    # --------------------------------------------------

    def get_model_architecture(self):
        return self.model_architecture
    
    # --------------------------------------------------
    
    def save_parameters(self):
        if self.model_architecture == 'NNet':
            network_info = '{}_{}_{}'.format(self.layers_info[-1], self.activation_function, self.optimiser)
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
    start_run = time.time()
    drug_sens = Drug_sensitivity_single_cell()

    combined_dataset = dd.read_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/prism_pancancer/csv_files/once/prism_pancancer_5percell_line_*.csv').set_index('combined_index')

    #filename for the reports
    filename = drug_sens.create_filename(list_parameters)
    drug_sens.set_parameters(list_parameters)

    model_architecture = drug_sens.get_model_architecture()
    
    #load and process the datasets
    train_set_index, validation_set_index, test_set_index = drug_sens.load_datasets()

    #initialise the dataset files
    if model_architecture == 'NNet':
        type_data = ['Train', 'Validation', 'Test']
    else:
        type_data = ['Train', 'Test']

    columns = ['index', 'sc_barcode', 'cell_line', 'screen_id', 'real_sensitivity']
    for i in range(len(type_data)):
        with open('pickle/{}_set_total.txt'.format(type_data[i]), 'w') as f:
            f.write(','.join(columns))
    
    #start the Drug Sensitivity model
    if model_architecture == 'NNet':
        model = drug_sens.initialize_model(size_input=combined_dataset.compute().iloc[:-1].shape[1])
    else:
        model = drug_sens.initialize_model(size_input=[])

    train_set = combined_dataset.loc[train_set_index]
    validation_set = combined_dataset.loc[validation_set_index]
    test_set = combined_dataset.loc[test_set_index]

    del combined_dataset
    gc.collect()

    #train the model
    model_trained = drug_sens.train_model(model, train_set_index, validation_set_index, train_set, validation_set)

    free_memory = [train_set, validation_set]
    for item in free_memory:
        del item
    gc.collect()
    
    drug_sens.run_test_set(model_trained, test_set_index, test_set)

    free_memory = [test_set, model_trained]
    for item in free_memory:
        del item
    gc.collect()

    #add the predicted values to the final dataset
    for i in range(len(type_data)):
        drug_sens.save_dataset(type_data[i])

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
    for i in range(len(type_data)):
        with open('pickle/{}_output.txt'.format(type_data[i]), 'r') as f:
            predictions = f.readlines()
        with open('pickle/{}_set_real_values.txt'.format(type_data[i]), 'r') as f:
            real_values = f.readlines()
        predictions = np.array([float(x.strip('\n')) for x in predictions])
        real_values = np.array([float(x.strip('\n')) for x in real_values])
        if type_data[i] == 'Validation':
            plot_name = '{}_set'.format(type_data[i])
            lines = ['\n{} real values max: {:.2f}'.format(type_data[i], predictions.max()),
                     '{} real values min: {:.2f}'.format(type_data[i], predictions.min()),
                     '{} predicted values max: {:.2f}'.format(type_data[i], predictions.max()),
                     '{} predicted values min: {:.2f}'.format(type_data[i], predictions.min())]
        else:
            plot_name = '{}ing_set'.format(type_data[i])
            lines = ['\n{}ing real values max: {:.2f}'.format(type_data[i], real_values.max()),
                     '{}ing real values min: {:.2f}'.format(type_data[i], real_values.min()),
                     '{}ing predicted values max: {:.2f}'.format(type_data[i], predictions.max()),
                     '{}ing predicted values min: {:.2f}'.format(type_data[i], predictions.min())]
        corr_value, _ = pearsonr(real_values, predictions)
        make_correlation_plot(real_values, predictions, plot_name)
        if type_data[i] == 'Validation':
            lines.extend(['\n{} correlation: {}'.format(type_data[i], corr_value), '\n'])
        else:
            lines.extend(['\n{}ing correlation: {}'.format(type_data[i], corr_value), '\n'])
        create_report(filename, lines)
    
    end_run = time.time()
    create_report(filename, ['\nTime for run: {:.2f}'.format(end_run - start_run)])
    print('Done!')


# -------------------------------------------------- INPUT --------------------------------------------------

try:
    input_values = sys.argv[1:]
    print(input_values)
    run_drug_prediction(input_values)

except EOFError:
    print('ERROR!')