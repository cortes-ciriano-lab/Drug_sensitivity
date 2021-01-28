# -------------------------------------------------- IMPORTS --------------------------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import pickle
import time
from sklearn.utils import shuffle
import gc
import sys
import torch.utils.data
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import os
import datetime
import random
import seaborn as sns

from lightgbm import LGBMRegressor
from full_network import NN_drug_sensitivity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# -------------------------------------------------- ANOTHER FUNCTIONS --------------------------------------------------

def create_report(filename, list_comments):
    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc_ctrp/{}'.format(filename), 'a') as f:
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
        self.sc_from = None

        #drug sensitivity
        self.path_data = None
        self.model_architecture = None
        self.run_type = None

        # if NNet
        self.layers_info = None

        # if RF or lGBM
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
        self.early_stop = None
        self.combination = None
        self.type_smiles_VAE = None
        
        self.type_lr = None

        self.filename_report = None

        self.ccle2depmap_dict = None

        self.ccle_per_barcode_dict = {}
        self.new_indexes2barcode_screen = None
        self.barcodes_per_cell_line = None
        
        self.train_barcodes = []
        self.validation_barcodes = []
        self.test_barcodes = []

    # --------------------------------------------------

    def set_parameters(self, list_parameters):
        print(list_parameters)
        self.learning_rate = float(list_parameters[1])
        self.size_batch = int(list_parameters[2])
        self.n_epochs = int(list_parameters[3])
        self.perc_train = float(list_parameters[4])
        self.dropout = float(list_parameters[6])
        self.gamma = float(list_parameters[7])
        self.step_size = int(list_parameters[8])
        self.seed = int(list_parameters[9])
        self.epoch_reset = int(list_parameters[10])
        self.type_of_split = list_parameters[11]
        self.to_test = list_parameters[12]
        self.type_lr = list_parameters[13]
        self.data_from = list_parameters[14]
        self.model_architecture = list_parameters[15]
        if 'yes' in list_parameters[16]: #tem de ser antes do runtype
            self.early_stop = int(list_parameters[16].split('-')[1])
        self.run_type = list_parameters[17]
        if list_parameters[18] == 'new':
            self.type_smiles_VAE = ''
        else:
            self.type_smiles_VAE = '_{}'.format(list_parameters[18])

        if self.model_architecture == 'NNet':
            self.layers_info = list_parameters[0].split('_')
            self.layers_info = [int(x) for x in self.layers_info]
            self.layers_info.append('1')
            self.perc_val = float(list_parameters[5])

        elif self.model_architecture == 'lGBM' or self.model_architecture == 'yrandom' or self.model_architecture == 'linear':
            self.number_trees = int(list_parameters[0])
            self.perc_val = 0.0

        if self.run_type == 'resume':
            self.n_epochs += int(list_parameters[-1])
        
        # if self.type_lr == 'non_cyclical':
        #     self.epoch_reset = self.n_epochs
        #     self.step_size = self.n_epochs

        # self.drug_from = self.data_from.split('_')[0]
        self.sc_from = self.data_from.split('_')[-1]
        self.path_data = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/{}'.format(self.sc_from)
        self.new_indexes2barcode_screen = pickle.load(open('{}/gdsc_ctrp_{}_new_indexes_newIndex2barcodeScreen_dict.pkl'.format(self.path_data, self.sc_from), 'rb'))
        
        global seed
        if seed != self.seed:
            self.set_seed(self.seed)

        #add information to report
        if self.run_type == 'resume':
            lines = ['\n', '*** RESUME FOR MORE {} EPOCHS*** \n'.format(list_parameters[19])]
        else:
            lines = ['** REPORT - DRUG SENSITIVITY **\n',
                    '* Parameters',
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

        self.ccle_per_barcode = pickle.load(open('{}/ccle_per_barcode_dict.pkl'.format(self.path_data), 'rb'))
        self.barcodes_per_cell_line = pickle.load(open('{}/barcodes_per_cell_line_dict.pkl'.format(self.path_data), 'rb'))
        
        if self.run_type != 'resume':
            self.save_parameters()

    # --------------------------------------------------

    def set_seed(self, value):
        global seed
        seed = value
        np.random.seed(value)
        torch.manual_seed(value)
        random.seed(value)

    # --------------------------------------------------

    def get_indexes(self):
        if self.type_of_split == 'random':
            final_indexes = list(self.new_indexes2barcode_screen.keys())
            final_indexes = shuffle(final_indexes)

            train_number = int(self.perc_train * len(final_indexes))
            validation_number = int(self.perc_val * len(final_indexes))

            train_set = final_indexes[:train_number]
            validation_set = final_indexes[train_number:train_number+validation_number]
            test_set = final_indexes[train_number+validation_number:]

        elif self.type_of_split == 'random7':
            final_indexes = list(self.new_indexes2barcode_screen.keys())
            final_indexes = shuffle(final_indexes)

            test_set = []
            other_set = []
            j = 0
            for i in range(0, len(final_indexes), int(len(final_indexes)/7)):
                if j+1 == int(self.to_test):
                    test_set.extend(final_indexes[i:int(i+len(final_indexes)/7)])
                else:
                    other_set.extend(final_indexes[i:int(i+len(final_indexes)/7)])
                j += 1

            train_set = other_set[:-len(test_set)]
            validation_set = other_set[-len(test_set):]

        with open('pickle/Train_set_index.txt', 'w') as f:
            f.write('\n'.join(train_set))
        with open('pickle/Validation_set_index.txt', 'w') as f:
            f.write('\n'.join(validation_set))
        with open('pickle/Test_set_index.txt', 'w') as f:
            f.write('\n'.join(test_set))

        print(len(train_set), len(validation_set), len(test_set))
        lines = ['\n** DATASETS **',
                 'Training set: {}'.format(len(train_set)),
                 'Validation set: {}'.format(len(validation_set)),
                 'Test set: {}'.format(len(test_set)),
                 '\n']
        create_report(self.filename_report, lines)
        return train_set, validation_set, test_set

    # --------------------------------------------------

    def get_batches(self, ccle_data, drug_bottlenecks, indexes, type_dataset):
        data = []
        sensitivity = []
        for index in indexes:
            ccle, screen, sens = self.new_indexes2barcode_screen[index]
            data.append(np.concatenate((ccle_data[ccle[0]], drug_bottlenecks[screen[0]]), axis=None))
            sensitivity.append([sens])
            if type_dataset == 'Train':
                self.train_barcodes.append((ccle[1], ccle[0], screen[1], screen[0], sens))
            elif type_dataset == 'Validation':
                self.validation_barcodes.append((ccle[1], ccle[0], screen[1], screen[0], sens))
            else:
                self.test_barcodes.append((ccle[1], ccle[0], screen[1], screen[0], sens))
            
        return torch.Tensor(data).type('torch.FloatTensor'), torch.Tensor(np.array(sensitivity)).type('torch.FloatTensor')

    # --------------------------------------------------

    def save_dataset(self, type_dataset):
        with open('pickle/{}_output.txt'.format(type_dataset), 'r') as f:
            output = f.readlines()
            output = [x.strip('\n') for x in output]

        with open('pickle/{}_set_barcodes.txt'.format(type_dataset), 'r') as f:
            values = f.readlines()
            values = [x.strip('\n') for x in values]

        sens_list = []
        with open('pickle/{}_set_total.txt'.format(type_dataset), 'w') as f:
            f.write('\t'.join(['cell_line', 'screen_id', 'real_sensitivity', 'predicted_sensitivity']))
            for i in range(len(values)):
                ccle, _, screen, _, sens = values[i].split('\t')
                if ':::' in screen:
                    screen = screen.split(':::')[0]
                sens_list.append(str(sens))
                f.write('\n{}\t{}\t{}\t{}'.format(ccle, screen, sens, output[i]))
        
        with open('pickle/{}_set_real_values.txt'.format(type_dataset), 'w') as f:
            f.write('\n'.join(sens_list))

    # --------------------------------------------------

    def initialize_model(self, size_input):
        if self.model_architecture == 'NNet':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(self.device)
            self.first_layer = int(size_input)
            model = NN_drug_sensitivity(input_size=self.first_layer,
                                        layers=self.layers_info,
                                        dropout_prob=self.dropout)
            model.to(self.device)
            lines = ['\n*About the network',
                     'Type of network: Neural Network',
                     '{}'.format(str(model.modules())),
                     'Runs on: {} \n'.format(self.device)]

        elif self.model_architecture == 'lGBM':
            model = LGBMRegressor(n_estimators=self.number_trees, random_state=self.seed, learning_rate=self.learning_rate)
            model.set_params(metric = 'mse')
            lines = ['\n*About the network',
                     'Type of network: Light Gradient Boosted Machine',
                     'Number of trees: {} \n'.format(self.number_trees)]
        
        elif self.model_architecture == 'yrandom':
            model = LGBMRegressor(n_estimators=self.number_trees, random_state=self.seed, learning_rate=self.learning_rate)
            model.set_params(metric = 'mse')
            lines = ['\n*About the network',
                     'Type of network: Light Gradient Boosted Machine (Y-scrambling)',
                     'Number of trees: {} \n'.format(self.number_trees)]
        
        elif self.model_architecture == 'linear':
            model = LinearRegression()
            lines = ['\n*About the network',
                     'Type of network: Multiple Linear Regression']

        #save parameters as a pkl file
        if self.run_type != 'resume':
            create_report(self.filename_report, lines)

        return model

    # --------------------------------------------------

    def __train_validation_nnet(self, model, ccle_data, drug_bottlenecks, train_set_index, validation_set_index):
        if self.run_type == 'start':
            n_epochs_not_getting_better = 0
            best_epoch = None
            results = {'loss_values_training':{},
                       'loss_values_validation':{},
                       'learning_rates':{},
                       'times_training':{},
                       'times_validation':{}}
            start_point = 0
        elif self.run_type == 'resume':
            results = pickle.load(open('pickle/Training_Validation_results.pkl', 'rb'))
            start_point = len(list(results['loss_values_validation'].keys()))-1
            for i in range(start_point+1):
                if i == 0:
                    best_loss = (results['loss_values_training'][i], results['loss_values_validation'][i])
                    n_epochs_not_getting_better = 0
                else:
                    loss = results['loss_values_validation'][i]
                    if loss < best_loss[1]:
                        best_loss = (results['loss_values_training'][i], results['loss_values_validation'][i])
                        n_epochs_not_getting_better = 0
                    else:
                        n_epochs_not_getting_better += 1
                        
            model = self.load_model(model)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        best_model = copy.deepcopy(model.state_dict())  # save the best model yet with the best accuracy and lower loss value
        decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        # Training and Validation
        for epoch in range(start_point, self.n_epochs):
            if (epoch + 1) % self.epoch_reset == 0 and epoch != (self.n_epochs - 1):
                print('-' * 10)
                print('Epoch: {} of {}'.format(epoch + 1, self.n_epochs))
                if epoch != 0:
                    optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
                    decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

            # epoch learning rate value
            results['learning_rates'][epoch] = optimizer.state_dict()['param_groups'][0]['lr']

            # TRAINING
            start_train_time = time.time()
            model, optimizer, train_loss_epoch = self.__train_nnet__(model, optimizer, ccle_data, drug_bottlenecks, train_set_index, 'Train')
            end_train_model = time.time()
            results['loss_values_training'][epoch] = train_loss_epoch
            results['times_training'][epoch] = end_train_model - start_train_time
            
            # VALIDATION
            start_validation_time = time.time()
            validation_loss_epoch = self.__validation_and_test_nnet__(model, ccle_data, drug_bottlenecks, validation_set_index, 'Validation')
            end_validation_time = time.time()
            results['loss_values_validation'][epoch] = validation_loss_epoch
            results['times_validation'][epoch] = end_validation_time - start_validation_time
            
            if epoch == 0 or validation_loss_epoch < best_loss[1]:  # means that this model is best one yet
                best_loss = (train_loss_epoch, validation_loss_epoch)
                best_model = copy.deepcopy(model.state_dict())
                self.best_train_barcodes = self.train_barcodes
                self.best_validation_barcodes = self.validation_barcodes
                n_epochs_not_getting_better = 0
                pickle.dump(best_model, open('pickle/best_model_parameters.pkl', 'wb'))
                best_epoch = copy.copy(epoch)
            
            else:
                n_epochs_not_getting_better += 1
            
            self.train_barcodes = []
            self.validation_barcodes = []
                
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

            pickle.dump(results, open('pickle/Training_Validation_results.pkl', 'wb'))

            if self.early_stop and n_epochs_not_getting_better == self.early_stop:
                create_report(self.filename_report, ['\nWarning!!! Training stopped because the loss was not improving.'])
                break
        
        # Saving the results
        results = pd.DataFrame.from_dict(results)
        results.columns = ['Loss_Training', 'Loss_Validation', 'Learning_rates', 'Duration_Training', 'Duration_Validation']
        results.to_csv('Training_Validation_results.txt', header=True, index=True)

        model.load_state_dict(best_model)

        print('Training: Done!')
        lines = ['\nTraining loss: {:.2f}'.format(best_loss[0]),
                'Validation loss: {:.2f}'.format(best_loss[1]),
                'Number of epochs: {:.0f} of {:.0f} \n'.format(epoch + 1, self.n_epochs)]
        create_report(self.filename_report, lines)

        return model

    # --------------------------------------------------

    def __train_nnet__(self, model, optimizer, ccle_data, drug_bottlenecks, train_set_index, type_dataset):
        train_loss_epoch = 0.0
        model.train()  # set model for training
        n_batches = 0
        for i in range(0, len(train_set_index), self.size_batch):
            inputs, real_values = self.get_batches(ccle_data, drug_bottlenecks, train_set_index[i:int(i + self.size_batch)], type_dataset)
            inputs = inputs.to(self.device)
            real_values = real_values.to(self.device)
            optimizer.zero_grad()  # set the gradients of all parameters to zero
            train_predictions = model(inputs)  # output predicted by the model
            train_current_loss = self.__loss_function(real_values, train_predictions)
            train_current_loss.backward()  # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss_epoch += train_current_loss.item()
            n_batches += 1

        train_loss_epoch = train_loss_epoch / n_batches

        return model, optimizer, train_loss_epoch

    # --------------------------------------------------

    def __validation_and_test_nnet__(self, model, ccle_data, drug_bottlenecks, dataset_index, type_dataset):
        loss_epoch = 0.0
        model.eval()
        n_batches = 0
        with torch.no_grad():
            for i in range(0, len(dataset_index), self.size_batch):
                inputs, real_values = self.get_batches(ccle_data, drug_bottlenecks, dataset_index[i:int(i + self.size_batch)], type_dataset)
                inputs = inputs.to(self.device)
                real_values = real_values.to(self.device)
                predictions = model(inputs)  # output predicted by the model
                current_loss = self.__loss_function(real_values, predictions)
                loss_epoch += current_loss.item()
                if type_dataset == 'Test':
                    with open('pickle/{}_output.txt'.format(type_dataset), 'a') as f:
                        if i != 0:
                            f.write('\n')
                        f.write('\n'.join([str(x[0]) for x in predictions.detach().cpu().numpy().tolist()]))
                n_batches += 1

        loss_epoch = loss_epoch / n_batches
        return loss_epoch
        
    # --------------------------------------------------
    
    def __get_predictions_nnet__(self, model, ccle_data, drug_bottlenecks, type_dataset):
        if type_dataset == 'Train':
            indexes = self.best_train_barcodes
            with open('pickle/Train_set_barcodes.txt', 'w') as f:
                f.write('\n'.join(['\t'.join(map(str,x)) for x in self.best_train_barcodes]))
        else:
            indexes = self.best_validation_barcodes
            with open('pickle/Validation_set_barcodes.txt', 'w') as f:
                f.write('\n'.join(['\t'.join(map(str,x)) for x in self.best_validation_barcodes]))

        model.eval()
        with torch.no_grad():
            with open('pickle/{}_output.txt'.format(type_dataset), 'w') as f:
                for i in range(len(indexes)):
                    _, ccle_id, _, screen_id, sens = indexes[i]
                    inputs = torch.Tensor(np.concatenate((ccle_data[int(ccle_id)], drug_bottlenecks[int(screen_id)]), axis=None)).type('torch.FloatTensor').to(self.device)
                    predictions = model(inputs)
                    if i != 0:
                        f.write('\n')
                    f.write('{}'.format(str(predictions.detach().cpu().numpy().tolist()[0])))

    # --------------------------------------------------

    def __train_lgbm_or_linear(self, model, ccle_data, drug_bottlenecks, train_set_index):
        X_train = []
        y_real = []
        
        for index in train_set_index:
            ccle, screen, sens = self.new_indexes2barcode_screen[index]
            X_train.append(np.concatenate((ccle_data[ccle[0]], drug_bottlenecks[screen[0]]), axis=None))
            y_real.append([sens])
            self.train_barcodes.append((ccle[1], ccle[0], screen[1], screen[0], sens))
        
        if self.model_architecture == 'yrandom':
            y_real = shuffle(y_real)
        
        model.fit(np.array(X_train), np.array(y_real))
        
        y_pred = model.predict(X_train)
        
        y_pred = y_pred.tolist()
        
        if self.model_architecture == 'linear':
            lines = ['\n Values of the model: {}'.format(model.get_params()),
                 '\n']
            create_report(self.filename_report, lines)
            with open('pickle/Train_output.txt', 'w') as f:
                f.write('\n'.join(['{:f}'.format(x[0]) for x in y_pred]))

        else:
            with open('pickle/Train_output.txt', 'w') as f:
                f.write('\n'.join(['{:f}'.format(x) for x in y_pred]))

        with open('pickle/Train_set_barcodes.txt', 'w') as f:
            f.write('\n'.join(['\t'.join(map(str,x)) for x in self.train_barcodes]))

        del X_train
        del y_real
        del y_pred
        gc.collect()

        return model

    # --------------------------------------------------

    def train_model(self, model, ccle_data, drug_bottlenecks, train_set_index, validation_set_index = []):
        start_training = time.time()

        if self.model_architecture == 'NNet':
            model = self.__train_validation_nnet(model, ccle_data, drug_bottlenecks, train_set_index, validation_set_index)

        elif self.model_architecture == 'lGBM' or self.model_architecture == 'yrandom' or self.model_architecture == 'linear':
            model = self.__train_lgbm_or_linear(model, ccle_data, drug_bottlenecks, train_set_index)
        

        end_training = time.time()
        create_report(self.filename_report, ['Duration: {:.2f} \n'.format(end_training - start_training)])
        
        if self.model_architecture == 'NNet':
            self.__get_predictions_nnet__(model, ccle_data, drug_bottlenecks, 'Train')
            self.__get_predictions_nnet__(model, ccle_data, drug_bottlenecks, 'Validation')
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
        elif self.model_architecture == 'lGBM' or self.model_architecture == 'yrandom' or self.model_architecture == 'linear':
            pickle.dump(model, open('pickle/drug_sensitivity_model.pkl', 'wb'))

    # --------------------------------------------------

    def load_model(self, model):
        if self.model_architecture == 'NNet':
            model_parameters = pickle.load(open('pickle/drug_sensitivity_model.pkl', 'rb'))
            model.load_state_dict(model_parameters)
        elif self.model_architecture == 'lGBM' or self.model_architecture == 'yrandom' or self.model_architecture == 'linear':
            model = pickle.load(open('pickle/drug_sensitivity_model.pkl', 'rb'))

        return model

    # --------------------------------------------------

    def __run_test_set_nnet(self, model, ccle_data, drug_bottlenecks, test_set_index):
        test_loss = self.__validation_and_test_nnet__(model, ccle_data, drug_bottlenecks, test_set_index, 'Test')
        with open('pickle/Test_set_barcodes.txt', 'w') as f:
            f.write('\n'.join(['\t'.join(map(str,x)) for x in self.test_barcodes]))

        print('Test loss: {:.2f} \n'.format(test_loss))
        create_report(self.filename_report, ['Testing loss: {:.2f}'.format(test_loss)])

    # --------------------------------------------------

    def __run_test_set_lgbm_or_linear(self, model, ccle_data, drug_bottlenecks, test_set_index):
        y_real = []
        X = []
        with open('pickle/Test_set_barcodes.txt', 'w') as file:
            for index in test_set_index:
                ccle, screen, sens = self.new_indexes2barcode_screen[index]
                X.append(np.concatenate((ccle_data[ccle[0]], drug_bottlenecks[screen[0]]), axis=None))
                y_real.append([sens])
                file.write('{}\t{}\t{}\t{}\t{}\n'.format(ccle[1], ccle[0], screen[1], str(screen[0]), str(sens)))

        y_pred = model.predict(np.array(X))
        y_pred = y_pred.tolist()

        with open('pickle/Test_output.txt', 'w') as f:
            if self.model_architecture == 'linear':
                f.write('\n'.join(['{:f}'.format(x[0]) for x in y_pred]))
            else:
                f.write('\n'.join(['{:f}'.format(x) for x in y_pred]))

        mse = mean_squared_error(np.array(y_real), np.array(y_pred))
        print('Mean squared error: {:.2f}'.format(mse))

        lines = ['\n \nTesting loss: {:.2f}'.format(mse),
                 '\n']
        create_report(self.filename_report, lines)

    # --------------------------------------------------

    def run_test_set(self, model, ccle_data, drug_bottlenecks, test_set_index):
        if self.model_architecture == 'NNet':
            self.__run_test_set_nnet(model, ccle_data, drug_bottlenecks, test_set_index)
        elif self.model_architecture == 'lGBM' or self.model_architecture == 'yrandom' or self.model_architecture == 'linear':
            self.__run_test_set_lgbm_or_linear(model, ccle_data, drug_bottlenecks, test_set_index)

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
        if list_parameters[19] == 'new':
            type_smiles_VAE = ''
        else:
            type_smiles_VAE = '_{}'.format(list_parameters[19])
        filename_output = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(list_parameters[15], list_parameters[0], list_parameters[1], list_parameters[2], list_parameters[3], list_parameters[4], list_parameters[5], list_parameters[6], list_parameters[7], list_parameters[8], list_parameters[9], list_parameters[10], list_parameters[11], list_parameters[12], list_parameters[13], list_parameters[16])
        self.filename_report = '{}{}/{}/output_{}.txt'.format(list_parameters[14], type_smiles_VAE, list_parameters[15], filename_output)
        return self.filename_report

    # --------------------------------------------------

    def get_model_architecture(self):
        return self.model_architecture

    # --------------------------------------------------

    def get_path_data(self):
        return self.path_data

    # --------------------------------------------------

    def get_sc_from(self):
        return self.sc_from

    # --------------------------------------------------
    
    def save_parameters(self):
        if self.model_architecture == 'NNet':
            layers_info = [str(x) for x in self.layers_info]
            layers_info.pop(0)
            if layers_info[-1] == '1':
                network_info = '_'.join(layers_info[:-1])
            else:
                network_info = '_'.join(layers_info)
            list_parameters = [network_info, np.format_float_positional(self.learning_rate), str(self.size_batch), str(self.n_epochs), str(self.perc_train),
                         str(self.perc_val), str(self.dropout), str(self.gamma), str(self.step_size), str(self.seed), str(self.epoch_reset), self.type_of_split,
                         str(self.to_test), self.type_lr, self.data_from, self.model_architecture, self.early_stop]

        else:
            list_parameters = [str(self.number_trees), np.format_float_positional(self.learning_rate), str(self.size_batch),str(self.n_epochs), str(self.perc_train),
                         str(self.perc_val), str(self.dropout), str(self.gamma), str(self.step_size), str(self.seed), str(self.epoch_reset), self.type_of_split,
                         str(self.to_test), self.type_lr, self.data_from, self.model_architecture, self.early_stop]

        if self.type_smiles_VAE != '':
            list_parameters.append(self.type_smiles_VAE.strip('_'))

        pickle.dump(list_parameters, open('pickle/list_initial_parameters_single_cell.pkl', 'wb'))

# -------------------------------------------------- RUN --------------------------------------------------

def run_drug_prediction(list_parameters, run_type):
    start_run = time.time()
    print(str(datetime.datetime.now().time()))
    drug_sens = Drug_sensitivity_single_cell()
    
    if run_type == 'resume':
        more_epoch = list_parameters[-2]
        list_parameters = pickle.load(open('pickle/list_initial_parameters_single_cell.pkl', 'rb'))
        filename = drug_sens.create_filename(list_parameters)
        list_parameters.extend([run_type, more_epoch])
    else:
        filename = drug_sens.create_filename(list_parameters)
    
    drug_sens.set_parameters(list_parameters)
    model_architecture = drug_sens.get_model_architecture()
    path_data = drug_sens.get_path_data()
    sc_from = drug_sens.get_sc_from()
    
    if run_type == 'start':
        train_set_index, validation_set_index, test_set_index = drug_sens.get_indexes()

    elif run_type == 'resume':
        with open('pickle/Train_set_index.txt', 'r') as f:
            train_set_index = f.readlines()
            train_set_index = [x.strip('\n') for x in train_set_index]
        if model_architecture == 'NNet':
            with open('pickle/Validation_set_index.txt', 'r') as f:
                validation_set_index = f.readlines()
                validation_set_index = [x.strip('\n') for x in validation_set_index]
        with open('pickle/Test_set_index.txt', 'r') as f:
            test_set_index = f.readlines()
            test_set_index = [x.strip('\n') for x in test_set_index]
    
    #load and process the datasets
    ccle_data = pd.read_csv('{}/ccle_dataset.csv'.format(path_data), header = None, index_col = 0)
    if list_parameters[-1] == 'new':
        drug_bottlenecks = pd.read_csv('{}/molecular/gdsc_ctrp_bottlenecks.csv'.format(path_data), header = None, index_col = 0, sep = ';')
    elif list_parameters[-1] == 'fp':
        drug_bottlenecks = pd.read_csv('{}/molecular/gdsc_ctrp_fp.csv'.format(path_data), index_col = 0)
    else:
        drug_bottlenecks = pd.read_csv('{}/molecular/gdsc_ctrp_bottlenecks_old.csv'.format(path_data), header = None, index_col = 0, sep = '\t')
    n_genes = int(ccle_data.shape[1] + drug_bottlenecks.shape[1])
    ccle_data = ccle_data.to_numpy()
    drug_bottlenecks = drug_bottlenecks.to_numpy()
    
    #start the Drug Sensitivity model
    if model_architecture == 'NNet':
        model = drug_sens.initialize_model(size_input=n_genes)
    elif model_architecture == 'lGBM' or model_architecture == 'yrandom' or model_architecture == 'linear':
        model = drug_sens.initialize_model(size_input=[])
    
    #train the model
    model_trained = drug_sens.train_model(model, ccle_data, drug_bottlenecks, train_set_index, validation_set_index)
    drug_sens.run_test_set(model_trained, ccle_data, drug_bottlenecks, test_set_index)

    #add the predicted values to the final dataset
    drug_sens.save_dataset('Train')
    if model_architecture == 'NNet':
        drug_sens.save_dataset('Validation')
    drug_sens.save_dataset('Test')

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
    if model_architecture == 'NNet':
        type_dataset = ['Train', 'Validation', 'Test']
    elif model_architecture == 'lGBM' or model_architecture == 'yrandom' or model_architecture == 'linear':
        type_dataset = ['Train', 'Test']

    for i in range(len(type_dataset)):
        with open('pickle/{}_output.txt'.format(type_dataset[i]), 'r') as f:
            predictions = f.readlines()
        with open('pickle/{}_set_real_values.txt'.format(type_dataset[i]), 'r') as f:
            real_values = f.readlines()
        predictions = np.array([float(x.strip('\n')) for x in predictions])
        real_values = np.array([float(x.strip('\n')) for x in real_values])
        if type_dataset[i] == 'Validation':
            plot_name = '{}_set'.format(type_dataset[i])
            lines = ['\n{} real values max: {:.2f}'.format(type_dataset[i], real_values.max()),
                     '{} real values min: {:.2f}'.format(type_dataset[i], real_values.min()),
                     '{} predicted values max: {:.2f}'.format(type_dataset[i], predictions.max()),
                     '{} predicted values min: {:.2f}'.format(type_dataset[i], predictions.min())]
        else:
            plot_name = '{}ing_set'.format(type_dataset[i])
            lines = ['\n{}ing real values max: {:.2f}'.format(type_dataset[i], real_values.max()),
                     '{}ing real values min: {:.2f}'.format(type_dataset[i], real_values.min()),
                     '{}ing predicted values max: {:.2f}'.format(type_dataset[i], predictions.max()),
                     '{}ing predicted values min: {:.2f}'.format(type_dataset[i], predictions.min())]
        corr_value, _ = pearsonr(real_values, predictions)
        make_correlation_plot(real_values, predictions, plot_name)
        if type_dataset[i] == 'Validation':
            lines.extend(['\n{} correlation: {}'.format(type_dataset[i], corr_value), '\n'])
        else:
            lines.extend(['\n{}ing correlation: {}'.format(type_dataset[i], corr_value), '\n'])
        create_report(filename, lines)
    
    end_run = time.time()
    create_report(filename, ['\nTime for run: {:.2f}'.format(end_run - start_run)])
    print('Done!')

# -------------------------------------------------- INPUT --------------------------------------------------

try:
    input_values = sys.argv[1:]
    run_type = input_values[-2]
    run_drug_prediction(input_values, run_type)

except EOFError:
    print('ERROR!')