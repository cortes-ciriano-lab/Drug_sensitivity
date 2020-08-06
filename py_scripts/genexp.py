# -------------------------------------------------- IMPORTS --------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import pandas as pd
import pickle
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.spatial import distance
import matplotlib._color_data as mcd
import umap

from full_network import AE_gene_expression

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- COLOR PALETTE --------------------------------------------------

color_palette = []
for k, v in mcd.XKCD_COLORS.items():
  if 'dark' in k and 'cream' not in k:
    color_palette.append(v)

# -------------------------------------------------- GENEXP --------------------------------------------------

class Genexp():
    
    def __init__(self):
      self.first_hidden = None
      self.second_hidden = None
      self.bottleneck_layer = None
      self.learning_rate = None
      self.input_size = None
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

    # --------------------------------------------------

    def __set_initial_parameters(self, list_parameters):
      layers = list_parameters[0]
      self.first_hidden, self.second_hidden, self.bottleneck_layer = layers.split('_')[0], layers.split('_')[1], layers.split('_')[2]
      self.learning_rate = list_parameters[1]
      self.input_size = list_parameters[2]
      self.size_batch = list_parameters[3]
      self.n_epochs = list_parameters[4]
      self.dropout = list_parameters[7]
      self.gamma = list_parameters[8]
      self.step_size = list_parameters[9]
      self.epoch_reset = list_parameters[11]
      self.seed = list_parameters[10]
      self.perc_train = list_parameters[5]
      self.perc_val = list_parameters[6]

    # --------------------------------------------------

    def __load_initial_parameters(self):
      list_parameters = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/list_initial_parameters_genexp.pkl', 'rb'))
      self.device = list_parameters[-1]
      self.__set_initial_parameters(list_parameters)

      if seed != self.seed:
        seed = self.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    # --------------------------------------------------
    
    def __load_datasets(self):
      whole_dataset = pd.read_csv("/hps/research1/icortes/acunha/data/CCLE/CCLE_expression.csv", sep=',', header=0, index_col=0)
      metadata = pd.read_csv('/hps/research1/icortes/acunha/data/CCLE/sample_info.csv', index_col=0, header=0, usecols=['DepMap_ID', 'disease', 'lineage'])  # information about the cell lines

      # check variance of each gene - only if necessary
      if self.input_size != 1.0:
        var_gene = pd.DataFrame(whole_dataset.var(), index=list(whole_dataset.columns), columns=['Gene_variance'])
        var_gene = var_gene.sort_values('Gene_variance', ascending=False)
        if int(float(self.input_size) * len(whole_dataset)) > 1500:
          size = int(float(self.input_size) * len(whole_dataset))
        else:
          size = 1500
        var_gene = list(var_gene.index)[:size]
        whole_dataset = whole_dataset.loc[:, var_gene]
        print('Number of genes: ', whole_dataset.shape[1])

      # Preprocessing the data
      whole_dataset = whole_dataset.loc[whole_dataset.index.isin(list(metadata.index.values))]  # filter - only cell lines with known origins
      cancers = []
      lineage = []
      for i in range(len(whole_dataset.index)):  # add the cancer type to the dataframe
        line = whole_dataset.iloc[i].name
        cancers.append(metadata['disease'].loc[line])
        lineage.append(metadata['lineage'].loc[line])
      whole_dataset['Cancer_type'] = cancers
      whole_dataset['Lineage'] = lineage
    
      # Split the dataset
      validation_number = int(self.perc_val * len(whole_dataset))
      train_number = int(self.perc_train * len(whole_dataset))
      train_set, validation_set, test_set = np.split(whole_dataset.sample(frac=1), [train_number, validation_number + train_number])

      pickle.dump(whole_dataset, open('pickle/whole_dataset.pkl', 'wb'))
      pickle.dump(train_set, open('pickle/train_set.pkl', 'wb'))
      pickle.dump(validation_set, open('pickle/validation_set.pkl', 'wb'))
      pickle.dump(test_set, open('pickle/test_set.pkl', 'wb'))

      print('Train dataset: {}, Validation dataset: {}, Test dataset: {}. \n'.format(len(train_set), len(validation_set), len(test_set)))

      return train_set, validation_set, test_set

    # --------------------------------------------------

    def __initialize_model(self, n_genes):
      if torch.cuda.is_available():
        self.device = 'cuda'
      else:
        self.device = 'cpu'

      model = AE_gene_expression(input_size=int(n_genes), first_hidden=self.first_hidden, second_hidden=self.second_hidden, bottleneck=self.bottleneck_layer, dropout_prob=self.dropout)
      model.to(self.device)

      return model

    # --------------------------------------------------

    def __train_validation(self, model, train_set, validation_set):
      # Divide the training dataset into batches
      if self.device == 'cuda':
        train_set_torch = torch.tensor(train_set.iloc[:, :-2].values).type('torch.cuda.FloatTensor')
        validation_set_torch = torch.tensor(validation_set.iloc[:, :-2].values).type('torch.cuda.FloatTensor')
      else:
        train_set_torch = torch.tensor(train_set.iloc[:, :-2].values).type('torch.FloatTensor')
        validation_set_torch = torch.tensor(validation_set.iloc[:, :-2].values).type('torch.FloatTensor')

      train_loader = torch.utils.data.DataLoader(train_set_torch, batch_size=self.size_batch, shuffle=False)
      validation_loader = torch.utils.data.DataLoader(validation_set_torch, batch_size=self.size_batch, shuffle=False)

      epoch_stop = int(2.3 * self.epoch_reset)
      got_better = False
      n_epochs_not_getting_better = 0
      
      optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)
      best_model = copy.deepcopy(model.state_dict()) #save the best model yet with the best accuracy and lower loss value
      decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size = self.step_size, gamma = self.gamma)

      # Save the loss values (plot) + learning rates of each epoch (plot)
      loss_values_training = {} #different loss values in different epochs (training)
      loss_values_validation = {} #different loss values in different epoch (validation)
      learning_rates = {} #learning rates values per epoch
      times_training = {} #time spent per epoch (training)
      times_validation = {} #time spent per epoch (validation)

      # Training and Validation
      for epoch in range(self.n_epochs):
        train_loss_epoch = 0.0
        validation_loss_epoch = 0.0
        
        if epoch % self.epoch_reset == 0 and epoch != (self.n_epochs-1):
          print('-'*10)
          print('Epoch: {} of {}'.format(epoch, self.n_epochs))
          if epoch != 0:
            optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)
            decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size = self.step_size, gamma = self.gamma)

        # epoch learning rate value
        learning_rates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']

        # TRAINING
        start_train_time = time.time()
        model.train() #set model for training
        for train_batch in train_loader:
          optimizer.zero_grad()  # set the gradients of all parameters to zero
          train_predictions = model(train_batch)  # output predicted by the model
          train_current_loss = self.__loss_function(train_batch, train_predictions[0])
          train_current_loss.backward()  # backpropagation
          optimizer.step()
          train_loss_epoch += train_current_loss.item()
        end_train_model = time.time()
            
        # epoch training values
        train_loss_epoch = train_loss_epoch / len(train_loader)  # loss value for this epoch (training)
        loss_values_training[epoch] = train_loss_epoch
        times_training[epoch] = end_train_model-start_train_time

        # VALIDATION
        start_validation_time = time.time()
        model.eval()
        validation_predictions_complete, validation_bottleneck_complete = [], []
        with torch.no_grad():
          for validation_batch in validation_loader:
            optimizer.zero_grad()  # set the gradients of all parameters to zero
            validation_predictions = model(validation_batch)  # output predicted by the model
            validation_current_loss = self.__loss_function(validation_batch, validation_predictions[0])
            validation_loss_epoch += validation_current_loss.item()
            validation_predictions_complete.append(validation_predictions[0])
            validation_bottleneck_complete.append(validation_predictions[1])
        end_validation_time = time.time()
        
        # epoch validation values
        validation_loss_epoch = validation_loss_epoch / len(validation_loader)
        loss_values_validation[epoch] = validation_loss_epoch  # loss value for this epoch (validation)
        times_validation[epoch] = end_validation_time-start_validation_time
        
        if epoch == 0 or validation_loss_epoch < best_loss[0]: #means that this model is best one yet
          best_loss = (validation_loss_epoch, train_loss_epoch)
          best_model = copy.deepcopy(model.state_dict())
          validation_predictions_with_best_model = [validation_predictions_complete, validation_bottleneck_complete]
          got_better = True
          n_epochs_not_getting_better = 0
        else:
          got_better = False
          n_epochs_not_getting_better +=1

        with open("model_values/loss_value_while_running.txt", 'w') as f:
          f.write('Epoch: {} \n'.format(epoch))
          f.write('Actual loss : {:.2f} \n'.format(validation_loss_epoch))
          f.write('Best loss: {:.2f} \n'.format(best_loss[0]))
          f.write('Time (training): {:.2f} \n'.format(end_train_model-start_train_time))
          f.write('Time (validation): {:.2f} \n'.format(end_validation_time-start_validation_time))
          f.write('Not getting better for {} epochs. \n'.format(n_epochs_not_getting_better))
        
        # decay the learning rate
        decay_learning_rate.step()
        
        if n_epochs_not_getting_better == epoch_stop:
          break

        if epoch % 10 == 0:
          results = pd.DataFrame.from_dict((loss_values_training, loss_values_validation, learning_rates, times_training, times_validation)).T
          results.columns = ['Loss_Training', 'Loss_Validation', 'Learning_rates', 'Duration_Training', 'Duration_Validation']
          self.__plot_loss_lr(list(list(results.index)), list(results['Loss_Training']), list(results['Loss_Validation']), list(results['Learning_rates']))

      # Saving the results
      validation_predictions_with_best_model[0] = torch.cat(validation_predictions_with_best_model[0], dim=0)
      validation_predictions_with_best_model[1] = torch.cat(validation_predictions_with_best_model[1], dim=0)
      pickle.dump(validation_set_torch, open('pickle/validation_set_torch.pkl', 'wb'))
      pickle.dump(validation_predictions_with_best_model[0], open('pickle/validation_output.pkl', 'wb'))
      pickle.dump(validation_predictions_with_best_model[1], open('pickle/validation_bottleneck.pkl', 'wb'))

      # plot the variation of the train loss, validation loss and learning rates
      results = pd.DataFrame.from_dict((loss_values_training, loss_values_validation, learning_rates, times_training, times_validation)).T
      results.columns = ['Loss_Training', 'Loss_Validation', 'Learning_rates', 'Duration_Training', 'Duration_Validation']
      results.reset_index().to_csv('Training_Validation_results.txt', header=True, index=False)

      self.__plot_loss_lr(list(list(results.index)), list(results['Loss_Training']), list(results['Loss_Validation']), list(results['Learning_rates']))

      results_barplots = results.loc[results.index % 10 == 0]
      results_barplots.loc[:, ['Duration_Training', 'Duration_Validation']].plot(kind = "bar", rot=0, subplots=True, figsize = (16, 8))
      plt.savefig('plots/Duration_per_epoch.png', bbox_inches='tight')
      model.load_state_dict(best_model)
      loss_training = best_loss[1]
      loss_validation = best_loss[0]

      print('Training: Done!')
      self.save_output_file(['Training loss: {:.2f} \n'.format(loss_training),
                             'Validation loss: {:.2f} \n'.format(loss_validation),
                             'Number of epochs: {:.0f} of {:.0f} \n'.format(epoch + 1, self.n_epochs)])

      return model, loss_training, loss_validation

    # --------------------------------------------------

    def __loss_function(self, x_input, x_output):
      criterion = nn.MSELoss()
      reconstruction_loss = criterion(x_output, x_input)  # current loss value
      return reconstruction_loss

    # --------------------------------------------------

    def __train_model(self, model, train_set, validation_set):
      start_training = time.time()
      model, loss_training, loss_validation = self.__train_validation(model, train_set, validation_set)
      end_training = time.time()
      self.save_output_file(['Duration: {:.2f} \n'.format(end_training - start_training)])
      self.__save_model(model)

      return model

    # --------------------------------------------------

    def __save_model(self, model):
      pickle.dump(model, open('pickle/genexp_model.pkl', 'wb'))

    # --------------------------------------------------

    def __load_model(self, model):
      model_parameters =  pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/genexp_model.pkl', 'rb'))
      model.load_state_dict(model_parameters)
      return model

    # --------------------------------------------------

    def __run_eval(self, model, dataset):
      # Divide the training dataset into batches
      if self.device == 'cuda':
        dataset_torch = torch.tensor(dataset.iloc[:, :-2].values).type('torch.cuda.FloatTensor')
      else:
        dataset_torch = torch.tensor(dataset.iloc[:, :-2].values).type('torch.FloatTensor')

      data_loader = torch.utils.data.DataLoader(dataset_torch, batch_size=self.size_batch, shuffle=False)
      data_loss = 0.0
      data_predictions_complete, data_bottleneck_complete = [], []
      model.eval()
      with torch.no_grad():
        for data_batch in data_loader:
          data_predictions = model(data_batch)  # output predicted by the model
          current_loss = self.__loss_function(data_batch, data_predictions[0])
          data_loss += current_loss.item()
          data_predictions_complete.append(data_predictions[0])
          data_bottleneck_complete.append(data_predictions[1])

      loss = data_loss / len(data_batch)
      data_predictions_complete = torch.cat(data_predictions_complete, dim=0)
      data_bottleneck_complete = torch.cat(data_bottleneck_complete, dim=0)

      return loss, data_predictions_complete, data_bottleneck_complete

    # --------------------------------------------------

    def __plot_loss_lr(self, x, y_training, y_validation, y2_learning_rates):

      minimum = min(min(y_validation), min(y_training))
      maximum = max(max(y_validation), max(y_training))

      fig = plt.figure(figsize=(10, 16))
      (ax1, ax2) = fig.subplots(2, 1, sharex=True)
      ax1.plot(x, y_training, color='r', label='Loss (training)')
      ax1.set_ylabel('Loss')
      ax1.plot(x, y_validation, color='g', label='Loss (validation)')
      ax1.set_ylim(minimum * 0.95, maximum)
      ax2.set_xlabel('Number of epochs')
      ax2.set_ylabel('Learning rate')
      ax2.plot(x, y2_learning_rates, color='b', label='Learning rates')
      fig.legend(loc=1)
      fig.tight_layout()
      plt.savefig('plots/Loss_learningRate_per_epoch.png', bbox_inches='tight')

    # --------------------------------------------------

    def do_tSNE(self, data, labels, type_of_data, before_training=True):
      for p in [5, 25, 50, 100, 150, 200]:
        for n in [250, 500, 1000]:
          for lr in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            for me in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean']:

              if before_training:
                filename_tsne_plot = 'plots/genexp/{}/tSNE/tSNE_BeforeTraining_{}_{}_{}_{}.png'.format(type_of_data, p, n, lr, me)
                auto_tsne = TSNE(init='pca', perplexity=p, n_iter=n, learning_rate=lr, random_state=0, metric=me)
                title = 'tSNE - Before Training ({})'.format(type_of_data)
              else:
                filename_tsne_plot = 'plots/genexp/{}/tSNE/tSNE_AfterTraining_{}_{}_{}_{}.png'.format(type_of_data, p, n, lr, me)
                auto_tsne = TSNE(perplexity=p, n_iter=n, learning_rate=lr, random_state=0, metric=me)
                title = 'tSNE - After Training ({})'.format(type_of_data)

              results_tsne = auto_tsne.fit_transform(data.values)

              plt.figure(figsize=(20, 10))
              plt.title(title, fontsize=14, fontweight='bold')
              sns.scatterplot(x=results_tsne[:, 0], y=results_tsne[:, 1], data=data, palette='deep',legend='full', hue=labels, size=labels, sizes=(20, 200))
              plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
              plt.savefig(filename_tsne_plot, bbox_inches='tight')
              plt.close('all')

      return results_tsne

    # --------------------------------------------------

    def do_tSNE2(self, data_before, data_after, labels, type_of_data):
      for p in [5, 25, 50, 100, 150, 200]:
        for n in [250, 500, 1000]:
          for lr in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            for me in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean']:
              auto_tsne_before = TSNE(init='pca', perplexity=p, n_iter=n, learning_rate=lr, random_state=0, metric=me)
              auto_tsne_after = TSNE(perplexity=p, n_iter=n, learning_rate=lr, random_state=0, metric=me)
              results_tsne_before = auto_tsne_before.fit_transform(data_before.values)
              results_tsne_after = auto_tsne_after.fit_transform(data_after.values)

              # Plot
              title1 = 'tSNE - Before Training ({})'.format(type_of_data)
              title2 = 'tSNE - After Training ({})'.format(type_of_data)

              fig = plt.figure(figsize=(20, 30))
              ax1, ax2 = fig.subplots(2)
              ax1.set_title(title1, fontsize=14, fontweight='bold')
              sns.scatterplot(x=results_tsne_before[:, 0], y=results_tsne_before[:, 1], data=data_before,
                              palette='deep', legend='full', hue=labels, size=labels, sizes=(20, 200), ax=ax1)
              ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
              ax2.set_title(title2, fontsize=14, fontweight='bold')
              sns.scatterplot(x=results_tsne_after[:, 0], y=results_tsne_after[:, 1], data=data_after,
                              palette='deep', legend='full', hue=labels, size=labels, sizes=(20, 200), ax=ax2)
              ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
              fig.savefig('plots/genexp/{}/tSNE/tSNE_{}_{}_{}_{}.png'.format(type_of_data, p, n, lr, me), bbox_inches='tight')
              plt.close('all')

      return results_tsne_before, results_tsne_after

    # --------------------------------------------------

    def do_UMAP(self, data, labels, type_of_data, before_training=True):
      for n_neigh in [5, 25, 50, 100, 150, 200]:
        for n_e in [100, 500, 1000]:
          for lr in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            for me in ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis', 'cosine',
                       'correlation', 'hamming', 'jaccard', 'dice', 'kulsinski', 'rogerstanimoto',
                       'sokalmichener', 'sokalsneath', 'yule']:
              for i in ['spectral', 'random']:

                umap_setups = umap.UMAP(n_neighbors=n_neigh, learning_rate=lr, n_epochs=n_e, init=i, metric=me, transform_seed=42)
                results_umap = umap_setups.fit_transform(data.values)

                '''Plot'''
                if before_training:
                  filename_umap_plot = 'plots/genexp/{}/UMAP/UMAP_BeforeTraining_{}_{}_{}_{}_{}.png'.format(type_of_data, n_neigh, n_e, lr, me, i)
                  title = 'UMAP - Before Training ({})'.format(type_of_data)
                else:
                  filename_umap_plot = 'plots/genexp/{}/UMAP/UMAP_AfterTraining_{}_{}_{}_{}_{}.png'.format(type_of_data, n_neigh, n_e, lr, me, i)
                  title = 'UMAP - After Training ({})'.format(type_of_data)

                plt.figure(figsize=(20, 10))
                plt.title(title, fontsize=14, fontweight='bold')
                sns.scatterplot(x=results_umap[:, 0], y=results_umap[:, 1], data=data, palette='deep', legend='full', hue=labels, size=labels, sizes=(20, 200))
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.savefig(filename_umap_plot, bbox_inches='tight')
                plt.close('all')

      return results_umap

    # --------------------------------------------------

    def do_UMAP2(self, data_before, data_after, labels, type_of_data):
      for n_neigh in [5, 25, 50, 100, 150, 200]:
        for n_e in [100, 500, 1000]:
          for lr in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            for me in ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis', 'cosine',
                       'correlation', 'hamming', 'jaccard', 'dice', 'kulsinski', 'rogerstanimoto',
                       'sokalmichener', 'sokalsneath', 'yule']:
              for i in ['spectral', 'random']:
                umap_setups = umap.UMAP(n_neighbors=n_neigh, learning_rate=lr, n_epochs=n_e, init=i, metric=me, transform_seed=42)
                results_umap_before = umap_setups.fit_transform(data_before.values)
                results_umap_after = umap_setups.fit_transform(data_after.values)

                '''Plot'''
                title1 = 'UMAP - Before Training (' + str(type_of_data) + ')'
                title2 = 'UMAP - After Training (' + str(type_of_data) + ')'

                fig = plt.figure(figsize=(20, 30))
                ax1, ax2 = fig.subplots(2)
                ax1.set_title(title1, fontsize=14, fontweight='bold')
                sns.scatterplot(x=results_umap_before[:, 0], y=results_umap_before[:, 1], data=data_before, palette='deep', legend='full', hue=labels, size=labels, sizes=(20, 200), ax=ax1)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_title(title2, fontsize=14, fontweight='bold')
                sns.scatterplot(x=results_umap_after[:, 0], y=results_umap_after[:, 1], data=data_after, palette='deep', legend='full', hue=labels, size=labels, sizes=(20, 200), ax=ax2)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                fig.savefig('plots/genexp/{}/tSNE/tSNE_{}_{}_{}_{}_{}.png'.format(type_of_data, n_neigh, n_e, lr, me, i), bbox_inches='tight')
                plt.close('all')
      return results_umap_before, results_umap_after

    # --------------------------------------------------

    def plot_heatmap(self, data, name_column, type_of_data, before_training=True):
      data = data.sort_values(name_column)
      new_indexes = {}
      for i in range(len(data)):
        old_index = data.iloc[i, :].name
        new_index = '{}_{}'.format(old_index, data[name_column].iloc[i])
        new_indexes[old_index] = new_index
      data = data.rename(new_indexes)
      color_dict = dict(zip(data.iloc[:, -1].unique(), color_palette))
      row_colors = data.iloc[:, -1].map(color_dict)

      #calculate the matrix distance
      matrix_distance = pd.DataFrame(np.zeros((len(data), len(data))), columns=list(data.index.values)).set_index(data.index.values)
      for i in range(len(data)):
        vector_x1 = data.iloc[i, :-1]
        cellline_x1 = vector_x1.name
        for j in range(i, len(data)):
          vector_x2 = data.iloc[j, :-1]
          cellline_x2 = vector_x2.name
          d = distance.euclidean(vector_x1, vector_x2)
          matrix_distance.loc[cellline_x1, cellline_x2] = d

      if before_training:
        filename_heatmap_plot = 'plots/Heatmap_BeforeTraining_{}.png'.format(type_of_data)
        title = 'Before Training ({})'.format(type_of_data)
      else:
        filename_heatmap_plot = 'plots/Heatmap_AfterTraining_{}.png'.format(type_of_data)
        title = 'After Training ({})'.format(type_of_data)

      plt.figure(figsize=(15, 15))
      plt.title(title, fontsize=14, fontweight='bold')
      heatmap = sns.heatmap(matrix)
      for tick_label_y in heatmap.get_yticklabels():
        tick_text = tick_label_y.get_text()
        label = data.loc[tick_text].name
        tick_label_y.set_color(row_colors[label])
      for tick_label_x in heatmap.get_xticklabels():
        tick_text = tick_label_x.get_text()
        label = data.loc[tick_text].name
        tick_label_x.set_color(row_colors[label])
      plt.savefig(filename_heatmap_plot, bbox_inches='tight')

    # --------------------------------------------------

    def save_output_file(self, list_comments):
      layers = "{}_{}_{}".format(self.first_hidden, self.second_hidden, self.bottleneck_layer)
      filename = "{}_{:f}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(layers, self.learning_rate, self.input_size, self.size_batch, self.n_epochs, self.perc_train, self.perc_val, self.dropout, self.gamma, self.step_size, self.epoch_reset)
      with open("/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/results/output_genexp_{}.txt".format(filename), 'a') as f:
        f.write(''.join(list_comments))

    # --------------------------------------------------

    def make_plots_whole_dataset(self, whole_dataset):
      do_tSNE(data=whole_dataset.iloc[:, :-2], labels=list(whole_dataset['Cancer_type']), type_of_data='Whole_dataset')
      do_UMAP(data=whole_dataset.iloc[:, :-2], labels=list(whole_dataset['Cancer_type']), type_of_data='Whole_dataset')

    # --------------------------------------------------

    def make_plots_validation(self, validation_set, validation_bottleneck):
      validation_after = pd.DataFrame(data = validation_bottleneck.numpy())
      do_tSNE2(validation_set.iloc[:, :-2], validation_after, list(validation_set['Cancer_type']), type_of_data='Validation')
      do_UMAP2(validation_set.iloc[:, :-2], validation_after, list(validation_set['Cancer_type']), type_of_data='Validation')

    # --------------------------------------------------

    def make_plots_test(self, test_set, test_bottleneck):
      test_after = pd.DataFrame(data = test_bottleneck.numpy())
      do_tSNE2(test_set.iloc[:, :-2], test_after, list(test_set['Cancer_type']), type_of_data='Test')
      do_UMAP2(test_set.iloc[:, :-2], test_after, list(v['Cancer_type']), type_of_data='Test')

      # HEATMAPS
      plot_heatmap(data=test_set.iloc[:, :-1], name_column='Cancer_type', type_of_data='Test')  # before Training
      test_after['Cancer_type'] = list(test_set['Cancer_type'])
      plot_heatmap(data=test_after, name_column='Cancer_type', type_of_data='Test', before_training=False)  # after Taining

    # --------------------------------------------------

    def run_genexp_only(self, list_parameters):
      self.__set_initial_parameters(list_parameters)
      train_set, validation_set, test_set = self.__load_datasets()
      model = self.__initialize_model(n_genes=train_set.iloc[:, :-2].shape[1])
      model_trained = self.__train_model(model, train_set, validation_set)
      loss_testing, test_output, test_bottleneck = self.__run_eval(model_trained, test_set)

      pickle.dump(test_set_torch, open('pickle/test_set_torch.pkl', 'wb'))
      pickle.dump(test_output, open('pickle/test_output.pkl', 'wb'))
      pickle.dump(test_bottleneck, open('pickle/test_bottleneck.pkl', 'wb'))

      print('Test loss: {:.2f} \n'.format(loss_testing))
      self.save_output_file(['Testing loss: {:.2f} \n'.format(loss_testing)])

      list_parameters.append(self.device)
      pickle.dump(list_parameters, open('pickle/list_initial_parameters_genexp.pkl', 'wb'))

      print('Done!')

    # --------------------------------------------------

    def initialize_gene_expression(self):
      self.__load_initial_parameters()

    # --------------------------------------------------

    def start_gene_expression(self, n_genes):
      model = self.__initialize_model(n_genes)
      model = self.__load_model(model)
      return model

    # --------------------------------------------------

    def run_train_set(self, model_trained, train_set):
      return self.__run_eval(model_trained, train_set)

    # --------------------------------------------------

    def run_validation_set(self, model_trained, validation_set):
      return self.__run_eval(model_trained, validation_set)

    # --------------------------------------------------

    def run_test_set(self, model_trained, test_set):
      return self.__run_eval(model_trained, test_set)

    # --------------------------------------------------