import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle

list_datasets = os.listdir('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc_ctrp')
# list_datasets = [x for x in list_datasets if 'ccle' in x]
list_datasets = [x for x in list_datasets if 'centroids-bottlenecks' in x or 'ccle' in x]
for case in list_datasets:
    if case not in os.listdir('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/plots_general'):
        os.makedirs('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/plots_general/{}'.format(case))
    for arch in ['lGBM']: #, 'NNet']:
        path_dataset = '/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/new_results_gdsc_ctrp/{}/{}'.format(case, arch)
        for type_run in ['leave-mcFarland-out','random7', 'leave-one-cell-line', 'leave-one-drug', 'leave-one-tumour']:
            type_models = os.listdir(path_dataset)
            type_models = [x for x in type_models if type_run in x and 'txt' not in x] #
            for group_by in ['cell_line', 'drug']:
                try:
                    dataset = {}
                    for model in type_models:
                        if group_by == 'drug':
                            data = pd.read_csv('{}/{}/pickle/Test_set_total.txt'.format(path_dataset, model), sep = '\t', usecols = ['screen_id', 'real_sensitivity', 'predicted_sensitivity'])
                            print(data.head(5))
                            print(data.screen_id.unique().shape)
                            for drug in data.screen_id.unique():
                                if drug not in dataset:
                                    dataset[drug] = {'real_sensitivity':[], 'predicted_sensitivity':[]}
                                dataset[drug]['real_sensitivity'].extend(list(data.loc[(data.screen_id == drug), 'real_sensitivity']))
                                dataset[drug]['predicted_sensitivity'].extend(list(data.loc[(data.screen_id == drug), 'predicted_sensitivity']))
                        else:
                            data = pd.read_csv('{}/{}/pickle/Test_set_total.txt'.format(path_dataset, model), sep = '\t', usecols = ['cell_line', 'real_sensitivity', 'predicted_sensitivity'])
                            print(data.head(5))
                            print(data.cell_line.unique().shape)
                            for cell in data.cell_line.unique():
                                if cell not in dataset:
                                    dataset[cell] = {'real_sensitivity':[], 'predicted_sensitivity':[]}
                                dataset[cell]['real_sensitivity'].extend(list(data.loc[(data.cell_line == cell), 'real_sensitivity']))
                                dataset[cell]['predicted_sensitivity'].extend(list(data.loc[(data.cell_line == cell), 'predicted_sensitivity']))
                    
                    pickle.dump(dataset, open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/plots_general/{}/Full_dataset_{}_{}_{}_{}.pkl'.format(case, group_by, case, arch, type_run), 'wb'))
                    
                    corr_values = {}
                    if group_by == 'cell_line' and 'integrated' in case and type_run == 'leave-mcFarland-out':
                        print('a')
                        mcfarland = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/integrated/list_mcfarland_cells.pkl', 'rb'))
                        pancancer = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data_gdsc_ctrp/integrated/list_pancancer_cells.pkl', 'rb'))
                        cell_lines = list(set(mcfarland).difference(pancancer))
                        list_keys = list(set(dataset.keys()).difference(cell_lines))
                    else:
                        list_keys = list(dataset.keys())
                    for k in list_keys:
                        rho, _ = stats.spearmanr(np.array(dataset[k]['real_sensitivity']), np.array(dataset[k]['predicted_sensitivity']))
                        corr_values[k] = rho
                
                    corr_values = pd.DataFrame.from_dict(corr_values, orient = 'index')
                    corr_values.columns = ['rho']
                    corr_values.to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/plots_general/{}/Correlation_per_{}_{}_{}_{}.csv'.format(case, group_by, case, arch, type_run), header=True, index=True)
                    corr_values.dropna(inplace=True)
                    plt.figure(figsize=(8,6), constrained_layout=True)
                    corr_values.plot(kind = 'kde')
                    plt.title('Spearmanc correlation per {}'.format(group_by.replace('_', ' ')), fontsize=14)
                    plt.xlabel('Spearman correlation', fontsize=14)
                    plt.ylabel('Density', fontsize=14)
                    plt.legend(fontsize=14)
                    plt.savefig('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/plots_general/{}/Density_correlation_plots_per_{}_{}_{}_{}.png'.format(case, group_by, case, arch, type_run), dpi=200)
                    print('1 created')
                
                    count_values = {}
                    for i in np.arange(0.0, 1.0, 0.1):
                        count_values[']{:.1f}-{:.1f}]'.format(i, i+0.1)] = corr_values.loc[(corr_values.rho > i) & (corr_values.rho <= i+0.1), 'rho'].shape[0]
                    count_values['{}0'.format(r'$\leq$')] = corr_values.loc[corr_values.rho <= 0, 'rho'].shape[0]
                    count_values_d = pd.DataFrame(count_values.keys())
                    count_values_d.columns = ['Count']
                    count_values_d['rho'] = count_values.values()
                    count_values_d.sort_values(['Count'], inplace = True)
                    count_values_d.to_csv('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/plots_general/{}/Count_correlation_per_{}_{}_{}_{}.csv'.format(case, group_by, case, arch, type_run), header=True, index=True)
                
                    plt.figure(figsize=(8,6), constrained_layout=True)
                    sns.barplot(x='Count', y='rho', data=count_values_d, palette=sns.color_palette('ch:start=.2,rot=-.3', 11))
                    plt.xlabel('Spearman correlation', fontsize=14)
                    plt.ylabel('Count', fontsize=14)
                    plt.title('Spearman correlation per {}'.format(group_by.replace('_', ' ')), fontsize=14)
                    plt.savefig('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/plots_general/{}/Count_correlation_plots_per_{}_{}_{}_{}.png'.format(case, group_by, case, arch, type_run), dpi=200)
                    print('2 created')
                
                except:
                    pass