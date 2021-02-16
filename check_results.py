import os
import shutil
'''
num_genes = ['all_genes'] #, 'best_7000']
pathways = ['no_pathway'] #, 'canonical_pathways', 'chemical_genetic_perturbations', 'kegg_pathways']
architectures = ['lGBM'] #, 'NNet', 'RF', 'yrandom', 'linear']
combination = ['integrated_old', 'integrated_fp', 'integrated', 'pancancer_old', 'pancancer_fp', 'pancancer'] #total = 6
for j in range(len(combination)):
    for n in num_genes:
        for pathway in pathways:
            for arch in architectures:
                path = 'new_results_gdsc_ctrp2/gdsc_ctrp_{}/{}_{}/{}'.format(combination[j], n, pathway, arch)
                # path = 'new_results/secondary/pancancer_ic50/{}_{}/{}'.format(n, pathway, arch)
                #path = 'new_results_gdsc/gdsc_pancancer/{}_{}/{}'.format(n, pathway, arch)
                print(path)
                i = 0
                try:
                    files = os.listdir(path)
                    outputs = [x for x in files if 'txt' in x]
                    folders = [x for x in files if 'txt' not in x]
                    for folder in folders:
                        output = 'output_{}.txt'.format(folder)
                        if output not in outputs:
                            shutil.rmtree('{}/{}'.format(path, folder))
                            i += 1
                        else:
                            values = open('{}/{}'.format(path, output), 'r')
                            values = values.readlines()
                            values = values[-1].split(' ')
                            if values[0] != 'Time':
                                os.remove('{}/{}'.format(path, output))
                                shutil.rmtree('{}/{}'.format(path, folder))
                                i += 1
                    print(i)
                except:
                    pass
'''

architectures = ['lGBM', 'NNet']
list_folders = os.listdir('new_results_gdsc_ctrp')
list_folders = [x for x in list_folders if 'integrated-centroids-bottlenecks' in x]
# list_folders = [x for x in list_folders if 'ccle' in x]
for j in range(len(list_folders)):
    for arch in architectures:
        path = 'new_results_gdsc_ctrp/{}/{}'.format(list_folders[j], arch)
        print(path)
        i = 0
        try:
            files = os.listdir(path)
            outputs = [x for x in files if 'txt' in x]
            folders = [x for x in files if 'txt' not in x]
            for folder in folders:
                output = 'output_{}.txt'.format(folder)
                if output not in outputs:
                    shutil.rmtree('{}/{}'.format(path, folder))
                    i += 1
                else:
                    values = open('{}/{}'.format(path, output), 'r')
                    values = values.readlines()
                    values = values[-1].split(' ')
                    if values[0] != 'Time' or 'drug_sensitivity_model.pkl' not in os.listdir('{}/{}/pickle'.format(path, folder)):
                        os.remove('{}/{}'.format(path, output))
                        shutil.rmtree('{}/{}'.format(path, folder))
                        print('{}/{}'.format(path, output))
                        i += 1
                            
            print(i)
        except:
            pass
