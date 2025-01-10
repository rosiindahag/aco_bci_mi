"""

The results that are used:
1. aco_nofbcsp: ACO with two filter bank (8-13 Hz and 13-30 Hz)
    it has 10 ants and each ant has 100 iterations
2. random_nofbcsp: random search with two filter bank (8-13 Hz and 13-30 Hz)
    it has 100 iterations
3. fixedtime_nofbcsp: fix time with start idx=0 and window length idx=500
    start time = 0 ms and duration = 2000 ms or 2 s
    use the *_2 for 8-13 Hz
    and the *_3 for 13-30 Hz

"""

from config import *
import os
import json
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

def find_data(data, start_idx, length_idx, fmin, fmax):
    for iteration in data[f'frequency_band_{fmin}-{fmax}']['training_result'].values():
        for ant, ant_data in iteration.items():
            if 'window_params' in ant_data:
                window_params = ant_data['window_params']
                if (window_params['start_idx'] == start_idx and 
                    window_params['length_idx'] == length_idx):
                    return ant_data['accuracy'], ant_data['mean'], ant_data['sd'], ant_data['composite_score']
    return None, None, None, None

def find_data_ft(data, start_idx, length_idx, fmin, fmax):
    window_params = data[f'frequency_band_{fmin}-{fmax}']['training_result']['iter_0']['local_best']['window_params']
    if (window_params['start_idx'] == start_idx and 
        window_params['length_idx'] == length_idx):
        return window_params['accuracy'], window_params['mean'], window_params['sd'], window_params['composite_score']
    return None, None, None, None

def getTopTwoIdx(data_list):
    return [i[::-1] for i in sorted(enumerate(data_list), key=lambda x: x[::-1], reverse=True)][:2]

"""
1. Line plot (only allch_aco_nofbcsp and allch_random_nocsp)
    x-axis: iteration
    y-axis: accuracy

"""
dataset_name = os.path.basename(os.path.dirname(DATASET1_PATH))[5:7]

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.tight_layout(pad=5)
colors=['#018571','#a6611a','#80cdc1','#dfc27d']
print(colors)
x = [i+1 for i in range(100)]
freqband=[(8,13),(13,30)]
s=0
for d in range(3):
    for j in range(3):
        subject = rf"2a_A0{s+1}"
        z=0
        for k in range(2):
            path = rf"{subject}\allch_*_nofbcsp\results_*-{k+1}.json"
            file_list = glob.glob(path)
            for file_ in file_list:
                if "fixedtime" in file_:
                    continue
                # Open and read the JSON file
                with open(file_, 'r') as file:
                    data = json.load(file)
                alg = file_.split("\\")[1]
                alg = alg.split("_")[1]
                fmin=freqband[k][0]
                fmax=freqband[k][1]
                # Print the data
                y=[data[f'frequency_band_{fmin}-{fmax}']['training_result'][f"iter_{i+1}"]["best_solution"]["best_score"] for i in range(100)]
                
                axs[d,j].set_title(f'Subject {subject.split("_")[1]}', fontdict={'fontsize': 10})
                axs[d,j].plot(x,y,label=f'{alg}_{fmin}-{fmax}Hz',linewidth=5, color=colors[z])# alpha=0.7
                # axs[j].set_yticks([i/10 for i in range(0,10,1)],[i/10 for i in range(0,10,1)])
                axs[d,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                # axs[j].set_ylim(ymax=1) 
                axs[d,j].set_xlabel('Iteration', fontsize=10)
                axs[d,j].set_ylabel('Mean accuracy', fontsize=10)
                axs[d,j].tick_params(axis='both', labelsize=10)
                axs[d,j].legend(loc='lower right',fontsize=10)
                z+=1
        s+=1

# plt.subplots_adjust(top=0.85)
# fig.savefig('result1_conf_allch_lineplot_dataset1_1.png', dpi=300)
plt.show()

"""
1.1 Line plot (only allch_aco_nofbcsp and allch_random_nocsp)
    2x2 fig
    x-axis: iteration
    y-axis: accuracy

"""
dataset_name = os.path.basename(os.path.dirname(DATASET1_PATH))[5:7]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.tight_layout(pad=5)
colors=['#018571','#a6611a','#80cdc1','#dfc27d']
# colors=['blue','red','green','orange']
print(colors)
x = [i+1 for i in range(100)]
freqband=[(8,13),(13,30)]
included=[4,5,6,8]
s=0
for d in range(2):
    for j in range(2):
        subject = rf"2a_A0{included[s]}"
        z=0  
        for k in range(2):
            path = rf"{subject}\allch_*_nofbcsp\results_*-{k+1}.json"
            file_list = glob.glob(path)
            for file_ in file_list:
                if "fixedtime" in file_:
                    continue
                # Open and read the JSON file
                with open(file_, 'r') as file:
                    data = json.load(file)
                alg = file_.split("\\")[1]
                alg = alg.split("_")[1]
                fmin=freqband[k][0]
                fmax=freqband[k][1]
                # Print the data
                y=[data[f'frequency_band_{fmin}-{fmax}']['training_result'][f"iter_{i+1}"]["best_solution"]["best_score"] for i in range(100)]
                
                axs[d,j].set_title(f'Subject S{included[s]}', fontdict={'fontsize': 15})
                axs[d,j].plot(x,y,label=f'{alg}_{fmin}-{fmax}Hz',linewidth=5, color=colors[z], alpha=0.8)
                # axs[j].set_yticks([i/10 for i in range(0,10,1)],[i/10 for i in range(0,10,1)])
                axs[d,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                # axs[j].set_ylim(ymax=1) 
                axs[d,j].set_xlabel('Iteration', fontsize=15)
                axs[d,j].set_ylabel('Score', fontsize=15)
                axs[d,j].tick_params(axis='both', labelsize=15)
                axs[d,j].legend(loc='lower right',fontsize=10)
                z+=1
        s+=1
        
# plt.subplots_adjust(top=0.85)
fig.savefig('result1_conf_allch_lineplot_dataset1_11.png', dpi=300)
plt.show()

"""
1. Table

"""
# column_name = ['no', 'subject', '']
final_result=[]
for j in range(9):
    subject = rf"2a_A0{j+1}"
    temp_subdata = {}
    temp_subdata["subject"]=subject

    for k in range(2):
        path = rf"{subject}\allch_*_nofbcsp\results_*-{k+1}.json"
        file_list = glob.glob(path)
        
        for file_ in file_list:
            iter_i=100
            if "fixedtime" in file_:
                iter_i=0
            # Open and read the JSON file
            with open(file_, 'r') as file:
                data = json.load(file)
            alg = file_.split("\\")[1]
            alg = alg.split("_")[1]
            fmin=freqband[k][0]
            fmax=freqband[k][1]
            
            # Assign the data
            best_start_train = data[f"frequency_band_{fmin}-{fmax}"]["training_result"][f"iter_{iter_i}"]["best_solution"]["best_start_idx"]
            best_window_train = data[f"frequency_band_{fmin}-{fmax}"]["training_result"][f"iter_{iter_i}"]["best_solution"]["best_window_length"]
            best_start_test = data[f"frequency_band_{fmin}-{fmax}"]["testing_result"]["start_idx"]
            best_window_test = data[f"frequency_band_{fmin}-{fmax}"]["testing_result"]["window_length"]
            # print(best_start_train, best_window_train, best_start_test, best_window_test)
            
            if (best_start_train==best_start_test) and (best_window_train==best_window_test):
                test_acc = data[f"frequency_band_{fmin}-{fmax}"]["testing_result"]["test_accuracy"]
                if iter_i == 0:
                    train_accuracy, mean, sd, composite_score = find_data_ft(data,best_start_train,best_window_train,fmin,fmax)
                else:
                    train_accuracy, mean, sd, composite_score = find_data(data,best_start_train,best_window_train,fmin,fmax)
                # best_solutions.append((best_start_train,best_window_train))
                temp_subdata[f'start_time_ms_{alg}_{fmin}-{fmax}']=best_start_train*4
                temp_subdata[f'duration_ms_{alg}_{fmin}-{fmax}']=best_window_train*4
                temp_subdata[f'end_time_ms_{alg}_{fmin}-{fmax}']=best_start_train*4+best_window_train*4
                temp_subdata[f"train_10_fold_acc_{alg}_{fmin}-{fmax}"]=train_accuracy
                temp_subdata[f"compscore_{alg}_{fmin}-{fmax}"]=composite_score
                temp_subdata[f"mean_acc_{alg}_{fmin}-{fmax}"]=mean
                temp_subdata[f"sd_acc_{alg}_{fmin}-{fmax}"]=sd
                temp_subdata[f"test_acc_{alg}_{fmin}-{fmax}"]=test_acc
                # print(file_)
                # print(train_accuracy, mean, sd, composite_score, test_acc)
    final_result.append(temp_subdata)

df = pd.DataFrame(final_result)
df.to_csv('allch_result_dataset1.csv', encoding='utf-8', index=False)

print(df)


            