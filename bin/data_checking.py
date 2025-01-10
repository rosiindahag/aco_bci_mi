import json
from config import *
import os
import mne
import matplotlib.pyplot as plt
import glob

def find_data(data, start_idx, length_idx):
    for iteration in data['frequency_band_0']['training_result'].values():
        for ant, ant_data in iteration.items():
            if 'window_params' in ant_data:
                window_params = ant_data['window_params']
                if (window_params['start_idx'] == start_idx and 
                    window_params['length_idx'] == length_idx):
                    return ant_data['accuracy'], ant_data['mean'], ant_data['sd'], ant_data['composite_score']
    return None, None, None

##############################
subject = "2a_A02"

for i in range(16):
    path = rf"{subject}\results_26*-{i+1}.json"
    file_list = glob.glob(path)
    acc = []
    a = []
    b = []
    for file_ in file_list:
        # Open and read the JSON file
        with open(file_, 'r') as file:
            data = json.load(file)

        # Print the data
        print(file_)
        a.append(data["ACO_params"]["alpha"])
        b.append(data["ACO_params"]["beta"])
        print(data["ACO_params"]["alpha"], data["ACO_params"]["beta"])
        print(data['frequency_band_0']['training_result']["iter_100"]["best_solution"])
        print(data['frequency_band_0']['testing_result'])

        # # Find mean and sd for 10-fold cross validation
        # start_idx = 138
        # length_idx = 72
        # list_cv, mean, sd, comp_score = find_data(data, start_idx, length_idx)

        # if mean is not None and sd is not None:
        #     print(f"CV:{list_cv}, Mean: {mean}, SD: {sd}")
        # else:
        #     print("No matching data found.")


    