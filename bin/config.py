import os

# DATASET 1: BCI COMPETITION IV 2008 DATASET 2A
DATASET1_PATH=r'D:\\S2\\Thailand\\KU_TAIST 2023\\thesis_mi_bci\\bci_mi_code\\bciiv2a_dataset\\'
DATASET1_SUBJECT = [f for f in os.listdir(DATASET1_PATH) if os.path.isfile(os.path.join(DATASET1_PATH, f))]
DATASET1_EOG = ['EOG-left', 'EOG-central', 'EOG-right']
DATASET1_CHANNEL_MAPPING = {
                            'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC4', 
                            'EEG-4': 'C5', 'EEG-5': 'F8', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 
                            'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 
                            'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-Pz': 'Pz', 
                            'EEG-15': 'P2', 'EEG-16': 'POz', 'EOG-left': 'EOG-left', 'EOG-central': 'EOG-central', 
                            'EOG-right': 'EOG-right'
                          }
DATASET1_STD = 'standard_1020'
DATASET1_EVENT_DICT = {
    '1023': 1,   # reject
    '1072': 2,   # eye move
    '276': 3,    # eye open
    '277': 4,    # eye close
    '32766': 5,  # new run
    '768': 6,    # new trial
    '769': 7,    # class 1 left hand
    '770': 8,    # class 2 right hand
    '771': 9,    # class 3 foot
    '772': 10,   # class 4 tongue
    '783': 11,   # unknown
}
DATASET1_USED = [i for i in range(1,9*2,2)]
DATASET1_PATH_LABEL = r'D:\\S2\\Thailand\\KU_TAIST 2023\\thesis_mi_bci\\bci_mi_code\\true_label\\'

# DATASET 2: BCI COMPETITION IV 2008 DATASET 2B
DATASET2_PATH=r'D:\\S2\\Thailand\\KU_TAIST 2023\\thesis_mi_bci\\bci_mi_code\\bciiv2b_dataset\\'
DATASET2_SUBJECT = [f for f in os.listdir(DATASET2_PATH) if os.path.isfile(os.path.join(DATASET2_PATH, f))]
DATASET2_EOG = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
DATASET2_CHANNEL_MAPPING = {
                            'EEG:C3': 'C3', 'EEG:Cz': 'Cz', 'EEG:C4': 'C4', 
                            'EOG:ch01': 'EOG-left', 'EOG:ch02': 'EOG-central', 
                            'EOG:ch03': 'EOG-right'
                          }
DATASET2_STD = 'standard_1020'
DATASET2_EVENT_DICT = {
    '1023': 1,   # reject
    '1077': 2,   # eye move horizontal
    '1078': 3,     # eye move vertical
    '1079': 4,   # eye rotation
    '1081': 12,   # eye blinks
    '32766': 5,  # new run
    '768': 6,    # new trial
    '769': 7,    # class 1 cue onset left
    '770': 8,    # class 2 cue onset right
    '276': 9,    # eyes open
    '277': 10,   # eyes closed
    '783': 11,   # unknown
}
# DATASET2_USED = [j for i in range(9) for j in (5*i, 5*i+1)]
DATASET2_USED = [(5*i, 5*i+1) for i in range(9)]

# CONFIG FOR ACO PARAMS
aco_params = {"num_ants": 5,
              "num_iterations": 100,
              "alpha": [10.0 ** i for i in range(-3,1,1)],
              "beta": [10.0 ** i for i in range(-3,1,1)],
              "evaporation_rate": [i/10.0 for i in range(1,11,1)]
              }