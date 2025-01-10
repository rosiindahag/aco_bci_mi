from init import *
import init
from bin.aco import *

if __name__=="__main__":

    logging.basicConfig(level=logging.INFO)

    mne.set_log_level('WARNING')

    np.random.seed(0)

    file_path = DATASET1_PATH
    dataset_name = os.path.basename(os.path.dirname(DATASET1_PATH))[5:7]
    eog = DATASET1_EOG
    channel_mapping = DATASET1_CHANNEL_MAPPING
    std = DATASET1_STD
    event_dict = DATASET1_EVENT_DICT
    # selected_channels = ['C3', 'Cz', 'C4']
    selected_channels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC4', 'C5', 'F8', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    init.min_freq = 8.0 #13
    init.max_freq = 13.0 #30

    le = LabelEncoder()

    for i in DATASET1_USED:
        file_name_train = DATASET1_SUBJECT[i]
        file_name_test = DATASET1_SUBJECT[i-1]

        init.subject = file_name_train.split(".")[0][:3]

        print(f"SUBJECT: {init.subject}")
        init.folder_path = f"{dataset_name}_{init.subject}\\nofbcsp_lda\\"

        if not os.path.exists(init.folder_path):
            if not os.path.exists(init.folder_path.split("\\")[0]):
                os.mkdir(init.folder_path.split("\\")[0])
            os.mkdir(init.folder_path)

        raw_train, events_train, event_id_train = load_raw_file(file_path, file_name_train, channel_mapping, std, selected_channels, event_dict, eog)
        raw_test, events_test, event_id_test = load_raw_file(file_path, file_name_test, channel_mapping, std, selected_channels, event_dict, eog)
        sfreq = raw_train.info['sfreq']

        min_window_length = int(40 * sfreq / 1000)
        print(f"Processing frequency band {init.min_freq}-{init.max_freq} Hz")
            
        # Filter the data for the current frequency band (FIR)    
        raw_train_filter = raw_train.copy().filter(init.min_freq, init.max_freq, fir_design="firwin", skip_by_annotation="edge", verbose=False)
        raw_test_filter = raw_test.copy().filter(init.min_freq, init.max_freq, fir_design="firwin", skip_by_annotation="edge", verbose=False)

        epochs_train = get_epoch(raw_train_filter, events_train, [7,8])
        epochs_test = get_epoch(raw_test_filter, events_test, [11])

        y_train = le.fit_transform(epochs_train.events[:, -1])
        X_train = epochs_train.get_data(copy=False)
        
        temp_X_test = epochs_test.get_data(copy=False)

        y_test_dict = get_test_label(DATASET1_PATH_LABEL, f'{init.subject}E', 288, 1, 2) #288 trials, class1=left, class2=right
        y_test = [label for label in y_test_dict.values()]

        y_test = le.fit_transform(y_test)
        y_test_idx = [idx for idx in y_test_dict.keys()]

        X_test = [temp_X_test[i] for i in y_test_idx]
        X_test = np.array(X_test)
            
        signal_length = X_train.shape[2]

        dict_data = {    
            "dataset":dataset_name,    
            "subject":init.subject,
            "ACO_params":{  "num_ants":aco_params["num_ants"], 
                            "num_iterations":aco_params["num_iterations"], 
                            "min_window_length":min_window_length, 
                            # "max_window_length":max_window_length,
                            "max_window_length":signal_length, 
                            "alpha":aco_params["alpha"][3], #z
                            "beta":aco_params["beta"][0], #a
                            "evaporation_rate":0.08}}

        max_start_idx = signal_length - min_window_length  # Ensure valid start indices
        dict_data["ACO_params"]["signal_length"] = signal_length
        dict_data["ACO_params"]["num_starts"] = max_start_idx

        # Initialize ACO with the given parameters
        aco = ACO(**dict_data["ACO_params"])

        print(f'train data shape:{np.shape(X_train)}, test data shape:{np.shape(X_test)}')

        # Run ACO to find the best parameters
        best_window_length, best_start_idx, best_score, temp_result_iter_dict = aco.optimize(lambda window_length, start_idx: 
            evaluate_performance(X_train, y_train, start_idx, start_idx + window_length))

        print(f"Best window length: {best_window_length}")
        print(f"Best start index: {best_start_idx}")
        print(f"Best score: {best_score}")

        train_scores, test_results = evaluate_performance(
            X_train, y_train, best_start_idx, best_start_idx + best_window_length, X_test, y_test
        )

        # Print results
        print(f"Final Train Scores (CV): {train_scores}")
        print(f"Test Accuracy: {test_results['accuracy']}")
        print(f"Kappa Score: {test_results['kappa_score']}")
        dict_data[f"frequency_band_{init.min_freq}-{init.max_freq}"] = {"training_result":temp_result_iter_dict,
                                            "testing_result":{
                                                "start_idx":int(best_start_idx),
                                                "window_length":int(best_window_length),
                                                "test_accuracy":float(np.round(test_results['accuracy'],2)),
                                                "kappa_score":float(np.round(test_results['kappa_score'],2))
                                                }}
        vers = 1
        jsonfile_name = f"{init.folder_path}results_nofbcsplda_"+datetime.now().strftime('%d%m%Y')+f"-{vers}.json"

        while os.path.isfile(jsonfile_name):
            vers += 1
            jsonfile_name = jsonfile_name.split(".")[0].split("-")[0]+f"-{vers}.json"
            
        # Convert and write JSON object to file
        with open(jsonfile_name, "a") as outfile: 
            json.dump(dict_data, outfile)