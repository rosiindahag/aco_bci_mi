from init import *
import init

class FixedTime:
    def __init__(self, start_idx, window_length):
        self.start_idx = int(start_idx)
        self.window_length = int(window_length)
    def calculate(self, evaluate_function):
        def composite_score(scores):
            mean_sc = round(np.mean(np.array(scores)),2)
            sd_sc = round(np.std(np.array(scores)),2)
            comp_score = mean_sc-(sd_sc*0.1)
            return mean_sc, sd_sc, comp_score

        best_score = -np.inf
        best_window_length = None
        best_start_idx = None
        temp_result_iter = {}

        scores = []
        temp_result = {}
        
        end_idx=self.start_idx+self.window_length
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'start idx: {self.start_idx}, window length idx: {self.window_length}, end idx: {end_idx}')
        print(f'start (ms): {self.start_idx*4}, window length (ms): {self.window_length*4}, end (ms): {end_idx*4}')
        train_scores, pipeline = evaluate_function(self.window_length, self.start_idx)
            
        ### calculate composite score ####
        mean_sc, sd_sc, score = composite_score(train_scores)
        temp_result["local_best"] = {"window_params":{
                                            "start_idx":int(self.start_idx),
                                            "length_idx":int(self.window_length),
                                            "end_idx":int(end_idx),
                                            "accuracy":[float(i) for i in train_scores],
                                            "mean":float(mean_sc),
                                            "sd":float(sd_sc),
                                            "composite_score":float(score)
                                            }}
        scores.append(score)

        if score > best_score:
            best_score = score
            best_window_length = self.window_length
            best_start_idx = self.start_idx

        temp_result["best_solution"] = {"best_score":float(np.round(best_score,2)),
                                                "best_start_idx":int(best_start_idx),
                                                "best_window_length":int(best_window_length),
                                                "best_end_idx":int(best_start_idx+best_window_length)
                                            }
        temp_result_iter[f"iter_0"] = temp_result

        print(f'Best score = {best_score}')

        return best_window_length, best_start_idx, best_score, temp_result_iter

if __name__ == "__main__":
    file_path = DATASET1_PATH
    dataset_name = os.path.basename(os.path.dirname(DATASET1_PATH))[5:7]
    eog = DATASET1_EOG
    channel_mapping = DATASET1_CHANNEL_MAPPING
    std = DATASET1_STD
    event_dict = DATASET1_EVENT_DICT
    # selected_channels = ['C3', 'Cz', 'C4']
    selected_channels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC4', 'C5', 'F8', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    init.min_freq = 8 #13
    init.max_freq = 13 #30

    le = LabelEncoder()

    for i in DATASET1_USED:
        file_name_train = DATASET1_SUBJECT[i]
        file_name_test = DATASET1_SUBJECT[i-1]

        init.subject = file_name_train.split(".")[0][:3]

        print(f"SUBJECT: {init.subject}")
        init.folder_path = f"{dataset_name}_{init.subject}\\allch_fixedtime_nofbcsp\\"

        if not os.path.exists(init.folder_path):
            if not os.path.exists(init.folder_path.split("\\")[0]):
                os.mkdir(init.folder_path.split("\\")[0])
            os.mkdir(init.folder_path)

        raw_train, events_train, event_id_train = load_raw_file(file_path, file_name_train, channel_mapping, std, selected_channels, event_dict, eog)
        raw_test, events_test, event_id_test = load_raw_file(file_path, file_name_test, channel_mapping, std, selected_channels, event_dict, eog)
        sfreq = raw_train.info['sfreq']

        # Filter the data for the current frequency band (FIR)    
        min_window_length = int(40 * sfreq / 1000)  # Minimum window length (convert ms to samples)
        print(f"Processing frequency band {init.min_freq}-{init.max_freq} Hz")

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
            "FT_params":{   "start_idx":0.5*sfreq, 
                            "window_length":2*sfreq, 
                            }}

        # Initialize ACO with the given parameters
        ft = FixedTime(**dict_data["FT_params"])

        print(f'train data shape:{np.shape(X_train)}, test data shape:{np.shape(X_test)}')

        # Optimize start_idx and window_length using ACO
        best_window_length, best_start_idx, best_score, temp_result_iter_dict = ft.calculate(lambda window_length, start_idx: 
                                                                    evaluate_performance(X_train, y_train, start_idx, start_idx + window_length))
        print(f'Best window length: {best_window_length}, Best start index: {best_start_idx}, Best score: {best_score}')

        train_scores, test_results = evaluate_performance(
            X_train, y_train, best_start_idx, best_start_idx + best_window_length, X_test, y_test
        )

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
        # save result
        vers = 1
        jsonfile_name = f"{init.folder_path}results_fixedtimenofb_"+datetime.now().strftime('%d%m%Y')+f"-{vers}.json"

        while os.path.isfile(jsonfile_name):
            vers += 1
            jsonfile_name = jsonfile_name.split(".")[0].split("-")[0]+f"-{vers}.json"
 
        # Convert and write JSON object to file
        with open(jsonfile_name, "a") as outfile: 
            json.dump(dict_data, outfile)