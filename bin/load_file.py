import mne
import os
import logging

class DatasetFileLoader:
    """
    A class to load dataset files.

    Attributes:
        file_path (str): Path to the dataset directory.
        file_name (str): Name of the dataset file.
        eog (list): List of EOG channels.
        raw (mne.io.Raw): The raw EEG data object.
    """
    def __init__(self, file_path, file_name, eog):
        self.file_path = file_path
        self.file_name = file_name
        self.eog = eog
        full_path = os.path.join(self.file_path, self.file_name)
        self.raw = mne.io.read_raw_gdf(full_path, eog=eog, preload=True)

    def load_file(self):
        """
        Loads the dataset file into an MNE Raw object.

        Returns:
            mne.io.Raw: The raw EEG data.
        """
        logging.info("Loading file...")
        logging.info(self.raw.info)
        return self.raw

class DatasetPreprocessor:    
    """
    A class to preprocess raw EEG dataset.

    Attributes:
        raw (mne.io.Raw): The raw EEG data object.
    """
    def __init__(self, raw):
        self.raw = raw

    def rename_channels(self, channel_mapping):
        try:
            self.raw.rename_channels({ch: new_ch for ch, new_ch in channel_mapping.items() if ch in self.raw.ch_names})
            logging.info(f"Channel names have been changed: {self.raw.ch_names}")
        except Exception as e:
            logging.error(f"Error renaming channels: {e}")

    def set_channels(self, channels):
        try:
            self.raw.pick_channels(channels)
            logging.info(f"Picked channels: {channels}")
        except Exception as e:
            logging.error(f"Error picking channels: {e}")

    def set_montage(self, std_montage):
        try:
            self.raw.set_montage(std_montage, on_missing='warn')
            logging.info("Montage set successfully.")
        except Exception as e:
            logging.error(f"Error setting montage: {e}")

    def extract_events(self, event_dict):
        """
        Extract the events from annotations

        Returns:
            events and event_id (mne.events_from_annotations):
            events shape contains number of epochs, number of channels, number of time_sampling
            event_id contains the labels of each epoch 
            
        """
        try:
            events, event_id = mne.events_from_annotations(self.raw, event_id=event_dict)
            logging.info("Events extracted successfully.")
            return events, event_id
        except Exception as e:
            logging.error(f"Error extracting events: {e}")
            return None, None

def load_raw_file(file_path, file_name, channel_mapping, std, selected_channels, event_dict, eog):
    loader = DatasetFileLoader(file_path=file_path, file_name=file_name, eog=eog)

    raw = loader.load_file()

    preprocessor = DatasetPreprocessor(raw)
    preprocessor.rename_channels(channel_mapping)
    preprocessor.set_montage(std)
    preprocessor.set_channels(selected_channels)
    events, event_id = preprocessor.extract_events(event_dict)
    return raw, events, event_id

# def array_to_epoch(x,y,sfreq,info):
#     idx = 4*sfreq
#     events_new = np.column_stack((np.arange(x.shape[0])*int(idx),
#                                     np.zeros(x.shape[0], dtype=int),
#                                     y))
#     new_epochs = mne.EpochsArray(x, info, events=events_new, event_id={'left_hand':7,'right_hand':8})
#     return new_epochs

# def save_epoch(epochs, folder, filename):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     while os.path.exists(folder+"\\"+filename):
#         filename=filename.split(".")[0]
#         filename,i=filename.split("-")
#         i = int(i)
#         i += 1
#         filename += f"-{i}.fif"
#     print(epochs)
#     mne.Epochs.save(epochs, folder+"\\"+filename, split_size='2GB', fmt='double', overwrite=False, split_naming='neuromag', verbose=None)