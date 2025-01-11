# Time Window Segmentation for Motor Imagery EEG Classification using Ant Colony Optimization (ACO)

This project explores the optimization of time window parameters (start time and duration) for EEG signal processing with **Ant Colony Optimization (ACO)**. The results are compared against fixed-time and random-based approaches. 

![Proposed Method](prop_method.png)

## Methodology
Key steps and methodologies include:

- **EEG Signal Preprocessing**:  
  Using **Finite Impulse Response (FIR)** filtering for noise reduction.  
- **Feature Extraction & Classification**:  
  - Feature extraction using **Common Spatial Pattern (CSP)**.  
  - Classification using **Linear Discriminant Analysis (LDA)**.  
- **Optimization**:  
  - Time window parameter optimization with **Ant Colony Optimization (ACO)**.  
  - Comparisons with fixed-time and random-based approaches.  

The project uses the **BCI Competition IV** datasets:  
- **[Dataset 2a](https://www.bbci.de/competition/iv/#dataset2a)**  
- **[Dataset 2b](https://www.bbci.de/competition/iv/#dataset2b)**

## Configuration
Before running the project, ensure the dataset directories are correctly configured. Modify the `config.py` file to set the paths for the datasets.  

Example:
```python
# config.py
DATASET1_PATH = "/path/to/dataset_2a"
DATASET2_PATH = "/path/to/dataset_2b"
```
## Requirements (see Pipfile)
- Python 3.11
- numpy 1.26.4
- mne
- scikit-learn
- matplotlib

## Project Structure
The project files are organized as follows:

### File Descriptions

#### Core Scripts
- **`bin/config.py`**  
  Configuration file for setting up paths to the datasets and some variables. Modify this file to set the paths for Dataset 2a and Dataset 2b.

- **`bin/aco.py`**  
  Contains the implementation of **Ant Colony Optimization (ACO)** for optimizing time window parameters in EEG signals. 

- **`bin/data_checking.py`**  
  To check the result data

- **`bin/load_file.py`**
  Contains classes and functions for loading EEG raw data.

- **`bin/main_visualization.py`**
  To analyze and visualize the results.

- **`init.py`**  
  Contains the main pipeline functions (CSP for feature extraction, LDA for classification, and cross-validation).

- **`main_nofb_aco.py`**  
  The main file for running the full pipeline (with signal filtering using FIR) and finding the optimal time window parameters of raw EEG signals using ACO.
  
- **`main_nofb_fixedtime.py`**  
  The main file for running the full pipeline (with signal filtering using FIR) using fixed time window parameters.  
  To change the start time and duration, modify the following parameters:  
  - **`start_idx`**: Defines the start time (in seconds).  
  - **`window_length`**: Defines the duration (in seconds).  
  Example:
  ```python
          dict_data = {    
            "dataset": dataset_name,    
            "subject": init.subject,
            "FT_params": {   
                            "start_idx": 0.5 * sfreq, 
                            "window_length": 2 * sfreq, 
                          }
          }
- **`main_nofb_random.py`**  
  The main file for running the full pipeline (with signal filtering using FIR) and finding the optimal time window parameters of raw EEG signals using pure random search.
  
#### Other Files
- **`Pipfile`**  
  Contains the Python dependencies and environment configuration for the project. Use `pipenv install` to set up the environment.

## Reference
R. I. Agustin and P. Raphisak, "Optimized Time Window Segmentation for Motor Imagery EEG Classification: An Ant Colony Optimization-Based Approach," *Proceedings of the 13th International Winter Conference on Brain-Computer Interface*, 2025, accepted for publication.
