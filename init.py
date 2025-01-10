"""

This file is used to load the raw data (BCI Competition IV dataset 2a),
filter the data based on frequency bands,
extract the epoch data,
split it into training and testing sets,
and finally save the epoch data.

"""
from bin.config import *
from bin.load_file import *
from datetime import datetime
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import mne
from mne import Epochs
from mne.decoding import CSP
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import json
import warnings
import scipy.io as sio
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

logging.basicConfig(level=logging.INFO)

mne.set_log_level('WARNING')

np.random.seed(0)

def get_test_label(folder_path, subject, num_epochs, *labels):
    mat=sio.loadmat(f'{folder_path}/{subject}.mat')
    y_label = mat['classlabel'].reshape(num_epochs)
    y_labels={i:label for i,label in enumerate(y_label) if label==labels[0] or label==labels[1]}
    return y_labels

def get_epoch(raw, events, event_id):
    epochs= Epochs(
        raw, events, event_id=event_id, tmin=0, tmax=4, proj=False, baseline=None, preload=True)
    epochs.drop_bad()
    return epochs
    
def evaluate_performance(X_train, y_train, start_idx, end_idx, X_test=None, y_test=None, n_components=4):
    global subject
    global folder_path
    X_crop = X_train[:, :, start_idx:end_idx]  # Crop the time window
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_scores = []

    for train_idx, val_idx in cv.split(X_crop, y_train):
        # Split data into training and test sets for the fold
        X_newtrain, X_val = X_crop[train_idx], X_crop[val_idx]
        y_newtrain, y_val = y_train[train_idx], y_train[val_idx]

        # Apply CSP for the fold
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        X_train_features = csp.fit_transform(X_newtrain, y_newtrain)
        X_val_features = csp.transform(X_val)

        # Train LDA for the fold
        lda = LDA()
        lda.fit(X_train_features, y_newtrain)

        # Evaluate on the test fold
        val_accuracy = lda.score(X_val_features, y_val)
        train_scores.append(val_accuracy)

    train_scores = np.round(train_scores, 2)
    # print(f"Train scores (CV): {train_scores}")

    # Step 3: Refit CSP and LDA on the entire training set
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_train_features = csp.fit_transform(X_crop, y_train)

    lda = LDA()
    lda.fit(X_train_features, y_train)

    # Step 4: Test evaluation (if test data is provided)
    test_results = {}
    if X_test is not None and y_test is not None:
        X_band = X_test[:, :, start_idx:end_idx]
        X_test_features = csp.transform(X_band)

        # Evaluate LDA on the test set
        predictions = lda.predict(X_test_features)
        test_accuracy = lda.score(X_test_features, y_test)

        # Calculate confusion matrix and kappa score
        cm = confusion_matrix(y_test, predictions)
        kappa = cohen_kappa_score(y_test, predictions)

        # Store test results
        test_results = {
            'accuracy': test_accuracy,
            'confusion_matrix': cm,
            'kappa_score': kappa
        }

        # Display and save confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['left_hand', 'right_hand'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f"{subject}")
        if folder_path:
            plt.savefig(f'{folder_path}cm2_nofb_{min_freq}_{max_freq}.png')
        plt.show()

        # Print additional metrics
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, digits=2))

        print(f"Test accuracy: {test_accuracy:.2f}")
        print(f"Kappa score: {kappa:.2f}")

    return train_scores, test_results
