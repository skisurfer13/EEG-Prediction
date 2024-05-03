# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import streamlit as st
import os
import pandas as pd
import pywt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import random
import shap  # Ensure SHAP is installed

# Function to load EEG data
def load_eeg_data(base_dir):
    categories = ['preictal', 'interictal', 'ictal']
    labels = {'preictal': 0, 'interictal': 1, 'ictal': 2}
    X = []  # Raw Data Matrix 
    y = []  # Label vector
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        for file in os.listdir(cat_dir):
            file_path = os.path.join(cat_dir, file)
            if file.endswith('.mat'):
                mat_data = scipy.io.loadmat(file_path)
                data = mat_data[category]
                X.append(data.flatten())  # Flatten the EEG segment
                y.append(labels[category])
    return np.array(X), np.array(y)

# Load the EEG data
base_dir = '/Users/rahulc/Downloads/EEG_Epilepsy_Datasets'
X, y = load_eeg_data(base_dir)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=100)

# Feature extraction

def extract_wavelet_features(segment):
    coeffs = pywt.wavedec(segment, 'db4', level=7)
    features = []
    for coeff in coeffs:
        features.extend([np.min(coeff), np.max(coeff), np.mean(coeff), np.std(coeff)])
    return features

X_train_features = np.array([extract_wavelet_features(segment) for segment in X_train])
X_test_features = np.array([extract_wavelet_features(segment) for segment in X_test])

def extract_features(segment):
    # Calculate the desired statistical features
    min_val = np.min(segment)
    max_val = np.max(segment)
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    variance_val = np.var(segment)
    skewness_val = scipy.stats.skew(segment)
    kurtosis_val = scipy.stats.kurtosis(segment)
    
    wavelet = 'db2'
    level = 6
    wavelet_coeffs = pywt.wavedec(segment, wavelet, level=level)
    entropy = 0
    for coeff in wavelet_coeffs:
        coeff += 1e-10
        norm_coeff = coeff / np.sum(coeff)
        entropy = -np.sum(np.square(norm_coeff) * np.log(np.square(norm_coeff)))

    # Return them as a flat list (no other dimensions)
    return [min_val, max_val, mean_val, std_val, variance_val, skewness_val, kurtosis_val]

# Apply feature extraction to the dataset (X_train) and store them in a new matrix (X_train_features)
#X_train_features = np.array([extract_features(segment) for segment in X_train])
#X_test_features = np.array([extract_features(segment) for segment in X_test])


# Training the model
rf_pred_model = RandomForestClassifier(n_estimators=21, random_state=42)
rf_pred_model.fit(X_train_features, y_train)

# Function to display insights for a random sample
def display_random_sample():
    random_index = random.randint(0, len(X_test) - 1)
    sample = X_test[random_index]
    true_label = y_test[random_index]

    fig, ax = plt.subplots()
    ax.plot(sample)
    ax.set_title("EEG Segment Sample")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    #st.write("Extracting features...")
    # Use the new feature extraction method
    sample_features = np.array(extract_wavelet_features(sample)).reshape(1, -1)  # Make sure to use extract_features instead of extract_wavelet_features
    #st.write("Features extracted. Making prediction...")

    predicted_label = rf_pred_model.predict(sample_features)[0]
    #st.write("Prediction made. Displaying results...")

    st.title(f"Model's Prediction: {'Preictal' if predicted_label == 0 else 'Interictal' if predicted_label == 1 else 'Ictal'}")
    st.title(f"Actual Label: {'Preictal' if true_label == 0 else 'Interictal' if true_label == 1 else 'Ictal'}")

    probabilities = rf_pred_model.predict_proba(sample_features)[0]
    st.write("Prediction Probabilities:", {f'Class {i}': f'{prob*100:.2f}%' for i, prob in enumerate(probabilities)})

    importances = rf_pred_model.feature_importances_
    indices = np.argsort(importances)[-5:]
    st.write("Top 5 Important Features:")
    for i in indices:
        st.write(f"{feature_names[i]}: {importances[i]:.4f}")

    

# Use a button to trigger the display of a new sample
if st.button('Show Random Sample'):
    display_random_sample()