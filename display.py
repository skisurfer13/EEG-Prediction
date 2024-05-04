import streamlit as st
import pandas as pd
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
import gdown
import zipfile

st.title("Epi-Sense Visualization")

# Function to download and extract the zip file from Google Drive using gdown
def download_and_extract_zip(file_id, zip_filename, extract_to):
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(download_url, zip_filename, quiet=False)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_filename)

# Load the EEG data from the extracted folder
def load_eeg_data(base_dir):
    categories = ['preictal', 'interictal', 'ictal']
    labels = {'preictal': 0, 'interictal': 1, 'ictal': 2}
    X = []  # Raw Data Matrix
    y = []  # Label vector
    print(f"Checking base directory: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist!")
        return np.array([]), np.array([])

    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        print(f"Checking category directory: {cat_dir}")
        if os.path.exists(cat_dir):
            for file in os.listdir(cat_dir):
                print(f"Processing file: {file}")
                file_path = os.path.join(cat_dir, file)
                if file.endswith('.mat'):
                    mat_data = scipy.io.loadmat(file_path)
                    data = mat_data.get(category)
                    if data is not None:
                        X.append(data.flatten())  # Flatten the EEG segment
                        y.append(labels[category])
        else:
            print(f"Directory {cat_dir} does not exist!")
    return np.array(X), np.array(y)

# Download and extract dataset
file_id = "1Y0Cw2emtNxQX0Ei47rbR33Da9Yeqt39L"  # File ID extracted from the Google Drive link
zip_filename = "EEG_Epilepsy_Datasets.zip"
extracted_folder = "EEG_Epilepsy_Datasets"
if not os.path.exists(extracted_folder):
    download_and_extract_zip(file_id, zip_filename, extracted_folder)

# Load the EEG data
X, y = load_eeg_data(extracted_folder)

if X.size == 0 or y.size == 0:
    st.error("No EEG data found. Please check if the files were downloaded correctly.")
else:
    st.write(f"Loaded {len(X)} samples of EEG data.")
