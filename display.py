import streamlit as st
import pandas as pd
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
import requests
import zipfile
import logging

# Initialize the logger
logging.basicConfig(level=logging.INFO)

st.title("Epi-Sense Visualization")

# Function to download and extract the zip file from Google Drive
def download_and_extract_zip(url, zip_filename, extract_to):
    logging.info("Downloading the dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logging.info("Extracting the dataset...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_filename)

# Load the EEG data from the extracted folder
def load_eeg_data(base_dir):
    categories = ['preictal', 'interictal', 'ictal']
    labels = {'preictal': 0, 'interictal': 1, 'ictal': 2}
    X = []  # Raw Data Matrix
    y = []  # Label vector
    logging.info(f"Checking base directory: {base_dir}")
    # Check the actual subdirectory that contains the data
    for category in categories:
        cat_dir = None
        # Search subdirectories for category
        for subdir, _, _ in os.walk(base_dir):
            if os.path.basename(subdir) == category:
                cat_dir = subdir
                break
        logging.info(f"Checking category directory: {cat_dir}")
        if cat_dir and os.path.exists(cat_dir):
            logging.info(f"Directory exists: {cat_dir}")
            for file in os.listdir(cat_dir):
                logging.info(f"Processing file: {file}")
                file_path = os.path.join(cat_dir, file)
                if file.endswith('.mat'):
                    mat_data = scipy.io.loadmat(file_path)
                    data = mat_data.get(category)
                    if data is None:
                        logging.error(f"Data not found in {file_path}")
                    else:
                        X.append(data.flatten())  # Flatten the EEG segment
                        y.append(labels[category])
        else:
            logging.error(f"Directory {cat_dir} does not exist!")
    return np.array(X), np.array(y)

# Download and extract dataset
zip_url = "https://drive.google.com/uc?export=download&id=1Y0Cw2emtNxQX0Ei47rbR33Da9Yeqt39L"
zip_filename = "dataset.zip"
extracted_folder = "EEG_Epilepsy_Datasets"
if not os.path.exists(extracted_folder):
    download_and_extract_zip(zip_url, zip_filename, extracted_folder)

# Load the EEG data
X, y = load_eeg_data(extracted_folder)

if X.size == 0 or y.size == 0:
    st.error("No EEG data found. Please check if the files were downloaded correctly.")
else:
    st.write(f"Loaded {len(X)} samples of EEG data.")
