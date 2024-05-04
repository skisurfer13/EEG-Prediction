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

st.title("Epi-Sense Visualization")

# Function to download and extract the zip file from Google Drive
def download_and_extract_zip(url, zip_filename, extract_to):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_filename)

# Load the EEG data from the extracted folder
def load_eeg_data(base_dir):
    categories = ['preictal', 'interictal', 'ictal']
    labels = {'preictal': 0, 'interictal': 1, 'ictal': 2}
    X = []  # Raw Data Matrix
    y = []  # Label vector
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        print(f"Checking directory: {cat_dir}")
        if os.path.exists(cat_dir):
            print(f"Directory exists: {cat_dir}")
            for file in os.listdir(cat_dir):
                print(f"Processing file: {file}")
                file_path = os.path.join(cat_dir, file)
                if file.endswith('.mat'):
                    mat_data = scipy.io.loadmat(file_path)
                    data = mat_data.get(category)
                    if data is None:
                        print(f"Data not found in {file_path}")
                    else:
                        X.append(data.flatten())  # Flatten the EEG segment
                        y.append(labels[category])
        else:
            print(f"Directory {cat_dir} does not exist!")
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



# Load y_test
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

def load_results(file_path):
    return pd.read_pickle(file_path)

def calculate_and_display_fusion_prediction(index, results_dfs):
    class_labels = {0: 'Preictal', 1: 'Interictal', 2: 'Ictal'}
    cumulative_probabilities = [0] * len(class_labels)
    for df in results_dfs.values():
        probabilities = df.iloc[index]['Probabilities']
        cumulative_probabilities = [sum(x) for x in zip(cumulative_probabilities, probabilities)]
    averaged_probabilities = [prob / len(results_dfs) for prob in cumulative_probabilities]
    final_predicted_class_index = averaged_probabilities.index(max(averaged_probabilities))
    return final_predicted_class_index, averaged_probabilities

def display_fusion_prediction(index, results_dfs):
    predicted_class_index, averaged_probabilities = calculate_and_display_fusion_prediction(index, results_dfs)
    class_labels = {0: 'Preictal', 1: 'Interictal', 2: 'Ictal'}
    correct_label = class_labels[y_test[index]]
    predicted_class = class_labels[predicted_class_index]
    averaged_probabilities_percent = [prob * 100 for prob in averaged_probabilities]
    
    # Display the text information in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div style='font-size: 20px; font-family: serif; color: white;'>Correct Label: <strong>{correct_label}</strong></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='font-size: 20px; font-family: serif; color: white;'>Fusion Prediction: <strong>{predicted_class}</strong></div>", unsafe_allow_html=True)
    with col3:
        # Calculate and display the accuracy right after the prediction
        fusion_predictions = [calculate_and_display_fusion_prediction(i, results_dfs)[0] for i in range(30)]
        accuracy = accuracy_score(y_test[:30], fusion_predictions)
        st.markdown(f"<div style='font-size: 20px; font-family: serif; color: white;'>Fusion Accuracy: {accuracy * 100:.2f}%</div>", unsafe_allow_html=True)
    
    st.title("")  # Add a blank title for vertical space
    
    # Convert probabilities to a DataFrame for better visualization
    prob_df = pd.DataFrame({
        'Class': list(class_labels.values()),
        'Probability (%)': averaged_probabilities_percent
    })
    
    # Plotting a horizontal bar chart with Seaborn for a prettier visualization
    fig, ax = plt.subplots(facecolor='#0D1117')
    ax.set_facecolor('#0D1117')  # Set plot background to dark gray
    sns.barplot(x='Probability (%)', y='Class', data=prob_df, palette='coolwarm', ax=ax)
    ax.set_ylabel('')  # Remove the 'Class' label from the y-axis
    ax.set_title('Fusion Prediction Probabilities', color='white', fontname='serif')
    ax.set_xlabel('Probability (%)', color='white', fontname='serif')
    ax.set_yticklabels(prob_df['Class'], color='white', fontname='serif')
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    st.pyplot(fig)

# Load data from all models
models = ['rf', 'ada']
results_dfs = {model: load_results(f"{model}_pred.pkl") for model in models}

# Customize font properties
font_properties = {'fontname': 'serif', 'fontsize': 14, 'color': 'white'}

# Initialize session state
if 'show_fusion' not in st.session_state:
    st.session_state['show_fusion'] = False

# Limit the sample index to the first 30 samples
if X.size > 0 and y.size > 0:
    sample_index = st.slider('Select Test Sample Index', 0, len(X)-1, 0)
    
    if st.button('Show EEG Sample & Fusion Prediction'):
        st.session_state['show_fusion'] = True

    if st.session_state['show_fusion']:
        # Display the EEG segment
        fig, ax = plt.subplots(facecolor='#0D1117')
        ax.set_facecolor('#0D1117')  # Set plot background to dark gray
        ax.plot(X[sample_index], color='yellow')  # Change EEG graph color to blue
        ax.set_title(f"EEG Recording: {sample_index}", fontdict=font_properties)
        ax.set_xlabel("Datapoint (0-1024)", fontdict=font_properties)
        ax.set_ylabel("Voltage", fontdict=font_properties)
        ax.tick_params(colors='white')
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        st.pyplot(fig)
        
        st.title("")  # Add a blank title for vertical space

        # Display the fusion prediction
        display_fusion_prediction(sample_index, results_dfs)
else:
    st.write("Please provide the correct path to the EEG dataset.")
