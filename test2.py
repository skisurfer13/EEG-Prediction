import streamlit as st
import scipy.io
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

#THESE FIRST THREE ARE THE SAME FOR ALL MODELS
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












#1D-CNN

import scipy.io
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

# Convert data to PyTorch tensors with an added channel dimension
# Convert data to PyTorch tensors and ensure they are correctly shaped with channel as the second dimension
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Shape [num_samples, 1, 1024]
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)    # Shape [num_samples, 1, 1024]
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the 1D CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 256, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = x.view(-1, 128 * 256) # Updated
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Create the model, optimizer, and loss function
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for inputs, labels in train_loader:
    print(inputs.shape)  # Should be [32, 1, 1024]
    print(labels.shape)  # Should be [32]
    break  # Exit after the first batch


# Training loop
# Training loop
def train_save_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # Ensure inputs are correctly shaped
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}: Training Complete")
    return model, train_losses  # Ensure this return statement is included

def train_and_save():
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    trained_model, train_losses = train_save_model(model, train_loader, criterion, optimizer, num_epochs=20)
    torch.save(trained_model.state_dict(), 'cnn_model.pth')

# Uncomment below to train and save the model, then comment it out after use.
#train_and_save()


    
# Example of training and saving the model
cnn_model = CNN()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_save_model(cnn_model, train_loader, criterion, optimizer)


# Before calling the train_save_model function, ensure the DataLoader is correctly set up:
print(f'Training data loader: {len(train_loader.dataset)} samples')
print(f'Testing data loader: {len(test_loader.dataset)} samples')

# Then call the train_save_model function
model, train_losses = train_save_model(model, train_loader, criterion, optimizer)


# Train the model
model, train_losses = train_save_model(model, train_loader, criterion, optimizer)


# Evaluation loop
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)  # Directly use inputs without permute
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    test_loss /= len(test_loader.dataset)
    accuracy = correct / total
    return test_loss, accuracy, y_true, y_pred

# Evaluate the model
test_loss, test_accuracy, y_true, cnn_pred = evaluate_model(model, test_loader)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, cnn_pred))












# KNN

# Define extract_knn_features function
from scipy.signal import welch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def extract_knn_features(segment, fs=200):

    # Calculate the desired statistical features
    min_val = np.min(segment)
    max_val = np.max(segment)
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    variance_val = np.var(segment)
    skewness_val = scipy.stats.skew(segment)
    kurtosis_val = scipy.stats.kurtosis(segment)
# calculate Zero Crossing Rate
    centered_segment = segment - np.mean(segment)
    zcr_val = ((centered_segment[:-1] * centered_segment[1:]) < 0).sum()

 #calculate signal magnitued Area
    sma_val = np.sum(np.abs(segment)) / len(segment)
#calculate   Spectral Centroid
    freqs, psd = welch(segment, fs=fs)
    centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) != 0 else 0
# Hjorth Parameters
    first_derivative = np.diff(segment, n=1)
    activity = variance_val
    mobility = np.sqrt(np.var(first_derivative) / activity) if activity != 0 else 0
    complexity = np.sqrt(np.var(np.diff(first_derivative, n=1)) / np.var(first_derivative)) / mobility if mobility != 0 else 0
   # Return them as a flat list (no other dimensions)
    return [min_val, max_val, mean_val, std_val, variance_val, skewness_val, kurtosis_val, zcr_val,  sma_val, centroid, activity, mobility, complexity]

# Apply feature extraction to the dataset (X_train) and store them in a new matrix (X_train_features)
X_train_features = np.array([extract_knn_features(segment) for segment in X_train])
X_test_features = np.array([extract_knn_features(segment) for segment in X_test])

knn_model = KNeighborsClassifier(n_neighbors=12)

# Fit the model on the training data
knn_model.fit(X_train_features, y_train)

# Predict the labels for the test data
knn_pred = knn_model.predict(X_test_features)
print('knn accuracy: ', accuracy_score(y_test, knn_pred))

knn_probabilities = knn_model.predict_proba(X_test_features)
#knn_probabilities = knn_probabilities[:, :-1]
print(knn_probabilities.shape)
joblib.dump(knn_model, 'knn_model.pkl')

#df_knn_probabilities = pd.DataFrame(knn_probabilities)
#print(df_knn_probabilities)
#knn_probabilities


#REST OF MODELS BELOW












#Display

import streamlit as st
import torch
import numpy as np


# Function to load the CNN model (using st.cache to prevent reloads)
@st.cache_resource
def load_cnn_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to load the KNN model (using st.cache to prevent reloads)
@st.cache_resource
def load_knn_model(model_path):
    return joblib.load(model_path)

# Function to predict using the loaded model
def predict_with_models(sample, cnn_model, knn_model):
    cnn_input = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cnn_output = cnn_model(cnn_input)
    cnn_probabilities = torch.softmax(cnn_output, dim=1).detach().numpy()[0]
    cnn_prediction = np.argmax(cnn_probabilities)

    knn_input = extract_knn_features(sample)  # Your feature extraction for KNN
    knn_probabilities = knn_model.predict_proba([knn_input])[0]
    knn_prediction = np.argmax(knn_probabilities)

    return cnn_prediction, cnn_probabilities, knn_prediction, knn_probabilities

def display_predictions(sample_index, X_test, y_test, cnn_model, knn_model):
    sample = X_test[sample_index]
    true_label = y_test[sample_index]

    cnn_pred, cnn_probs, knn_pred, knn_probs = predict_with_models(sample, cnn_model, knn_model)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### CNN Prediction:")
        st.write(f"Predicted: {'Preictal' if cnn_pred == 0 else 'Interictal' if cnn_pred == 1 else 'Ictal'}")
        st.write("Probabilities:")
        st.write(f"Preictal: {cnn_probs[0]*100:.2f}%")
        st.write(f"Interictal: {cnn_probs[1]*100:.2f}%")
        st.write(f"Ictal: {cnn_probs[2]*100:.2f}%")

    with col2:
        st.write("### KNN Prediction:")
        st.write(f"Predicted: {'Preictal' if knn_pred == 0 else 'Interictal' if knn_pred == 1 else 'Ictal'}")
        st.write("Probabilities:")
        st.write(f"Preictal: {knn_probs[0]*100:.2f}%")
        st.write(f"Interictal: {knn_probs[1]*100:.2f}%")
        st.write(f"Ictal: {knn_probs[2]*100:.2f}%")


def main():
    st.title("EEG Prediction Model Comparisons")

    # Load models
    cnn_model = load_cnn_model('cnn_model.pth')
    knn_model = load_knn_model('knn_model.pkl')

    # Slider for selecting the sample
    sample_index = st.slider('Select Sample Index', 0, len(X_test) - 1, 0)

    # Display predictions
    display_predictions(sample_index, X_test, y_test, cnn_model, knn_model)

if __name__ == "__main__":
    main()