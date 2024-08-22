from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import welch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Function to perform stationarity tests
def stationarity_tests(data):
    # Augmented Dickey-Fuller Test
    adf_result = adfuller(data)
    st.write("ADF Statistic:", adf_result[0])
    st.write("p-value:", adf_result[1])
    st.write("Critical Values:")
    for key, value in adf_result[4].items():
        st.write(f'{key}: {value}')

    # KPSS Test
    kpss_result = kpss(data, regression='c')
    st.write("\nKPSS Statistic:", kpss_result[0])
    st.write("p-value:", kpss_result[1])
    st.write("Critical Values:")
    for key, value in kpss_result[3].items():
        st.write(f'{key}: {value}')

# Function for frequency domain analysis
def frequency_domain_analysis(data, fs=256):
    freqs, psd = welch(data, fs=fs)
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psd, color='blue')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    st.pyplot(plt)

# Function to extract statistical features
def extract_statistical_features(data):
    mean_val = np.mean(data)
    variance_val = np.var(data)
    skewness_val = scipy.stats.skew(data)
    kurtosis_val = scipy.stats.kurtosis(data)
    st.write(f"Mean: {mean_val}")
    st.write(f"Variance: {variance_val}")
    st.write(f"Skewness: {skewness_val}")
    st.write(f"Kurtosis: {kurtosis_val}")

# EDA Section in Streamlit
st.header("Exploratory Data Analysis (EDA)")
sample_index = st.slider('Select Test Sample Index for EDA', 0, 29, 0)
if st.button('Perform EDA'):
    st.subheader("Time-Series Analysis and Stationarity Tests")
    stationarity_tests(X[sample_index])

    st.subheader("Frequency Domain Analysis")
    frequency_domain_analysis(X[sample_index])

    st.subheader("Statistical Feature Extraction")
    extract_statistical_features(X[sample_index])
