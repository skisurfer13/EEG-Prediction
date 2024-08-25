
---

# EEG-Prediction: Epi-Sense Visualization with Advanced Statistical Analysis
## 🧠 Overview

The **EEG-Prediction** project, also known as **Epi-Sense**, focuses on classifying EEG signals into seizure states: **preictal**, **interictal**, and **ictal**. This project combines machine learning, statistical analysis, and signal processing techniques to provide a robust, interactive, and educational tool for EEG analysis.

The project is designed to:
- Visualize raw EEG data.
- Apply various machine learning models.
- Perform in-depth statistical analysis.
- Provide an intuitive interface for exploring and understanding the EEG data.

## 🎯 Key Features

1. **EEG Fusion Prediction**  
   Utilize a fusion of machine learning models (CNN, KNN, RF, SVM, AdaBoost) to classify EEG signals. The fusion approach improves accuracy by averaging predictions from multiple models, ensuring robustness against individual model weaknesses.

2. **Exploratory Data Analysis (EDA)**  
   Perform in-depth analysis of the EEG data, including stationarity tests, frequency domain analysis, and extraction of statistical features. The EDA section provides crucial insights into the data's characteristics, guiding model selection and analysis.

3. **Signal Processing and Noise Reduction**  
   Apply band-pass filtering to clean EEG signals, focusing on relevant frequency bands (0.5–50 Hz). Visualize the filtered data and observe the impact of noise reduction on signal clarity.

4. **Bayesian Inference and Probabilistic Models**  
   Implement Bayesian Ridge Regression to make probabilistic predictions. Visualize the posterior distributions and understand the uncertainty in predictions, crucial for medical decision-making.

5. **Survival Analysis**  
   Model seizure occurrence patterns using Kaplan-Meier analysis. Understand the probability of remaining seizure-free over time, providing valuable insights for epilepsy management.

6. **Advanced Data Analysis and Model Insights** (🚧 Under Testing)  
   Explore advanced visualizations such as correlation heatmaps, time-series analysis, and hyperparameter tuning heatmaps. Note that this section is under testing due to potential app crashes.

7. **Statistical Hypothesis Testing**  
   Conduct an ANOVA test to determine whether the means of the three EEG classes (preictal, interictal, ictal) are significantly different. Visualize the test results with boxplots and interpret the statistical significance.

## 🖥️ App Demo

[Launch the Streamlit App](https://eeg-prediction-nnzmjq3bpqkxwbr8zt7unt.streamlit.app/) to explore the interactive visualizations and analyses.

## ⚠️ Known Issues

- The **Advanced Data Analysis and Model Insights** section is under testing and may cause app crashes due to TLEs.

## 📊 Future Improvements

- Real-time EEG data streaming support.
- Enhanced model interpretability with additional visual explanations. Addition of SHAP, LIME and other methods for interpretability analysis. 
- Integration of additional classification models and advanced ensembling techniques.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, fork the repository, and create pull requests.

---

