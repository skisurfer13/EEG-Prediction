# AIM-S24-EEG-Project

This repo was made for the Spring AIM 2024 (AI Mentorship) Program. The project was a mix of theory and practical application. It classifies EEG Data to determine whether the given electrical signal is in the seizure state (ictal), preseizure state (preictal), or between seizures state (interictal). It is made up of 4 parts:

  1. Individual model files - CNN, KNN, RF, SVM, and ADABoost. Each has a .py extension following it. Each mentee worked on one.
  2. Fusion model - Aggregates the five models using the probability averaging ensembling technique to come up with a final prediction. Performed in display.py
  3. Pickle Files - Each model has a corresponding pickle file (model_pred.pkl) which contains the models predictions for the test data. These were made to increase speed for presentation purposes only (instead of needing to wait for each model to finish running). To generate new pickle files (i.e. store new predictions), simply run the desired model.
  4. Visualization - Also performed in display.py. Works in conjunction with streamlit for cloud deployment.


Additional files:
  requirements.txt - config file for streamlit
  cnn_model.pth - architecture code for cnn
  y_test.pkl - pickle file containing answer key for fusion code performance metrics
  pklreader.py - to help load in and view pickle files for user convenience 
  
Click here to see it in action: https://aims24-irsjnqnuzhdveg84yns6eq.streamlit.app/
