from sklearn.linear_model import BayesianRidge

# Fit Bayesian Ridge Regression on the EEG data
def apply_bayesian_inference(X, y):
    model = BayesianRidge()
    model.fit(X, y)
    st.write("Model Coefficients:", model.coef_)
    st.write("Model Intercept:", model.intercept_)
    return model

st.subheader("Bayesian Inference and Probabilistic Models")
if st.button('Apply Bayesian Inference'):
    bayesian_model = apply_bayesian_inference(X, y)
    prediction = bayesian_model.predict([X[sample_index]])
    st.write(f"Bayesian Prediction for Sample {sample_index}: {prediction[0]}")
