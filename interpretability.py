import shap

# Load model (assuming you have a CNN model)
model = load_model('cnn_model.pth')
explainer = shap.DeepExplainer(model, X_train[:100])  # Use training data for explanation
shap_values = explainer.shap_values(X[sample_index:sample_index+1])

st.subheader("Model Interpretability with SHAP")
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X[sample_index], matplotlib=True)
