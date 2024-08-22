import shap
from sklearn.ensemble import RandomForestClassifier

# Fit a RandomForest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Explain model predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

st.subheader("Model Interpretability with SHAP")
if st.button('Visualize SHAP Values'):
    st.write(f"SHAP Values for Sample {sample_index}")
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1][sample_index], X[sample_index], matplotlib=True)
