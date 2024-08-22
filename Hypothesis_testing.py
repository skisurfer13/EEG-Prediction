from scipy.stats import f_oneway, ttest_ind

def perform_anova_test(X, y):
    preictal = X[y == 0]
    interictal = X[y == 1]
    ictal = X[y == 2]
    
    # ANOVA Test
    f_stat, p_value = f_oneway(preictal, interictal, ictal)
    st.write(f"ANOVA F-Statistic: {f_stat}")
    st.write(f"p-value: {p_value}")

st.subheader("Statistical Hypothesis Testing")
if st.button('Perform ANOVA'):
    perform_anova_test(X, y)
