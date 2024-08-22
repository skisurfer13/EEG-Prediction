from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import permutation_test_score

# Function to perform cross-validation
def perform_cross_validation(X, y):
    cv = StratifiedKFold(n_splits=5)
    model = RandomForestClassifier()
    cv_scores = cross_val_score(model, X, y, cv=cv)
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Mean CV Score: {np.mean(cv_scores)}")

# Function for permutation testing
def perform_permutation_test(X, y):
    model = RandomForestClassifier()
    score, permutation_scores, pvalue = permutation_test_score(model, X, y, cv=5, n_permutations=100)
    st.write(f"Permutation Test Score: {score}")
    st.write(f"Permutation Test p-value: {pvalue}")

st.subheader("Model Evaluation and Statistical Validation")
if st.button('Perform Cross-Validation'):
    perform_cross_validation(X, y)

if st.button('Perform Permutation Test'):
    perform_permutation_test(X, y)
