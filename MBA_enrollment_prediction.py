import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load Data
path = r"C:\Users\Nachu\OneDrive - University of Pittsburgh\ECON_2824\Homework\mba_decision_dataset.csv"
data = pd.read_csv(path)

# Standardize categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = (
    data[categorical_columns]
    .apply(lambda col: col.str.strip().str.lower() if col.dtype == "object" else col)
    .fillna("unknown")  # Mark missing values as unknown
)

# One-Hot Encoding for categorical features 
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Standardize column names
data.columns = (
    data.columns
    .str.strip('* ')
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.replace(' ', '_')
    .str.lower()
)

# Drop irrelevant features
data.drop(columns=["person_id", "gender_other"], inplace=True, errors="ignore")

# Define target and features
X = data.drop(columns=["decided_to_pursue_mba_yes"])
y = data["decided_to_pursue_mba_yes"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features 
scaler = StandardScaler()
numerical_cols = ["age", "gregmat_score", "annual_salary_before_mba", "expected_postmba_salary", "years_of_work_experience"]

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

#  Model 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

#  Model 2: Logistic Regression (Baseline)
logit_model = LogisticRegression(max_iter=2000, random_state=42, solver="liblinear")
logit_model.fit(X_train, y_train)
y_pred_logit = logit_model.predict(X_test)

# Compare Model Performance
print("\n Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("n Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logit))

print("\n Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("\n Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logit))

#  Though Logit performs slightly better than Random Forest (59% vs 56.85%), the recall for False (not pursuing MBA) is 0 in
# Logit. It predicts True correctly 100% of the time - i.e., it defaults to True for all cases, and ends up being right 59% 
# of the time (the actual % of MBA pursuers). We need to adjust the model.
# Even for Random Forest, recall for False is just 9%.

# Adjust Logistic Regression (Balanced Weights)
logit_model_balanced = LogisticRegression(max_iter=2000, random_state=42, solver="liblinear", class_weight="balanced")
logit_model_balanced.fit(X_train, y_train)
y_pred_balanced = logit_model_balanced.predict(X_test)

print("\n Logistic Regression Accuracy (Balanced Weights):", accuracy_score(y_test, y_pred_balanced))
print("\n Classification Report (Balanced Weights):\n", classification_report(y_test, y_pred_balanced))

#  Further adjustment - Train Logistic Regression with L1 and L2 regularization
logit_model_l1 = LogisticRegression(max_iter=2000, random_state=42, solver="liblinear", class_weight="balanced", penalty="l1")
logit_model_l1.fit(X_train, y_train)
y_pred_l1 = logit_model_l1.predict(X_test)

logit_model_l2 = LogisticRegression(max_iter=2000, random_state=42, solver="liblinear", class_weight="balanced", penalty="l2")
logit_model_l2.fit(X_train, y_train)
y_pred_l2 = logit_model_l2.predict(X_test)

print("\n Logistic Regression Accuracy (L1 Regularization):", accuracy_score(y_test, y_pred_l1))
print("\n Classification Report (L1 Regularization):\n", classification_report(y_test, y_pred_l1))

print("\n Logistic Regression Accuracy (L2 Regularization):", accuracy_score(y_test, y_pred_l2))
print("\n Classification Report (L2 Regularization):\n", classification_report(y_test, y_pred_l2))

#  There seems to be no improvement. Accuracy and recall are similar to 'balanced weights'.
# There are probably factors beyond what is available in this dataset that influence MBA decision.
# Another possibility - There is a better model for this dataset. 

#  Random Forest Feature Importance
rf_feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
rf_feature_importance = rf_feature_importance.sort_values(by='Importance', ascending=False)

print("\n Top 5 Most Important Features (Random Forest):")
print(rf_feature_importance.head(5))

print("\n Bottom 5 Least Important Features (Random Forest):")
print(rf_feature_importance.tail(5))

#  Check Most Important Features in Logistic Regression
logit_coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': logit_model_l1.coef_[0]})
logit_coefficients = logit_coefficients.sort_values(by='Coefficient', ascending=False)

print("\n Top 5 Features Encouraging MBA Decision:")
print(logit_coefficients.head(5))

print("\n Top 5 Features Discouraging MBA Decision:")
print(logit_coefficients.tail(5))

#  The overall influence of each of these features (individually) is pretty low, 
# so I must figure out the best feature interaction that influences this decision.
# It could be something like annual_salary_before_MBA <50000 + mba_funding_source_scholarship.
# This model as such is not very capable of predicting MBA decisions.
# Personally, I expected funding source-self-funded to be more influential in the decision to not pursue MBA (than just 4%!).
# It doesn't make sense why a scholarship would discourage MBA!
# If Entrepreneurs are more likely to pursue MBA - their desired postmba role would not be marketing director, both are unrelated. 
# The model is not able to capture this opposite relationship.

#  Adding Interaction Features (Based on Feature Importance)

X_train["low_salary_scholarship"] = (X_train["annual_salary_before_mba"] < 50000) & (X_train["mba_funding_source_scholarship"] == 1)
X_test["low_salary_scholarship"] = (X_test["annual_salary_before_mba"] < 50000) & (X_test["mba_funding_source_scholarship"] == 1)

X_train["entrepreneur_networking"] = (X_train["current_job_title_entrepreneur"] == 1) & (X_train["reason_for_mba_networking"] == 1)
X_test["entrepreneur_networking"] = (X_test["current_job_title_entrepreneur"] == 1) & (X_test["reason_for_mba_networking"] == 1)

X_train["consultant_skill_enhancement"] = (X_train["current_job_title_consultant"] == 1) & (X_train["reason_for_mba_skill_enhancement"] == 1)
X_test["consultant_skill_enhancement"] = (X_test["current_job_title_consultant"] == 1) & (X_test["reason_for_mba_skill_enhancement"] == 1)

# Train Logistic Regression with Improved Interactions
logit_model_interaction = LogisticRegression(max_iter=2000, random_state=42, solver="liblinear", class_weight="balanced")
logit_model_interaction.fit(X_train, y_train)

#  Predict and Evaluate
y_pred_interaction = logit_model_interaction.predict(X_test)

print("\n Logistic Regression Accuracy (With Interactions):", accuracy_score(y_test, y_pred_interaction))
print("\n Classification Report (With Interactions):\n", classification_report(y_test, y_pred_interaction))

#  Check Feature Importance in Logistic Regression
logit_coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': logit_model_interaction.coef_[0]})
logit_coefficients = logit_coefficients.sort_values(by='Coefficient', ascending=False)

print("\n Top 5 Features Encouraging MBA Decision:")
print(logit_coefficients.head(5))

print("\n Top 5 Features Discouraging MBA Decision:")
print(logit_coefficients.tail(5))

#Feature interaction did not do much to improve the model 
# - except the interaction of low salary pre MBA and scholarship, no interaction was significant.


# Checking for Gender Bias in Predictions
if "gender_male" in X_train.columns:
    gender_effect = data.groupby("gender_male")["decided_to_pursue_mba_yes"].mean()
    print("\n MBA Pursuit Rate by Gender:")
    print(gender_effect)

# There is no significant effect of gender in decision to pursue MBA.
# If there was indeed bias:
#  - Can check if dataset actually reflects the distorted rates based on gender.
#  - Check model accuracy without including gender as a feature.
#  - Add weights to the training data feature if gender is inaccurately represented 