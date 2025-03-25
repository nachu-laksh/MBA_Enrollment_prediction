# MBA_Enrollment_prediction

---

## 📌 Objective

- Predict MBA pursuit using logistic regression and random forest
- Explore feature importance and interaction effects
- Evaluate model fairness (gender bias)
- Interpret business-related insights behind enrollment behavior

---

## 🛠️ Methods

- **Preprocessing**: One-hot encoding, scaling, and cleaning
- **Models**: Logistic Regression (with L1/L2 regularization, class weights), Random Forest
- **Feature Engineering**: Created interaction terms (e.g., low salary + scholarship)
- **Evaluation**: Accuracy, precision, recall, F1-score, and fairness checks

---

## 🔍 Key Findings

- **Positive Predictors**: Networking interest, skill enhancement, entrepreneurship
- **Negative Predictors**: Self-funding, certain post-MBA career choices
- **Model Accuracy**: ~48% using balanced logistic regression
- **Fairness**: No significant gender bias detected
- **Limitation**: Feature interactions had limited effect; accuracy suggests missing variables

---

## 📊 Technologies Used

- pandas, scikit-learn  
- Logistic Regression, Random Forest  
- Feature Importance & Classification Reports

---

## 📈 Future Improvements

- Collect more detailed behavioral and psychometric data
- Test non-linear models (e.g., XGBoost, Neural Networks)
- Improve feature interaction design

---



