# 📍 Day 5: K-Nearest Neighbors (KNN) Classifier – Heart Disease Prediction


## 💡 Project Summary

In this project, I implemented the **K-Nearest Neighbors (KNN)** algorithm to classify whether a patient is likely to have heart disease, based on various medical attributes. KNN is a non-parametric, instance-based algorithm that predicts the class of a sample by looking at the 'k' most similar data points.

This is a continuation of the same dataset used in **Day 4 (Random Forest)** to compare the performances of different classifiers on the same problem.

---

## 📂 Dataset Overview

- **Dataset**: `heart.csv`
- **Samples**: 1025
- **Features**: 13 (medical parameters like age, cholesterol, chest pain type, etc.)
- **Target Variable**: `target`  
  - `1` → Heart disease  
  - `0` → No heart disease

All features are clean and there are no missing values. Categorical and continuous features were identified for better analysis.

---

## 📊 Exploratory Data Analysis (EDA)

- Target distribution visualized using bar and pie charts
- Correlation heatmap to find top features influencing heart disease
- Feature type split into:
  - Qualitative: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`, `target`
  - Quantitative: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`

---

## 🤖 Model: K-Nearest Neighbors Classifier

- **Library**: `sklearn.neighbors.KNeighborsClassifier`
- **Train-Test Split**: 80% training, 20% testing
- **K Value**: Used default `k=5`
- **Distance Metric**: Euclidean distance (default)

### 📈 Evaluation Metrics

- **Training Accuracy**: 91.7%
- **Testing Accuracy**: 74.6%
- **Precision**: 0.77 (for detecting heart disease)
- **Recall**: 0.74
- **F1-Score**: 0.75
- **AUC Score**: ~0.75

<details>
<summary>📉 Confusion Matrix & ROC Curve</summary>

- Confusion matrix plotted to understand misclassifications
- ROC Curve generated to evaluate classifier performance at various thresholds

</details>

---

## 🔍 Why KNN Did Not Perform Well Here

While KNN is simple and often effective for smaller datasets, it didn't perform as well as the Random Forest model (which achieved 100% accuracy on the same data). Here's why:

- **KNN is highly sensitive to feature scaling** and the distribution of data. Since it relies on distance calculations, any difference in units or scale between features can skew predictions.
- **High dimensionality** (13 features) impacts KNN's performance due to the “curse of dimensionality.” In high dimensions, distance metrics lose effectiveness.
- Unlike Random Forest, **KNN doesn't learn a decision boundary**. It simply memorizes the training data and makes predictions based on the closest points at inference time.
- **KNN is slower and less scalable**, especially with large datasets, because it computes distances to every training point at prediction time.
- It also **doesn’t provide feature importance**, which limits interpretability and insights.

---

## 🔁 Comparison with Day 4 (Random Forest)

Compared to Day 4’s Random Forest Classifier, KNN clearly underperforms in both accuracy and reliability. Random Forest handled the feature space better, managed outliers and noise more effectively, and gave perfect classification on the test set (although possibly overfitting).

KNN, on the other hand, gave us a **realistic performance**, and is often better suited for smaller feature sets and when the data is very well scaled and normalized.

---

## 📌 Conclusion

KNN is a solid beginner-friendly algorithm and useful for quick experiments, but for structured medical data like this, **tree-based models like Random Forest are much more powerful and insightful**.

---

## 🛠️ Tools Used

- Python
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn

---
