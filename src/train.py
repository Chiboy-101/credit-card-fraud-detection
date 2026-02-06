# Import necessary modules and libraries needed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

# Load the dataset
df = pd.read_csv("creditcard.csv")
df.head()  # View first 5 rows of the dataset
df.shape

# Check for missing values
df.isnull().sum()  # None found

# Perform exploratory Data Analysis

# Countplot of Class
sns.countplot(data=df, x="Class")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.xlabel("Transaction Class")
plt.ylabel("Count")
plt.title("Fraud vs Non-Fraud Count")
plt.show()  # High class Imbalance identified

# Distribution of Amount
sns.histplot(data=df, x="Amount", bins=50)
plt.title("Distribution of Transaction Amount")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()  # (Heavily right-skewed), very few large amounts exists in the dataset

# Amount Distribution by class
plt.figure(figsize=(6, 4))
sns.boxplot(x="Class", y="Amount", data=df)
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.title("Amount Distribution by Class")
plt.show()  # Fraud usually occurs with small amounts

# Labels and features
X = df.drop("Class", axis=1)
y = df["Class"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28, stratify=y
)

# Scale non-PCA features
scaler = StandardScaler()
X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

# Load models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced"
    ),  # baseline model
    "Random Forest Classifier": RandomForestClassifier(class_weight="balanced"),
}  # To uncover non-linear patters

# Loop through each model and evaluate

results = []  # list to store results, later to be converted to a dataframe
for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)

    # Predict labels and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # For ROC-AUC

    # Evaluate
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Append results
    results.append(
        {"Model": name, "Precision": precision, "Recall": recall, "ROC-AUC": roc_auc}
    )

# Create DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Perform Hyper-Parameter Tuning
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
}

rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    scoring="recall",  # prioritize catching fraud
    cv=3,
    n_iter=10,
    n_jobs=-1,
    verbose=1,
)

# Perform Threshold Tuning
search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("Best RF Parameters:", search.best_params_)

y_prob = best_rf.predict_proba(X_test)[:, 1]  # type: ignore

thresholds = np.arange(0.1, 0.9, 0.1)
results = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    results.append(
        {
            "Threshold": t,
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
        }
    )

threshold_df = pd.DataFrame(results)
print(threshold_df)

# Plot Precision, Recall, F1 vs Threshold
plt.figure(figsize=(8, 5))
plt.plot(
    threshold_df["Threshold"], threshold_df["Precision"], marker="o", label="Precision"
)
plt.plot(threshold_df["Threshold"], threshold_df["Recall"], marker="o", label="Recall")
plt.plot(threshold_df["Threshold"], threshold_df["F1"], marker="o", label="F1-score")

plt.title("Threshold vs Precision/Recall/F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.show()

# Final Evaluation

# Example: choose threshold 0.4
final_threshold = 0.4
y_pred_final = (y_prob >= final_threshold).astype(int)

# Metrics
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)
roc_auc = roc_auc_score(y_test, y_prob)

print(
    f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}"
)

# Save the best model

with open("best_rf_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

# Load the model later
# with open("best_rf_model.pkl", "rb") as f:
# loaded_model = pickle.load(f)

# Test loaded model
# y_pred = loaded_model.predict(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.joblib")

# Later, load the scaler to transform new data
# scaler = joblib.load("scaler.joblib")
# X_new[['Amount', 'Time']] = scaler.transform(X_new[['Amount', 'Time']])
