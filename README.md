# 💳 Credit Card Fraud Detection

Detect fraudulent transactions in credit card data using **Machine Learning**.  
This project combines **Random Forests, SMOTE, and threshold optimization** to build a highly accurate fraud detection system.  

---

## 🚀 Project Overview

Credit card fraud is costly and difficult to detect due to **highly imbalanced datasets** (fraud is very rare).  
This project aims to:

- Explore the dataset visually and statistically.
- Train and tune **Random Forest** and **Logistic Regression** models.
- Optimize probability thresholds to balance **Precision vs Recall**.
- Save models and scalers for production deployment.

---

## 📊 Dataset

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) 
> ⚠️ Dataset is too large for GitHub. Download it from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/` folder.
- Transactions: 284,807  
- Fraud cases: 492 (~0.17%)  
- Features:  
  - `V1` to `V28` (anonymized PCA components)  
  - `Time` – seconds elapsed between transactions  
  - `Amount` – transaction amount  
  - `Class` – 0 = Normal, 1 = Fraud  

---

## 🔍 Exploratory Data Analysis

- **Class imbalance:** Heavily skewed toward non-fraud transactions.  
- **Transaction amounts:** Most frauds occur with small amounts.  
- **Visualizations:**  
  - Count of fraud vs non-fraud  
  - Distribution of transaction amounts  
  - Amount distribution by class  

---

## 🛠 Features & Preprocessing

- **Features:** All except `Class`.  
- **Target:** `Class` column.  
- **Scaling:** StandardScaler for `Amount` and `Time`. 
---

## 🧰 Models

- **Logistic Regression** – baseline model.  
- **Random Forest Classifier** – captures non-linear patterns.  
- **Class imbalance handling:** `class_weight='balanced'`  
- **Threshold tuning:** Optimal threshold selected to maximize **F1 score**.  

---

## 📈 Evaluation Metrics (Threshold = 0.4)

| Metric     | Score  |
|------------|--------|
| Precision  | 0.833  |
| Recall     | 0.816  |
| F1-score   | 0.825  |
| ROC-AUC    | 0.976  |

> ✅ These results indicate a **highly accurate model** that balances catching fraud and minimizing false positives.

---

## 💾 Saved Artifacts

- **Model:** `models/best_rf_model.pkl`  
- **Scaler:** `models/scaler.joblib`  

These can be loaded for **predicting new transactions** without retraining.

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Install dependencies:

```bash
pip install -r requirements.txt

3. Run training script

```bash
python src/train_model.py
