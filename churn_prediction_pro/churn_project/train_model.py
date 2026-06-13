"""
Bank Customer Churn Prediction - Model Training
Run this script first to train and save the ML model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   BANK CUSTOMER CHURN PREDICTION - MODEL TRAINING")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. LOAD OR GENERATE DATA
# ─────────────────────────────────────────────────────────────
data_path = "data/churn_data.csv"

if os.path.exists(data_path):
    print(f"\n[1] Loading data from {data_path}")
    df = pd.read_csv(data_path)
else:
    print("\n[1] Generating synthetic dataset (10,000 customers)...")
    np.random.seed(42)
    n = 10000

    geography = np.random.choice(['France', 'Spain', 'Germany'], n, p=[0.50, 0.25, 0.25])
    gender = np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45])
    age = np.random.randint(18, 75, n)
    tenure = np.random.randint(0, 11, n)
    credit_score = np.random.randint(300, 851, n)
    balance = np.where(np.random.rand(n) > 0.3,
                       np.random.uniform(1000, 250000, n), 0)
    num_products = np.random.choice([1, 2, 3, 4], n, p=[0.50, 0.46, 0.02, 0.02])
    has_cr_card = np.random.choice([0, 1], n, p=[0.30, 0.70])
    is_active = np.random.choice([0, 1], n, p=[0.49, 0.51])
    estimated_salary = np.random.uniform(11, 200000, n)

    # Churn probability based on realistic factors
    churn_prob = (
        0.05
        + 0.20 * (geography == 'Germany')
        + 0.10 * (age > 45)
        + 0.15 * (num_products == 1)
        + 0.12 * (is_active == 0)
        + 0.08 * (balance > 100000) * (num_products == 1)
        + 0.06 * (credit_score < 500)
        - 0.05 * (tenure > 5)
        - 0.04 * (has_cr_card == 1)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.90)
    exited = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'CustomerId': np.random.randint(10000000, 19999999, n),
        'Surname': ['Customer_' + str(i) for i in range(n)],
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance.round(2),
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary.round(2),
        'Exited': exited
    })
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"   Saved to {data_path}")

print(f"   Dataset shape: {df.shape}")
print(f"   Churn rate: {df['Exited'].mean():.1%}")

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n[2] Preprocessing data...")

df = df.drop(columns=['CustomerId', 'Surname'], errors='ignore')
df = df.dropna()

# Feature Engineering
df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['AgeTenureInteraction'] = df['Age'] * df['Tenure']
df['ProductDensity'] = df['NumOfProducts'] / (df['Tenure'] + 1)
df['EngagementScore'] = df['IsActiveMember'] * df['NumOfProducts']
df['HasBalance'] = (df['Balance'] > 0).astype(int)

# Encoding
df_encoded = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False)

feature_cols = [c for c in df_encoded.columns if c != 'Exited']
X = df_encoded[feature_cols]
y = df_encoded['Exited']

print(f"   Features: {len(feature_cols)}")
print(f"   Feature list: {list(feature_cols)}")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[3] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────────────────────
print("\n[4] Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                       learning_rate=0.05, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"   Training {name}...", end=' ')
    X_tr = X_train_scaled if name == 'Logistic Regression' else X_train
    X_te = X_test_scaled  if name == 'Logistic Regression' else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    results[name] = {
        'model': model,
        'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred) * 100, 2),
        'recall':    round(recall_score(y_test, y_pred) * 100, 2),
        'f1':        round(f1_score(y_test, y_pred) * 100, 2),
        'roc_auc':   round(roc_auc_score(y_test, y_prob) * 100, 2),
    }
    print(f"Accuracy={results[name]['accuracy']}%  AUC={results[name]['roc_auc']}%")

# ─────────────────────────────────────────────────────────────
# 5. SELECT BEST MODEL
# ─────────────────────────────────────────────────────────────
print("\n[5] Selecting best model by ROC-AUC...")
best_name = max(results, key=lambda k: results[k]['roc_auc'])
best_result = results[best_name]
best_model = best_result['model']
print(f"   Best model: {best_name}")

print("\n   ── Model Comparison ──────────────────────────────────")
print(f"   {'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("   " + "─" * 55)
for name, r in results.items():
    marker = " ★" if name == best_name else ""
    print(f"   {name:<25} {r['accuracy']:>5}% {r['precision']:>5}% "
          f"{r['recall']:>5}% {r['f1']:>5}% {r['roc_auc']:>5}%{marker}")

print(f"\n   Classification Report ({best_name}):")
X_te_best = X_test_scaled if best_name == 'Logistic Regression' else X_test
y_pred_best = best_model.predict(X_te_best)
print(classification_report(y_test, y_pred_best, target_names=['Retained', 'Churned']))

# ─────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_cols)
    fi = fi.sort_values(ascending=False)
    print(f"\n   Top-10 Feature Importances ({best_name}):")
    for feat, val in fi.head(10).items():
        bar = '█' * int(val * 40)
        print(f"   {feat:<35} {val:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n[6] Saving model artifacts...")
os.makedirs('models', exist_ok=True)

artifacts = {
    'model':        best_model,
    'scaler':       scaler,
    'feature_cols': list(feature_cols),
    'model_name':   best_name,
    'metrics':      {k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                     for k, v in results.items()},
    'all_results':  results,
}

with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("   Saved: models/churn_model.pkl")
print("\n✅ Training complete! Run: streamlit run app.py")
print("=" * 60)
