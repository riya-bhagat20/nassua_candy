# 🏦 Bank Customer Churn Prediction

ML-powered churn risk scoring system for retail banks.

## 📁 Project Structure
```
churn_project/
├── app.py              ← Streamlit web app
├── train_model.py      ← ML model training script
├── requirements.txt    ← Python dependencies
├── data/
│   └── churn_data.csv  ← Auto-generated if not present
└── models/
    └── churn_model.pkl ← Saved model (after training)
```

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train_model.py
```

### Step 3 — Launch the web app
```bash
streamlit run app.py
```

## 📊 Features
- **Risk Calculator** — Real-time churn probability for any customer
- **Model Performance** — Compare 4 ML models with full metrics
- **Feature Analysis** — Feature importance + engineered features
- **Scenario Simulator** — What-if analysis with sensitivity charts

## 🤖 Models Trained
| Model | Notes |
|---|---|
| Logistic Regression | Interpretability baseline |
| Decision Tree | Rule-based, explainable |
| Random Forest | Best overall (default) |
| Gradient Boosting | High accuracy alternative |

## 📌 Target Variable
- `Exited = 1` → Customer churned
- `Exited = 0` → Customer retained

## 🔧 Using Your Own Data
Place your CSV at `data/churn_data.csv` with these columns:
`CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure,
Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited`

Then run `python train_model.py` again.
