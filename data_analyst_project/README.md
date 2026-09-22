# 🏥 Healthcare No-Show Prediction

A full end-to-end machine-learning project that predicts whether a patient will **miss their medical appointment**, using the `healthcare_noshows.csv` dataset.

---

## 📁 Project Structure

```
.
├── healthcare_noshows.csv   # Raw dataset
├── ml_backend.py            # Data pipeline, feature engineering, model training
├── app.py                   # Streamlit frontend (3-page dashboard)
├── requirements.txt         # Python dependencies
├── model_artifacts.joblib   # Saved model + scaler (auto-generated)
└── plots/                   # Auto-generated EDA plots (PNG)
```

---

## 🚀 Quick Start

### 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train the Model (Backend)

```bash
python ml_backend.py
```

This will:
- Load & clean `healthcare_noshows.csv`
- Engineer features (wait days, weekday, age groups, etc.)
- Train 3 models: **Logistic Regression**, **Random Forest**, **Gradient Boosting**
- Print cross-validation & test metrics
- Save the best model to `model_artifacts.joblib`
- Save EDA plots to the `plots/` folder

### 3 — Launch the Dashboard (Frontend)

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **📊 Overview & EDA** | Dataset statistics, class distribution, age/gender/SMS charts, correlation heatmap |
| **🤖 Model Performance** | Confusion matrix, ROC curve, feature importance, probability distribution |
| **🔮 Predict No-Show** | Interactive form → real-time prediction with gauge chart & risk factors |

---

## 🤖 Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Fast baseline, class-weight balanced |
| Random Forest | 200 trees, max depth 8, balanced |
| **Gradient Boosting** | 200 estimators, lr=0.05 — typically best ROC-AUC |

---

## 🔬 Features Used

| Feature | Description |
|---------|-------------|
| `age` | Patient age |
| `gender_enc` | 1 = Female, 0 = Male |
| `scholarship` | Bolsa Família welfare recipient |
| `hipertension` | Hypertension diagnosis |
| `diabetes` | Diabetes diagnosis |
| `alcoholism` | Alcoholism flag |
| `handcap` | Handicap flag |
| `sms_received` | SMS reminder was sent |
| `wait_days` | Days between scheduling and appointment |
| `appt_weekday` | Day of week (0=Mon … 5=Sat) |
| `appt_month` | Month of appointment |
| `comorbidity_count` | Sum of chronic conditions |
| `age_group_enc` | 0=child 1=teen 2=young_adult 3=adult 4=senior |
| `neighbourhood_freq` | Frequency-encoded neighbourhood |

---

## 📈 Typical Results

| Model | CV ROC-AUC | Test ROC-AUC |
|-------|------------|--------------|
| Logistic Regression | ~0.66 | ~0.66 |
| Random Forest | ~0.72 | ~0.72 |
| Gradient Boosting | ~0.74 | ~0.74 |

> No-show base rate ≈ 20% — the models significantly outperform a random baseline.
