"""
Healthcare No-Show Prediction — Backend
=========================================
Responsibilities:
  - Load & clean data
  - Feature engineering
  - Train / evaluate ML models (Logistic Regression, Random Forest, XGBoost-style Gradient Boosting)
  - Persist the best model + preprocessor as model_artifacts.joblib
  - Expose a `predict()` helper used by the Streamlit frontend
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH      = "healthcare_noshows.csv"
ARTIFACTS_PATH = "model_artifacts.joblib"
PLOTS_DIR      = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Load & Clean ──────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Parse dates
    df["scheduledday"]   = pd.to_datetime(df["scheduledday"], errors="coerce")
    df["appointmentday"] = pd.to_datetime(df["appointmentday"], errors="coerce")

    # Boolean columns → int
    bool_cols = ["scholarship", "hipertension", "diabetes", "alcoholism",
                 "handcap", "sms_received"]
    for col in bool_cols:
        if df[col].dtype == object:
            df[col] = df[col].str.upper().map({"TRUE": 1, "FALSE": 0})
        else:
            df[col] = df[col].astype(int)

    # Target: showed_up → 1 means showed up, 0 means no-show
    if df["showed_up"].dtype == object:
        df["showed_up"] = df["showed_up"].str.upper().map({"TRUE": 1, "FALSE": 0})
    else:
        df["showed_up"] = df["showed_up"].astype(int)

    # No-show label (what we predict): 1 = no-show, 0 = showed up
    df["no_show"] = (df["showed_up"] == 0).astype(int)

    # Drop clearly irrelevant id cols
    df.drop(columns=["patientid", "appointmentid"], inplace=True, errors="ignore")

    # Drop rows with invalid ages
    df = df[df["age"] >= 0]

    return df


# ── 2. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Days between scheduling and appointment (already in date.diff but we recompute for safety)
    df["wait_days"] = (df["appointmentday"] - df["scheduledday"]).dt.days.clip(lower=0)

    # Day of week of appointment (0=Mon … 6=Sun)
    df["appt_weekday"] = df["appointmentday"].dt.dayofweek

    # Month of appointment
    df["appt_month"] = df["appointmentday"].dt.month

    # Age buckets
    df["age_group"] = pd.cut(df["age"],
                             bins=[-1, 12, 17, 35, 60, 120],
                             labels=["child", "teen", "young_adult", "adult", "senior"])

    # Comorbidity count
    df["comorbidity_count"] = (df["hipertension"] + df["diabetes"] +
                               df["alcoholism"] + df["handcap"])

    # Drop raw date columns — no longer needed
    df.drop(columns=["scheduledday", "appointmentday", "date.diff"],
            inplace=True, errors="ignore")

    # Encode gender
    df["gender_enc"] = (df["gender"].str.upper() == "F").astype(int)
    df.drop(columns=["gender"], inplace=True, errors="ignore")

    # Encode age_group
    age_map = {"child": 0, "teen": 1, "young_adult": 2, "adult": 3, "senior": 4}
    df["age_group_enc"] = df["age_group"].map(age_map)
    df.drop(columns=["age_group"], inplace=True, errors="ignore")

    # Frequency-encode neighbourhood (top-50 kept, rest → "other")
    top_n = df["neighbourhood"].value_counts().nlargest(50).index
    df["neighbourhood"] = df["neighbourhood"].where(df["neighbourhood"].isin(top_n), "OTHER")
    neighbourhood_freq = df["neighbourhood"].value_counts(normalize=True)
    df["neighbourhood_freq"] = df["neighbourhood"].map(neighbourhood_freq)
    df.drop(columns=["neighbourhood"], inplace=True, errors="ignore")

    return df


# ── 3. Build feature matrix ──────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "scholarship", "hipertension", "diabetes", "alcoholism",
    "handcap", "sms_received", "wait_days", "appt_weekday", "appt_month",
    "comorbidity_count", "gender_enc", "age_group_enc", "neighbourhood_freq"
]
TARGET_COL = "no_show"


def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


# ── 4. Train models ───────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                      class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                           learning_rate=0.05, random_state=42),
    }
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    results = {}
    for name, clf in models.items():
        cv_scores = cross_val_score(clf, X_train_sc, y_train, cv=5,
                                    scoring="roc_auc", n_jobs=-1)
        clf.fit(X_train_sc, y_train)
        results[name] = {"model": clf, "cv_roc_auc": cv_scores.mean()}
        print(f"  {name:25s}  CV ROC-AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results, scaler


# ── 5. Evaluate & pick best ───────────────────────────────────────────────────
def evaluate_and_save(results, scaler, X_test, y_test):
    best_name, best_roc = None, 0
    for name, r in results.items():
        X_test_sc = scaler.transform(X_test)
        y_pred    = r["model"].predict(X_test_sc)
        y_proba   = r["model"].predict_proba(X_test_sc)[:, 1]
        roc       = roc_auc_score(y_test, y_proba)
        acc       = accuracy_score(y_test, y_pred)
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"  Accuracy : {acc:.4f}   ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Showed Up", "No-Show"]))
        r["test_roc_auc"] = roc
        r["test_acc"]     = acc
        r["y_pred"]       = y_pred
        r["y_proba"]      = y_proba
        if roc > best_roc:
            best_roc, best_name = roc, name

    print(f"\n✅  Best model: {best_name}  (Test ROC-AUC = {best_roc:.4f})")

    # Save artifacts
    artifacts = {
        "best_model_name": best_name,
        "best_model":      results[best_name]["model"],
        "scaler":          scaler,
        "feature_cols":    FEATURE_COLS,
        "results_summary": {n: {"cv_roc": r["cv_roc_auc"],
                                "test_roc": r["test_roc_auc"],
                                "test_acc": r["test_acc"]}
                            for n, r in results.items()},
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"💾  Artifacts saved → {ARTIFACTS_PATH}")
    return results, best_name


# ── 6. Save EDA plots ─────────────────────────────────────────────────────────
def save_eda_plots(df: pd.DataFrame):
    # No-show rate
    fig, ax = plt.subplots(figsize=(5, 3))
    df["no_show"].value_counts().rename({0: "Showed Up", 1: "No-Show"}).plot(
        kind="bar", ax=ax, color=["#3b82d4", "#e05c5c"], edgecolor="white")
    ax.set_title("Class Distribution"); ax.set_ylabel("Count")
    plt.tight_layout(); fig.savefig(f"{PLOTS_DIR}/class_dist.png", dpi=120); plt.close()

    # Age distribution by no-show
    fig, ax = plt.subplots(figsize=(7, 3))
    for label, grp in df.groupby("no_show"):
        grp["age"].plot.kde(ax=ax, label="No-Show" if label else "Showed Up")
    ax.set_title("Age Distribution"); ax.legend()
    plt.tight_layout(); fig.savefig(f"{PLOTS_DIR}/age_dist.png", dpi=120); plt.close()

    # Wait days
    fig, ax = plt.subplots(figsize=(7, 3))
    df.boxplot(column="wait_days", by="no_show", ax=ax, grid=False)
    ax.set_xticklabels(["Showed Up", "No-Show"])
    ax.set_title("Wait Days by No-Show"); plt.suptitle("")
    plt.tight_layout(); fig.savefig(f"{PLOTS_DIR}/wait_days.png", dpi=120); plt.close()

    # Feature correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout(); fig.savefig(f"{PLOTS_DIR}/correlation.png", dpi=120); plt.close()
    print(f"📊  EDA plots saved → {PLOTS_DIR}/")


# ── 7. Public predict helper (used by Streamlit) ─────────────────────────────
def predict(input_dict: dict) -> dict:
    """
    input_dict keys must match FEATURE_COLS.
    Returns {'prediction': 0|1, 'no_show_probability': float}
    """
    artifacts = joblib.load(ARTIFACTS_PATH)
    model     = artifacts["best_model"]
    scaler    = artifacts["scaler"]
    feat_cols = artifacts["feature_cols"]

    row   = pd.DataFrame([input_dict])[feat_cols]
    row_sc = scaler.transform(row)
    pred  = int(model.predict(row_sc)[0])
    prob  = float(model.predict_proba(row_sc)[0][1])
    return {"prediction": pred, "no_show_probability": round(prob, 4)}


# ── 8. Main training pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Loading data …")
    df = load_data()
    print(f"   Shape: {df.shape}  |  No-show rate: {df['no_show'].mean()*100:.1f}%")

    print("\n── Engineering features …")
    df = engineer_features(df)

    print("\n── Saving EDA plots …")
    save_eda_plots(df)

    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n   Train: {X_train.shape}  Test: {X_test.shape}")

    print("\n── Training models …")
    results, scaler = train_models(X_train, y_train)

    print("\n── Evaluating models …")
    evaluate_and_save(results, scaler, X_test, y_test)

    print("\n✅  Backend pipeline complete.")
