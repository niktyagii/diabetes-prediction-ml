# ── 1. IMPORTS ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from sklearn.impute import SimpleImputer
import joblib

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed (optional)")

# ── 2. LOAD DATA ─────────────────────────────────────────────
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded dataset from: {filepath}")
    return df


# ── 3. EDA ───────────────────────────────────────────────────
def run_eda(df):
    print("\nShape:", df.shape)
    print("\nMissing:\n", df.isnull().sum())

    plt.figure()
    df["Outcome"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.show()

    plt.figure()
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.show()


# ── 4. PREPROCESS ────────────────────────────────────────────
def preprocess(df):
    ZERO_INVALID = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[ZERO_INVALID] = df[ZERO_INVALID].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42), scaler, imputer


# ── 5. MODELS ────────────────────────────────────────────────
def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss")

    return models


# ── 6. TRAIN ─────────────────────────────────────────────────
def train_models(models, X_train, X_test, y_train, y_test):
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n{name}")
        print("Accuracy:", acc)
        print("AUC:", auc)

        results[name] = (model, auc)

    return results


# ── 7. SAVE BEST MODEL ───────────────────────────────────────
def save_best(results, scaler, imputer):
    best_model = max(results, key=lambda x: results[x][1])
    model = results[best_model][0]

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(imputer, "imputer.pkl")

    print("\nBest model:", best_model, "saved!")


# ── 8. MAIN ──────────────────────────────────────────────────
def main():
    filepath = r"C:\Users\nikhi\OneDrive\Desktop\aimlp\dataset\diabetes.csv"   # 👈 your dataset path

    df = load_data(filepath)

    run_eda(df)

    (X_train, X_test, y_train, y_test), scaler, imputer = preprocess(df)

    models = build_models()

    results = train_models(models, X_train, X_test, y_train, y_test)

    save_best(results, scaler, imputer)


if __name__ == "__main__":
    main()