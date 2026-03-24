import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST = True
except ImportError:
    XGBOOST = False

DATASET_NAMES = [
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "Telco-Customer-Churn.csv",
    "telco_churn.csv",
]


@st.cache_resource(show_spinner="Training model on Telco dataset…")
def load_model_and_data():
    df = _load_csv()
    df = _clean(df)
    model, scaler, feature_cols, metrics = _train(df)
    return model, scaler, feature_cols, df, metrics


def _load_csv() -> pd.DataFrame:
    for name in DATASET_NAMES:
        if os.path.isfile(name):
            return pd.read_csv(name)

    # kagglehub fallback
    try:
        import kagglehub
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    return pd.read_csv(os.path.join(root, f))
    except Exception:
        pass

    st.error(
        "Dataset not found. Download **WA_Fn-UseC_-Telco-Customer-Churn.csv** from "
        "[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
        "and place it in the project folder."
    )
    st.stop()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    # Feature engineering
    df["AvgMonthlySpend"]    = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsNewCustomer"]      = (df["tenure"] <= 3).astype(int)
    df["IsLongTermCustomer"] = (df["tenure"] >= 48).astype(int)
    df = pd.get_dummies(df, drop_first=True)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def _train(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if XGBOOST:
        model = XGBClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            verbosity=0
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )

    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    proba = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)

    metrics = {
        "accuracy":  accuracy_score(y_test, preds),
        "auc":       roc_auc_score(y_test, proba),
        "f1":        f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall":    recall_score(y_test, preds),
        "cm":        confusion_matrix(y_test, preds).tolist(),
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
        "y_test":    y_test.tolist(),
        "proba":     proba.tolist(),
        "model_name": "XGBoost" if XGBOOST else "Random Forest",
    }
    return model, scaler, feature_cols, metrics
