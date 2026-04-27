"""
Training script for fraud detection model.
Logs parameters, metrics, and model artifact to MLflow.

Usage:
    python src/train.py --data data/creditcard.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import mlflow
import mlflow.xgboost

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fraud-detection"


def load_and_preprocess(data_path: str):
    df = pd.read_csv(data_path)

    # Scale Amount and Time (V1-V28 are already PCA-scaled)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    return X, y, df.drop("Class", axis=1).columns.tolist()


def train(data_path: str, test_size: float = 0.2, random_state: int = 42):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, feature_names = load_and_preprocess(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "random_state": random_state,
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("smote", True)
        mlflow.log_param("train_samples", len(X_train_res))
        mlflow.log_param("fraud_ratio_original", round(y.mean(), 4))

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_res, y_train_res,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "avg_precision": round(average_precision_score(y_test, y_proba), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "fraud_recall": round(
                classification_report(y_test, y_pred, output_dict=True)["1"]["recall"], 4
            ),
        }

        mlflow.log_metrics(metrics)
        print("\nMetrics:", metrics)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

        # Save model + feature names locally and log as artifacts
        model.save_model("data/model.json")
        joblib.dump(feature_names, "data/feature_names.pkl")
        mlflow.log_artifact("data/model.json", artifact_path="model")
        mlflow.log_artifact("data/feature_names.pkl", artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        return run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    train(args.data, args.test_size)
