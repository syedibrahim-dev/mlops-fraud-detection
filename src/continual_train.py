"""
Continual Learning pipeline for fraud detection.

Implements 3 strategies from Lebichot et al. (Expert Systems w/ Apps, 2024):
  1. Fine-Tuning  (FT)  — retrain on new window only (baseline, forgets old)
  2. Experience Replay (ER) — mix new window with buffer of old samples
  3. Sliding Window   (SW) — retrain on last K windows of data

Evaluates each strategy using the Plasticity-Stability matrix:
  - Plasticity : F1 on the newest window (learning ability)
  - Stability  : F1 on old windows (retention of past knowledge)
  - Forgetting : performance drop on old windows over time

All results logged to MLflow. Model artifacts saved per window.

Usage:
    python src/continual_train.py --data data/creditcard.csv --n-windows 8
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import mlflow
import mlflow.xgboost

from plasticity import compute_plasticity_stability, log_to_mlflow, print_matrix
from drift_trigger import detect_drift, summarize as drift_summarize

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fraud-continual-learning"
REPLAY_BUFFER_SIZE = 500   # samples per old window kept in replay buffer
SLIDING_WINDOW_K = 3       # number of past windows to keep for SW strategy


# ── Data preparation ─────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"]   = scaler.fit_transform(df[["Time"]])
    return df


def split_into_windows(df: pd.DataFrame, n_windows: int) -> list[pd.DataFrame]:
    """Split dataset into n equal time-based windows using the Time column."""
    edges = np.linspace(df["Time"].min(), df["Time"].max(), n_windows + 1)
    windows = []
    for i in range(n_windows):
        w = df[(df["Time"] >= edges[i]) & (df["Time"] < edges[i + 1])].copy()
        windows.append(w)
    return windows


def window_to_xy(window: pd.DataFrame):
    X = window.drop("Class", axis=1).values
    y = window["Class"].values
    return X, y


def apply_smote(X, y, random_state=42):
    """Apply SMOTE only if there are enough minority samples."""
    if y.sum() < 2:
        return X, y
    try:
        sm = SMOTE(random_state=random_state, k_neighbors=min(5, y.sum() - 1))
        return sm.fit_resample(X, y)
    except Exception:
        return X, y


# ── Model training ────────────────────────────────────────────────────────────

def make_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=random_state,
        verbosity=0,
    )


def fit_model(X_train, y_train, X_val=None, y_val=None) -> xgb.XGBClassifier:
    X_tr, y_tr = apply_smote(X_train, y_train)
    model = make_model()
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
    return model


# ── CL Strategies ─────────────────────────────────────────────────────────────

def strategy_finetune(
    windows: list[pd.DataFrame],
    eval_sets: list[tuple]
) -> tuple[list, list]:
    """
    Fine-Tuning: retrain from scratch on new window only.
    Fastest but forgets old patterns completely.
    """
    models = []
    results = [[None] * len(windows) for _ in range(len(windows))]

    for i, window in enumerate(windows):
        X_train, y_train = window_to_xy(window)
        model = fit_model(X_train, y_train)
        models.append(model)

        for j in range(i + 1):
            X_eval, y_eval = eval_sets[j]
            from plasticity import evaluate_on_window
            results[i][j] = evaluate_on_window(model, X_eval, y_eval)

        print(f"  [FT]  Window {i+1}/{len(windows)} trained | "
              f"F1={results[i][i]['f1']:.3f}")

    return models, results


def strategy_experience_replay(
    windows: list[pd.DataFrame],
    eval_sets: list[tuple],
    buffer_size: int = REPLAY_BUFFER_SIZE
) -> tuple[list, list]:
    """
    Experience Replay: keep a fixed-size buffer of old samples.
    Mix buffer + new window for each retraining step.
    """
    models = []
    results = [[None] * len(windows) for _ in range(len(windows))]
    replay_buffer = pd.DataFrame()

    for i, window in enumerate(windows):
        if replay_buffer.empty:
            train_data = window
        else:
            # Sample from replay buffer
            buf_sample = replay_buffer.sample(
                n=min(buffer_size, len(replay_buffer)), random_state=42
            )
            train_data = pd.concat([buf_sample, window], ignore_index=True)

        X_train, y_train = window_to_xy(train_data)
        model = fit_model(X_train, y_train)
        models.append(model)

        # Update replay buffer with samples from current window
        new_samples = window.sample(
            n=min(buffer_size // max(1, i), len(window)), random_state=42
        )
        replay_buffer = pd.concat([replay_buffer, new_samples], ignore_index=True)

        for j in range(i + 1):
            X_eval, y_eval = eval_sets[j]
            from plasticity import evaluate_on_window
            results[i][j] = evaluate_on_window(model, X_eval, y_eval)

        print(f"  [ER]  Window {i+1}/{len(windows)} trained | "
              f"F1={results[i][i]['f1']:.3f} | "
              f"Buffer={len(replay_buffer)}")

    return models, results


def strategy_sliding_window(
    windows: list[pd.DataFrame],
    eval_sets: list[tuple],
    k: int = SLIDING_WINDOW_K
) -> tuple[list, list]:
    """
    Sliding Window: retrain on the last K windows.
    Balances recency with some retention of recent history.
    """
    models = []
    results = [[None] * len(windows) for _ in range(len(windows))]

    for i, window in enumerate(windows):
        start = max(0, i - k + 1)
        train_data = pd.concat(windows[start:i + 1], ignore_index=True)

        X_train, y_train = window_to_xy(train_data)
        model = fit_model(X_train, y_train)
        models.append(model)

        for j in range(i + 1):
            X_eval, y_eval = eval_sets[j]
            from plasticity import evaluate_on_window
            results[i][j] = evaluate_on_window(model, X_eval, y_eval)

        print(f"  [SW]  Window {i+1}/{len(windows)} trained | "
              f"F1={results[i][i]['f1']:.3f} | "
              f"Using windows {start+1}..{i+1}")

    return models, results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(data_path: str, n_windows: int = 8):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    windows = split_into_windows(df, n_windows)

    print(f"Split into {n_windows} windows:")
    for i, w in enumerate(windows):
        print(f"  Window {i+1}: {len(w):6d} txns, {w['Class'].sum():3d} fraud "
              f"({w['Class'].mean()*100:.2f}%)")

    # Build held-out eval sets (20% of each window)
    eval_sets = []
    train_windows = []
    for w in windows:
        X, y = window_to_xy(w)
        if len(np.unique(y)) > 1:
            X_tr, X_ev, y_tr, y_ev = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        else:
            X_tr, X_ev, y_tr, y_ev = X, X, y, y
        eval_sets.append((X_ev, y_ev))
        # Replace window data with training portion only
        train_windows.append(
            pd.DataFrame(
                np.column_stack([X_tr, y_tr]),
                columns=df.columns.tolist()
            ).assign(Class=y_tr)
        )

    # ── Drift detection between consecutive windows ──────────────────────────
    print("\nRunning drift detection between windows...")
    feature_cols = [c for c in df.columns if c != "Class"]
    drift_log = []
    for i in range(1, n_windows):
        result = detect_drift(
            windows[i - 1].drop("Class", axis=1),
            windows[i].drop("Class", axis=1),
            feature_cols
        )
        drift_log.append(result)
        print(f"  W{i}→W{i+1}: drift={'YES' if result['drift_confirmed'] else 'no'} | "
              f"PSI={result['psi_ratio']:.2f} KS={result['ks_ratio']:.2f}")

    # ── Run all 3 strategies ─────────────────────────────────────────────────
    strategies = {
        "finetune": strategy_finetune,
        "experience_replay": strategy_experience_replay,
        "sliding_window": strategy_sliding_window,
    }

    all_scores = {}

    with mlflow.start_run(run_name="continual_learning_comparison"):
        mlflow.log_param("n_windows", n_windows)
        mlflow.log_param("dataset", data_path)
        mlflow.log_param("replay_buffer_size", REPLAY_BUFFER_SIZE)
        mlflow.log_param("sliding_window_k", SLIDING_WINDOW_K)

        # Log drift summary
        confirmed_drifts = sum(1 for d in drift_log if d["drift_confirmed"])
        mlflow.log_metric("drift_events_detected", confirmed_drifts)

        for name, strategy_fn in strategies.items():
            print(f"\nRunning strategy: {name.upper()}")
            _, results = strategy_fn(train_windows, eval_sets)

            scores = compute_plasticity_stability(results, metric="f1")
            all_scores[name] = scores
            print_matrix(scores, name)

            # Log to MLflow
            mlflow.log_metrics({
                f"{name}_plasticity": scores["plasticity"],
                f"{name}_stability": scores["stability"],
                f"{name}_forgetting": scores["forgetting"],
            })

            # Save matrix as CSV artifact
            matrix_path = f"data/{name}_plasticity_matrix.csv"
            scores["matrix_df"].to_csv(matrix_path)
            mlflow.log_artifact(matrix_path)

        # Summary comparison
        print("\n\nFINAL COMPARISON")
        print(f"{'Strategy':<20} {'Plasticity':>12} {'Stability':>12} {'Forgetting':>12}")
        print("-" * 60)
        for name, scores in all_scores.items():
            print(f"{name:<20} {scores['plasticity']:>12.4f} "
                  f"{scores['stability']:>12.4f} {scores['forgetting']:>12.4f}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--n-windows", type=int, default=8)
    args = parser.parse_args()

    run(args.data, args.n_windows)
