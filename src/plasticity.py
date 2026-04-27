"""
Plasticity-Stability matrix computation for continual learning evaluation.

Based on: Lebichot et al., "Assessment of Catastrophic Forgetting in
Continual Credit Card Fraud Detection", Expert Systems with Applications, 2024.

- Plasticity: how well the model learns NEW patterns (diagonal of matrix)
- Stability: how well the model RETAINS old patterns (below-diagonal)
- Forgetting: drop from best-ever performance on a window to current performance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from typing import Optional
import mlflow


def evaluate_on_window(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate model on a single time window."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, y_proba) if y.sum() > 0 else 0.0, 4),
        "avg_precision": round(average_precision_score(y, y_proba) if y.sum() > 0 else 0.0, 4),
    }


def compute_plasticity_stability(
    results: list[list[Optional[dict]]],
    metric: str = "f1"
) -> dict:
    """
    Compute plasticity, stability, and forgetting from a results matrix.

    Args:
        results: results[i][j] = metrics dict when model trained up to window i
                 evaluated on window j. None if j > i (future window).
        metric: which metric to use for scoring ("f1", "roc_auc", "avg_precision")

    Returns:
        dict with plasticity, stability, forgetting scores + full matrix
    """
    n = len(results)
    matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            if results[i][j] is not None:
                matrix[i][j] = results[i][j][metric]

    # Plasticity: diagonal — how well model learned the newest window
    plasticity_scores = [matrix[i][i] for i in range(n) if not np.isnan(matrix[i][i])]
    plasticity = round(float(np.mean(plasticity_scores)), 4) if plasticity_scores else 0.0

    # Stability: below-diagonal — how well model retains old windows
    stability_scores = []
    for i in range(1, n):
        for j in range(i):
            if not np.isnan(matrix[i][j]):
                stability_scores.append(matrix[i][j])
    stability = round(float(np.mean(stability_scores)), 4) if stability_scores else 0.0

    # Forgetting: for each old window j, max performance ever - current performance
    forgetting_scores = []
    for j in range(n):
        col = [matrix[i][j] for i in range(j, n) if not np.isnan(matrix[i][j])]
        if len(col) > 1:
            forgetting_scores.append(col[0] - col[-1])  # first (best) - last (current)
    forgetting = round(float(np.mean(forgetting_scores)), 4) if forgetting_scores else 0.0

    return {
        "plasticity": plasticity,
        "stability": stability,
        "forgetting": forgetting,
        "matrix": matrix,
        "matrix_df": pd.DataFrame(
            matrix,
            index=[f"trained_w{i+1}" for i in range(n)],
            columns=[f"eval_w{j+1}" for j in range(n)]
        )
    }


def log_to_mlflow(scores: dict, strategy: str, window_idx: int):
    """Log plasticity-stability scores to MLflow for current window."""
    prefix = f"w{window_idx+1}_{strategy}"
    mlflow.log_metrics({
        f"{prefix}_plasticity": scores["plasticity"],
        f"{prefix}_stability": scores["stability"],
        f"{prefix}_forgetting": scores["forgetting"],
    })


def print_matrix(scores: dict, strategy: str):
    """Pretty-print the plasticity-stability matrix."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.upper()}")
    print(f"Plasticity (learn new): {scores['plasticity']:.4f}")
    print(f"Stability  (keep old) : {scores['stability']:.4f}")
    print(f"Forgetting            : {scores['forgetting']:.4f}")
    print(f"\nF1 Matrix (row=trained_up_to_window, col=eval_window):")
    print(scores["matrix_df"].to_string())
    print('='*60)
