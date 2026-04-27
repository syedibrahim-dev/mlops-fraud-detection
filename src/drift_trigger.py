"""
PSI + KS drift detector.
Monitors feature distributions between reference (training) window
and current (production) window. Triggers CL retraining when both signals fire.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1  : no significant shift
    PSI 0.1-0.2: moderate shift
    PSI > 0.2  : significant shift — trigger retraining
    """
    ref_min, ref_max = reference.min(), reference.max()
    if ref_max == ref_min:
        return 0.0

    breakpoints = np.linspace(ref_min, ref_max, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / len(current))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov test.
    Returns (statistic, p_value). p_value < 0.05 → significant drift.
    """
    stat, p_value = stats.ks_2samp(reference, current)
    return round(float(stat), 4), round(float(p_value), 6)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold: float = 0.2,
    ks_p_threshold: float = 0.05,
) -> dict:
    """
    Run PSI + KS on all features. Both must fire for drift to be confirmed.

    Returns:
        dict with per-feature scores, summary, and drift decision
    """
    feature_results = {}
    psi_fires = 0
    ks_fires = 0

    for col in feature_cols:
        ref = reference_df[col].values
        cur = current_df[col].values

        psi = compute_psi(ref, cur)
        ks_stat, ks_p = compute_ks(ref, cur)

        psi_fired = psi > psi_threshold
        ks_fired = ks_p < ks_p_threshold

        feature_results[col] = {
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "psi_fired": psi_fired,
            "ks_fired": ks_fired,
        }

        if psi_fired:
            psi_fires += 1
        if ks_fired:
            ks_fires += 1

    total = len(feature_cols)
    psi_ratio = psi_fires / total
    ks_ratio = ks_fires / total

    # Both detectors must agree on >30% of features to confirm drift
    drift_confirmed = psi_ratio > 0.3 and ks_ratio > 0.3

    # Top drifted features (ranked by PSI)
    top_drifted = sorted(
        feature_results.items(),
        key=lambda x: x[1]["psi"],
        reverse=True
    )[:5]

    return {
        "drift_confirmed": drift_confirmed,
        "psi_ratio": round(psi_ratio, 3),
        "ks_ratio": round(ks_ratio, 3),
        "psi_fires": psi_fires,
        "ks_fires": ks_fires,
        "top_drifted_features": [f for f, _ in top_drifted],
        "feature_results": feature_results,
    }


def summarize(result: dict):
    print(f"\nDrift Detection Summary")
    print(f"  PSI fired on {result['psi_fires']} features ({result['psi_ratio']*100:.1f}%)")
    print(f"  KS  fired on {result['ks_fires']} features ({result['ks_ratio']*100:.1f}%)")
    print(f"  Drift confirmed: {result['drift_confirmed']}")
    print(f"  Top drifted features: {result['top_drifted_features']}")
