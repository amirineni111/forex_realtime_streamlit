"""
The retrain pipeline, shared by the CLI and the dashboard.

Both entry points call ``evaluate_and_fit`` so a model promoted from the Model tab
is the same object, judged by the same numbers, as one promoted from
``scripts/train_model.py``. Splitting the logic across the two would let the
reported metrics drift apart, which is the sort of divergence that quietly erodes
trust in the gate.
"""
from __future__ import annotations

import statistics
from typing import Optional

import numpy as np

from .features import FEATURE_VERSION
from .model import (
    ALGO, MIN_TRAIN_SAMPLES, build_matrix, fit, gated_performance, walk_forward,
)
from .signals import _PROB_MARGIN, _RR, breakeven_win_rate
from .storage import Storage

# Candidate decision thresholds reported on every retrain: zero-cost breakeven,
# realistic-cost breakeven, and two round numbers for reference.
def _candidate_thresholds() -> list:
    return sorted({
        round(breakeven_win_rate(_RR, 0.0) + _PROB_MARGIN, 4),
        round(breakeven_win_rate(_RR, 0.125) + _PROB_MARGIN, 4),
        0.50,
        0.55,
    })


def evaluate_and_fit(
    storage: Storage,
    l2: float = 1.0,
    folds: int = 5,
    min_auc: float = 0.53,
) -> dict:
    """
    Walk-forward evaluate, then fit a final model on everything.

    Returns a report dict. ``report["error"]`` is set (and ``model`` is None) when
    there is not enough data — callers should surface it rather than proceeding.
    Nothing is written to the database here; persistence is the caller's decision.
    """
    rows = storage.load_training_rows(feature_version=FEATURE_VERSION)
    n = len(rows)
    if n < MIN_TRAIN_SAMPLES:
        return {
            "error": (
                f"Not enough data to train: {n} resolved trades with stored features, "
                f"{MIN_TRAIN_SAMPLES} required."
            ),
            "n_rows": n,
            "model": None,
        }

    X, y, r = build_matrix(rows)
    wf = walk_forward(X, y, r, l2=l2, folds=folds)
    if "error" in wf:
        return {"error": wf["error"], "n_rows": n, "model": None}

    gated = [gated_performance(wf["oos_p"], wf["oos_r"], t) for t in _candidate_thresholds()]
    scored = [g for g in gated if g.get("expectancy_r") is not None]
    best = max(scored, key=lambda g: g["expectancy_r"]) if scored else None

    fold_aucs = [f["auc"] for f in wf["folds"] if f["auc"] is not None]
    fold_auc_std = round(statistics.pstdev(fold_aucs), 4) if len(fold_aucs) > 1 else None

    model = fit(X, y, l2=l2)
    baseline = round(float(np.mean(wf["oos_r"])), 4)
    auc = wf["auc"]

    passes = bool(
        auc is not None and auc >= min_auc
        and best is not None and best["expectancy_r"] > 0
    )

    metrics = {
        "algo": ALGO,
        "feature_version": FEATURE_VERSION,
        "n_train": int(n),
        "n_test": int(len(wf["oos_y"])),
        "auc": auc,
        "brier": wf["brier"],
        "top_decile_prec": wf["top_decile_prec"],
        "base_rate": wf["base_rate"],
        "calibration": wf["calibration"],
        "folds": wf["folds"],
        "fold_auc_std": fold_auc_std,
        "baseline_expectancy_r": baseline,
        "best_threshold": best["threshold"] if best else None,
        "best_expectancy_r": best["expectancy_r"] if best else None,
        "l2": l2,
        "min_auc": min_auc,
    }

    return {
        "error": None,
        "model": model,
        "metrics": metrics,
        "gated": gated,
        "best": best,
        "baseline_expectancy_r": baseline,
        "calibration": wf["calibration"],
        "folds": wf["folds"],
        "fold_auc_std": fold_auc_std,
        "coefficients": model.top_coefficients(15),
        "passes": passes,
        "n_rows": n,
        "base_rate": wf["base_rate"],
    }


def save_candidate(storage: Storage, report: dict, notes: str = "") -> Optional[int]:
    """
    Persist a freshly trained model as an **inactive** candidate.

    Saving and promoting are separate on purpose: a model should be reviewed
    against its own metrics before it starts vetoing live signals.
    """
    if report.get("error") or report.get("model") is None:
        return None
    return storage.save_model(
        report["model"].to_json(), report["metrics"], activate=False, notes=notes,
    )


def gate_summary(report: dict) -> str:
    """One-line human verdict for the report."""
    if report.get("error"):
        return report["error"]
    auc = report["metrics"]["auc"]
    best = report.get("best")
    exp = best["expectancy_r"] if best else None
    min_auc = report["metrics"]["min_auc"]
    if report["passes"]:
        return (
            f"PASS - out-of-sample AUC {auc:.4f} (>= {min_auc}) and best gated "
            f"expectancy {exp:+.4f}R at p>={best['threshold']:.3f}."
        )
    reasons = []
    if auc is None or auc < min_auc:
        reasons.append(f"AUC {auc:.4f} below the {min_auc} minimum" if auc is not None
                       else "AUC could not be computed")
    if exp is None or exp <= 0:
        reasons.append(
            f"best gated expectancy {exp:+.4f}R is not positive" if exp is not None
            else "no threshold produced any trades"
        )
    return "FAIL - " + "; ".join(reasons) + "."
