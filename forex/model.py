"""
The learning loop: a calibrated P(target before stop) model over logged features.

Why logistic regression rather than gradient boosting
-----------------------------------------------------
The labelled set is ~900 trades, and those are far fewer *effective* observations
than they look: EUR_USD, GBP_USD, AUD_USD and NZD_USD firing long in the same scan
is one USD bet recorded four times. A boosted-tree model on a few hundred effective
samples with 29 features memorises that correlation structure and reports a
flattering in-sample number. An L2-regularised linear model in probability space is
the honest choice at this sample size, and it has two properties that matter more
than raw capacity here: the coefficients are directly readable (you can see which
features carry the edge), and the outputs are close to calibrated, which is required
because the decision rule compares the probability to a cost-adjusted breakeven.

Swap in LightGBM once there are several thousand resolved trades — ``ForexModel``
is deliberately just a ``predict_proba`` over the ``features`` contract, so the
trainer and the serving path do not care what is behind it.
"""
from __future__ import annotations

import json
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .features import FEATURE_NAMES, FEATURE_VERSION, to_vector

ALGO = "logistic_l2"

# Below this there is nothing to learn — refuse rather than emit a model that
# will confidently veto real setups on noise.
MIN_TRAIN_SAMPLES = 120


class ForexModel:
    """Standardise → L2 logistic regression → probability. Serialises to JSON."""

    def __init__(
        self,
        coefficients: Sequence[float],
        intercept: float,
        mean: Sequence[float],
        std: Sequence[float],
        feature_names: Sequence[str] = FEATURE_NAMES,
        feature_version: int = FEATURE_VERSION,
    ) -> None:
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.intercept = float(intercept)
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.feature_names = list(feature_names)
        self.feature_version = int(feature_version)

    # ── inference ───────────────────────────────────────────────────────────

    def predict_proba_vector(self, vector: Sequence[float]) -> float:
        x = (np.asarray(vector, dtype=float) - self.mean) / self.std
        z = float(np.dot(x, self.coefficients) + self.intercept)
        return _sigmoid(z)

    def predict_proba(self, features: dict) -> float:
        """P(target before stop) for a feature dict from ``features.build_features``."""
        return self.predict_proba_vector(
            [float(features.get(name, 0.0) or 0.0) for name in self.feature_names]
        )

    def top_coefficients(self, n: int = 10) -> List[Tuple[str, float]]:
        """Largest-magnitude standardised coefficients — i.e. what drives the edge."""
        pairs = list(zip(self.feature_names, self.coefficients.tolist()))
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return pairs[:n]

    # ── persistence ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps({
            "algo": ALGO,
            "feature_version": self.feature_version,
            "feature_names": self.feature_names,
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        })

    @classmethod
    def from_json(cls, payload: str) -> "ForexModel":
        d = json.loads(payload)
        return cls(
            coefficients=d["coefficients"],
            intercept=d["intercept"],
            mean=d["mean"],
            std=d["std"],
            feature_names=d.get("feature_names", list(FEATURE_NAMES)),
            feature_version=d.get("feature_version", FEATURE_VERSION),
        )


def _sigmoid(z: float) -> float:
    # Branch to avoid overflow in exp for large-magnitude logits.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _sigmoid_arr(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ── fitting ────────────────────────────────────────────────────────────────

def _fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 60,
) -> Tuple[np.ndarray, float]:
    """
    Newton-Raphson (IRLS) fit of L2-regularised logistic regression.

    X is already standardised. The intercept is fitted unpenalised as an extra
    column, so the model can match the base rate without regularisation dragging
    it toward 0.5. Converges in a handful of iterations at this problem size.
    """
    n, k = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])          # intercept last
    w = np.zeros(k + 1)
    # Penalty applies to slopes only — the intercept entry stays 0.
    penalty = np.full(k + 1, float(l2))
    penalty[-1] = 0.0
    ridge = np.diag(penalty)

    for _ in range(iters):
        p = _sigmoid_arr(Xb @ w)
        # Clamp so S never becomes exactly singular on a perfectly separated fold.
        s = np.clip(p * (1.0 - p), 1e-6, None)
        grad = Xb.T @ (y - p) - penalty * w
        hess = (Xb.T * s) @ Xb + ridge
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        w_new = w + step
        if not np.all(np.isfinite(w_new)):
            break
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new

    return w[:-1], float(w[-1])


def _standardise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Constant columns get std=1; centring makes them exactly zero so they
    # contribute nothing rather than producing a divide-by-zero.
    std[std < 1e-9] = 1.0
    return mean, std


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> ForexModel:
    mean, std = _standardise(X)
    Xs = (X - mean) / std
    coef, intercept = _fit_logistic(Xs, y, l2=l2)
    return ForexModel(coef, intercept, mean, std)


# ── metrics ────────────────────────────────────────────────────────────────

def roc_auc(y: Sequence[float], p: Sequence[float]) -> Optional[float]:
    """Rank-based AUC with proper tie handling. None if only one class present."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # average rank for ties
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def brier(y: Sequence[float], p: Sequence[float]) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def top_decile_precision(y: Sequence[float], p: Sequence[float], frac: float = 0.1) -> Optional[float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    k = max(1, int(round(len(p) * frac)))
    if len(p) < 10:
        return None
    idx = np.argsort(-p, kind="mergesort")[:k]
    return float(y[idx].mean())


def calibration_bins(y: Sequence[float], p: Sequence[float], bins: int = 5) -> list:
    """Predicted vs realised win rate per probability bucket."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if not m.any():
            continue
        out.append({
            "bin": i + 1,
            "n": int(m.sum()),
            "pred": round(float(p[m].mean()), 4),
            "actual": round(float(y[m].mean()), 4),
        })
    return out


# ── walk-forward evaluation ────────────────────────────────────────────────

def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    l2: float = 1.0,
    folds: int = 5,
    min_train: int = MIN_TRAIN_SAMPLES,
) -> dict:
    """
    Expanding-window out-of-sample evaluation over time-ordered rows.

    Every reported number comes from predictions made by a model that never saw
    the row it is scoring. In-sample metrics on this kind of data are meaningless
    — the point of the exercise is to find out whether the edge survives the
    walk forward, which is exactly what the hand-tuned scorer never had to prove.
    """
    n = len(y)
    usable = n - min_train
    if usable < folds * 5:
        folds = max(1, usable // 10) if usable >= 10 else 0
    if folds < 1:
        return {"error": f"not enough data for walk-forward (n={n}, min_train={min_train})"}

    step = usable // folds
    oos_p: List[float] = []
    oos_y: List[float] = []
    oos_r: List[float] = []
    fold_reports = []

    for i in range(folds):
        train_end = min_train + i * step
        test_end = n if i == folds - 1 else train_end + step
        if train_end >= n or test_end <= train_end:
            continue
        Xtr, ytr = X[:train_end], y[:train_end]
        if len(set(ytr.tolist())) < 2:
            continue
        m = fit(Xtr, ytr, l2=l2)
        Xte = X[train_end:test_end]
        preds = [m.predict_proba_vector(row) for row in Xte]
        yte = y[train_end:test_end]
        oos_p.extend(preds)
        oos_y.extend(yte.tolist())
        oos_r.extend(r[train_end:test_end].tolist())
        fold_reports.append({
            "fold": i + 1,
            "train_n": int(train_end),
            "test_n": int(test_end - train_end),
            "auc": roc_auc(yte, preds),
            "actual_win_rate": round(float(np.mean(yte)), 4),
        })

    if not oos_p:
        return {"error": "walk-forward produced no out-of-sample predictions"}

    return {
        "oos_p": oos_p,
        "oos_y": oos_y,
        "oos_r": oos_r,
        "folds": fold_reports,
        "auc": roc_auc(oos_y, oos_p),
        "brier": round(brier(oos_y, oos_p), 5),
        "top_decile_prec": top_decile_precision(oos_y, oos_p),
        "base_rate": round(float(np.mean(oos_y)), 4),
        "calibration": calibration_bins(oos_y, oos_p),
    }


def gated_performance(
    oos_p: Sequence[float], oos_r: Sequence[float], threshold: float,
) -> dict:
    """
    What the strategy would actually have returned taking only trades the model
    rated above ``threshold``. This is the number that decides whether the loop
    is working — AUC can look respectable while expectancy stays negative.
    """
    p = np.asarray(oos_p, dtype=float)
    r = np.asarray(oos_r, dtype=float)
    mask = p >= threshold
    taken = int(mask.sum())
    base = {
        "threshold": round(threshold, 4),
        "trades_taken": taken,
        "trades_available": int(len(p)),
        "selectivity": round(taken / len(p), 4) if len(p) else 0.0,
    }
    if taken == 0:
        base.update({"expectancy_r": None, "total_r": 0.0})
        return base
    base.update({
        "expectancy_r": round(float(r[mask].mean()), 4),
        "total_r": round(float(r[mask].sum()), 2),
        "baseline_expectancy_r": round(float(r.mean()), 4),
    })
    return base


def build_matrix(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert storage training rows into (X, y, r).

    y = 1 when the target was reached before the stop. r is the realised
    net-of-cost R, used to score the decision rule rather than the ranking.
    """
    X, y, r = [], [], []
    for row in rows:
        X.append(to_vector(row["features"]))
        y.append(1.0 if row.get("outcome") == "WIN" else 0.0)
        rm = row.get("r_multiple")
        r.append(float(rm) if rm is not None else 0.0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(r, dtype=float)
