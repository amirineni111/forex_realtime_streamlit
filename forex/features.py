"""
Canonical feature extraction for the forex direction model.

This module is the single source of truth for what the model sees. Both the live
scanner (serving) and the trainer (from stored ``features_json``) go through
``build_features``, which is what keeps train/serve skew from creeping in.

Design note — everything here is **direction-relative**. A feature is expressed
from the point of view of the trade being taken, so the model learns "does this
kind of setup work" rather than having to learn long and short as separate
regimes. ``direction`` itself is kept as a feature so a genuine long/short
asymmetry can still be represented (the recorded history shows one: longs won
41.3% vs shorts 37.6% over 910 trades).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .timeutil import hour_utc

# Ordered feature contract. Changing this list invalidates stored models, which
# is why FEATURE_VERSION travels with every persisted model and every logged row.
FEATURE_NAMES: Sequence[str] = (
    "direction",
    "total_score",
    "momentum_score",
    "reversion_score",
    "session_score",
    "mtf_score",
    "sr_score",
    "adx14",
    "rsi_dir",
    "atr_pct",
    "bb_width_pct",
    "spread_pips",
    "stop_pips",
    "cost_ratio",
    "ema_gap_atr",
    "macd_hist_atr",
    "dist_to_target_level_atr",
    "dist_to_protective_level_atr",
    "h1_agrees",
    "h4_agrees",
    "strength_diff_dir",
    "range_pos_dir",
    "day_change_dir",
    "hour_sin",
    "hour_cos",
    "sess_overlap",
    "sess_london",
    "sess_ny",
    "sess_asian",
)

# Bumped 1 -> 2 when the signal timeframe moved from M5 to M15.
#
# The feature *names* did not change, which is exactly why this bump is necessary
# rather than optional: every ATR-normalised ratio, the RSI, the session-range
# position and the stop size all mean something different on a 15-minute bar than on
# a 5-minute one. M5 ATR on a major is 2-4 pips and M15 is roughly 3x that, so an
# identically-named feature is drawn from a different distribution.
#
# Pooling the two would train a model on a mixture of two populations and then serve
# it to one of them. Since ``load_training_rows`` filters on this column, the bump is
# what keeps the ~1,000 M5-era rows out of the next retrain, and it also invalidates
# the stored M5 model (walk-forward AUC 0.484 — below chance) rather than letting it
# be served against inputs it never saw.
FEATURE_VERSION = 2

# Cap for ATR-normalised ratios. Without this a near-zero ATR turns one bar into
# an enormous outlier that dominates a linear model's fit.
_CLIP = 10.0


def _f(value: Optional[float], default: float = 0.0) -> float:
    """Coerce to float, mapping None/NaN/inf to a neutral default."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _clip(value: float, limit: float = _CLIP) -> float:
    return max(-limit, min(limit, value))


def _agreement(htf_direction: Optional[str], direction: int) -> float:
    """+1 when the higher timeframe backs the trade, -1 when it fights it, 0 if flat."""
    if htf_direction not in ("LONG", "SHORT"):
        return 0.0
    htf = 1 if htf_direction == "LONG" else -1
    return 1.0 if htf == direction else -1.0


_hour_utc = hour_utc


def build_features(snap: dict, direction: int) -> dict:
    """
    Build the model feature dict for taking ``direction`` (+1 long / -1 short) on
    the setup described by ``snap`` (a ForexSnapshot dump or a forex_snapshots row).

    Every value is finite; missing inputs collapse to a neutral 0.0 so a partially
    populated snapshot still scores rather than raising.
    """
    d = 1.0 if direction >= 0 else -1.0

    close = _f(snap.get("close"))
    atr = _f(snap.get("atr14"))
    # Guard every ATR-normalised ratio behind a positive-ATR check.
    atr_ok = atr > 0

    spread = _f(snap.get("spread_pips"))
    stop_pips = _f(snap.get("stop_pips"))
    cost_ratio = (spread / stop_pips) if stop_pips > 0 else 0.0

    rsi = snap.get("rsi14")
    # Orient RSI to the trade: high means "RSI supports this direction".
    rsi_dir = (_f(rsi, 50.0) if d > 0 else 100.0 - _f(rsi, 50.0)) if rsi is not None else 50.0

    ema_gap_atr = (
        _clip((_f(snap.get("ema9")) - _f(snap.get("ema20"))) / atr * d) if atr_ok else 0.0
    )
    macd_hist_atr = (
        _clip(_f(snap.get("macd_histogram")) / atr * d) if atr_ok else 0.0
    )

    # Structure ahead of / behind the trade. For a long the "target level" is the
    # resistance overhead (how much runway before it stalls) and the "protective
    # level" is the support beneath. Mirrored for a short. This is what the old
    # direction-blind S/R bonus failed to distinguish.
    support = snap.get("nearest_support")
    resistance = snap.get("nearest_resistance")
    ahead = resistance if d > 0 else support
    behind = support if d > 0 else resistance

    if atr_ok and ahead is not None and close:
        dist_target = _clip(abs(_f(ahead) - close) / atr)
    else:
        dist_target = 0.0
    if atr_ok and behind is not None and close:
        dist_protective = _clip(abs(close - _f(behind)) / atr)
    else:
        dist_protective = 0.0

    strength_diff_dir = (
        (_f(snap.get("base_strength"), 50.0) - _f(snap.get("quote_strength"), 50.0)) * d
        if snap.get("base_strength") is not None and snap.get("quote_strength") is not None
        else 0.0
    )

    # Where price sits in the session range, oriented to the trade.
    # +1 = extended in our favour (breakout entry), -1 = against us (reversion entry).
    hi = _f(snap.get("session_high"))
    lo = _f(snap.get("session_low"))
    if hi > lo and close:
        pos = (close - lo) / (hi - lo)
        range_pos_dir = _clip((pos - 0.5) * 2.0 * d, 2.0)
    else:
        range_pos_dir = 0.0

    hour = _hour_utc(snap.get("as_of"))
    if hour is None:
        hour_sin = hour_cos = 0.0
    else:
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)

    session = snap.get("current_session") or ""

    return {
        "direction": d,
        "total_score": _f(snap.get("total_score")),
        "momentum_score": _f(snap.get("momentum_score")),
        "reversion_score": _f(snap.get("reversion_score")),
        "session_score": _f(snap.get("session_score")),
        "mtf_score": _f(snap.get("mtf_score")),
        "sr_score": _f(snap.get("sr_score")),
        "adx14": _f(snap.get("adx14"), 20.0),
        "rsi_dir": rsi_dir,
        # Scale-free volatility. Raw ATR is not comparable across pairs (USD_JPY
        # ATR is ~100x EUR_USD ATR in price units), which would otherwise let the
        # model use ATR as a pair identifier.
        "atr_pct": _clip(atr / close * 100.0, 5.0) if close else 0.0,
        "bb_width_pct": _f(snap.get("bb_width_pct")),
        "spread_pips": spread,
        "stop_pips": stop_pips,
        "cost_ratio": _clip(cost_ratio, 2.0),
        "ema_gap_atr": ema_gap_atr,
        "macd_hist_atr": macd_hist_atr,
        "dist_to_target_level_atr": dist_target,
        "dist_to_protective_level_atr": dist_protective,
        "h1_agrees": _agreement(snap.get("h1_direction"), int(d)),
        "h4_agrees": _agreement(snap.get("h4_direction"), int(d)),
        "strength_diff_dir": strength_diff_dir,
        "range_pos_dir": range_pos_dir,
        "day_change_dir": _clip(_f(snap.get("day_change_pct")) * d, 5.0),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "sess_overlap": 1.0 if session == "London_NY_Overlap" else 0.0,
        "sess_london": 1.0 if session == "London" else 0.0,
        "sess_ny": 1.0 if session == "New_York" else 0.0,
        "sess_asian": 1.0 if session == "Asian" else 0.0,
    }


def to_vector(features: dict) -> list:
    """Flatten a feature dict into FEATURE_NAMES order."""
    return [_f(features.get(name)) for name in FEATURE_NAMES]
