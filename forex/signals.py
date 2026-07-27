from __future__ import annotations
import json
from typing import List, Optional, Tuple

from .market_sessions import current_session


def _macd_magnitude_pts(macd_histogram: float, atr14: Optional[float], macd: Optional[float]) -> float:
    """
    Convert MACD-histogram size into 0-15 pts. Normalizes against ATR (both in price
    units, so the ratio is unitless): a histogram ~0.3×ATR is treated as full strength.
    Falls back to the legacy hist/macd ratio when ATR is unavailable.
    """
    if atr14 and atr14 > 0:
        return min(15.0, 50.0 * abs(macd_histogram) / atr14)
    return min(15.0, 15 * abs(macd_histogram) / max(abs(macd or 0.0), 0.00001))


def _momentum(
    ema9: Optional[float],
    ema20: Optional[float],
    macd_histogram: Optional[float],
    macd: Optional[float],
    rsi14: Optional[float],
    atr14: Optional[float] = None,
) -> Tuple[float, str, str]:
    """Score momentum signal (max 40 pts). Returns (score, direction, reason)."""
    score = 0.0
    reasons = []
    direction = "NEUTRAL"

    # EMA alignment (max 15 pts)
    if ema9 is not None and ema20 is not None:
        if ema9 > ema20:
            score += 15
            direction = "LONG"
            reasons.append("EMA9>EMA20 bullish")
        elif ema9 < ema20:
            score += 15
            direction = "SHORT"
            reasons.append("EMA9<EMA20 bearish")

    # MACD histogram direction + ATR-normalized strength (max 15 pts)
    if macd_histogram is not None and macd is not None:
        if macd_histogram > 0 and macd > 0:
            pts = _macd_magnitude_pts(macd_histogram, atr14, macd)
            score += pts
            reasons.append(f"MACD bullish (+{pts:.0f}pts)")
        elif macd_histogram < 0 and macd < 0:
            pts = _macd_magnitude_pts(macd_histogram, atr14, macd)
            score += pts
            reasons.append(f"MACD bearish (+{pts:.0f}pts)")
        elif macd_histogram > 0:  # histogram positive but MACD crossing zero
            score += 8
            reasons.append("MACD histogram positive crossing")
        elif macd_histogram < 0:
            score += 8
            reasons.append("MACD histogram negative crossing")

    # RSI 40-60 trending confirmation (max 10 pts)
    if rsi14 is not None:
        if direction == "LONG" and 40 <= rsi14 <= 65:
            score += 10
            reasons.append(f"RSI {rsi14:.1f} momentum zone")
        elif direction == "SHORT" and 35 <= rsi14 <= 60:
            score += 10
            reasons.append(f"RSI {rsi14:.1f} momentum zone")

    signal = f"LONG_MOMENTUM" if direction == "LONG" else (
        "SHORT_MOMENTUM" if direction == "SHORT" else "NEUTRAL_MOMENTUM"
    )
    return round(score, 1), signal, "; ".join(reasons)


def _mean_reversion(
    rsi14: Optional[float],
    close: Optional[float],
    bb_upper: Optional[float],
    bb_lower: Optional[float],
    bb_middle: Optional[float],
    session_high: Optional[float],
    session_low: Optional[float],
) -> Tuple[float, str, str]:
    """Score mean reversion signal (max 40 pts). Returns (score, direction, reason)."""
    score = 0.0
    reasons = []
    direction = "NEUTRAL"

    # RSI extreme (max 20 pts)
    if rsi14 is not None:
        if rsi14 < 30:
            pts = 20 * (30 - rsi14) / 30
            score += min(20, pts)
            direction = "LONG"
            reasons.append(f"RSI oversold {rsi14:.1f}")
        elif rsi14 > 70:
            pts = 20 * (rsi14 - 70) / 30
            score += min(20, pts)
            direction = "SHORT"
            reasons.append(f"RSI overbought {rsi14:.1f}")

    # Bollinger Band proximity (max 10 pts)
    if close is not None and bb_upper is not None and bb_lower is not None and bb_middle is not None:
        band_width = bb_upper - bb_lower
        if band_width > 0:
            dist_lower = (close - bb_lower) / band_width
            dist_upper = (bb_upper - close) / band_width
            if dist_lower <= 0.15:
                score += 10
                direction = "LONG"
                reasons.append("Price at lower Bollinger Band")
            elif dist_upper <= 0.15:
                score += 10
                direction = "SHORT"
                reasons.append("Price at upper Bollinger Band")
            elif dist_lower <= 0.30:
                score += 5
                direction = direction if direction != "NEUTRAL" else "LONG"
                reasons.append("Price near lower Bollinger Band")
            elif dist_upper <= 0.30:
                score += 5
                direction = direction if direction != "NEUTRAL" else "SHORT"
                reasons.append("Price near upper Bollinger Band")

    # Session range position (max 10 pts)
    if (
        close is not None
        and session_high is not None
        and session_low is not None
        and session_high > session_low
    ):
        session_range = session_high - session_low
        pos = (close - session_low) / session_range
        if pos <= 0.30:
            score += 10
            direction = "LONG" if direction == "NEUTRAL" else direction
            reasons.append(f"Bottom 30% of session range ({pos*100:.0f}%)")
        elif pos >= 0.70:
            score += 10
            direction = "SHORT" if direction == "NEUTRAL" else direction
            reasons.append(f"Top 30% of session range ({pos*100:.0f}%)")

    signal = "LONG_REVERSION" if direction == "LONG" else (
        "SHORT_REVERSION" if direction == "SHORT" else "NEUTRAL_REVERSION"
    )
    return round(score, 1), signal, "; ".join(reasons)


def _session_breakout(
    close: Optional[float],
    session_high: Optional[float],
    session_low: Optional[float],
    atr14: Optional[float],
    session: Optional[str],
) -> Tuple[float, str, str]:
    """Score session breakout signal (max 20 pts). Returns (score, direction, reason)."""
    score = 0.0
    reasons = []
    direction = "NEUTRAL"

    # Session quality bonus (max 10 pts)
    if session == "London_NY_Overlap":
        score += 10
        reasons.append("London/NY overlap (most liquid)")
    elif session in ("London", "New_York"):
        score += 5
        reasons.append(f"{session} session active")

    # Breakout beyond session high/low by ≥ 1×ATR (max 10 pts)
    if (
        close is not None
        and session_high is not None
        and session_low is not None
        and atr14 is not None
        and atr14 > 0
    ):
        if close > session_high:
            breakout_dist = (close - session_high) / atr14
            pts = min(10, 10 * breakout_dist)
            score += pts
            direction = "LONG"
            reasons.append(f"Breaking session high ({breakout_dist:.1f}×ATR)")
        elif close < session_low:
            breakout_dist = (session_low - close) / atr14
            pts = min(10, 10 * breakout_dist)
            score += pts
            direction = "SHORT"
            reasons.append(f"Breaking session low ({breakout_dist:.1f}×ATR)")

    signal = "LONG_BREAKOUT" if direction == "LONG" else (
        "SHORT_BREAKOUT" if direction == "SHORT" else "NEUTRAL_BREAKOUT"
    )
    return round(score, 1), signal, "; ".join(reasons)


def _mtf_confluence(
    dominant: Optional[str],
    h1_dir: Optional[str],
    h4_dir: Optional[str],
) -> Tuple[float, str]:
    """
    Score higher-timeframe confirmation of the M5 ``dominant`` direction.

    The M5 direction is the thing being confirmed — it is NOT a vote. It used to be
    passed in as one of three votes, so a setup whose higher timeframes were both
    NEUTRAL scored "FULL" (+30 pts) purely on its own say-so: 103 of 143 historical
    FULL rows had a NEUTRAL/absent higher timeframe, and 20 had *both* neutral.
    Those 30 free points are most of what let a mediocre setup clear the 70-point
    STRONG threshold, which is why the STRONG tier never outperformed.

    FULL now requires both H1 and H4 present and agreeing.
    """
    if dominant not in ("LONG", "SHORT"):
        return 0.0, "NONE"

    votes = [d for d in (h1_dir, h4_dir) if d in ("LONG", "SHORT")]
    if not votes:
        return 0.0, "UNCONFIRMED"

    agree = sum(1 for v in votes if v == dominant)
    oppose = len(votes) - agree
    if oppose:
        return 0.0, "OPPOSED" if agree == 0 else "CONFLICT"
    return (30.0, "FULL") if len(votes) == 2 else (15.0, "PARTIAL")


def _sr_proximity(
    close: Optional[float],
    atr14: Optional[float],
    sr_levels: list,
    dominant_direction: str,
) -> Tuple[float, str, bool, bool, Optional[float], Optional[float]]:
    """
    Score structure *relative to the trade direction*.

    A level only helps when it sits **behind** the trade (support beneath a long,
    resistance above a short) — that is where the stop shelters. A level directly
    **ahead** is a wall: it caps the move before the target can be reached.

    The previous version was direction-blind — both the `dist <= 0.3` branches
    awarded +25 and set at_key_level regardless of which way the trade pointed, so
    a long pinned under resistance scored identically to a long bouncing off
    support. Combined with the MTF bug this manufactured the STRONG tier: 10 of 11
    historical STRONG_BUY rows were at_key_level with an average sr_score of 24.1.

    Returns (score, reason, at_key_level, blocked_ahead, nearest_support, nearest_resistance).
    ``score`` may be negative when structure opposes the trade.
    """
    if not sr_levels or not close or not atr14 or atr14 <= 0:
        return 0.0, "", False, False, None, None

    supports = [lv["price"] for lv in sr_levels if lv["type"] == "S" and lv["price"] <= close]
    resistances = [lv["price"] for lv in sr_levels if lv["type"] == "R" and lv["price"] >= close]

    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    if dominant_direction not in ("LONG", "SHORT"):
        return 0.0, "", False, False, nearest_support, nearest_resistance

    if dominant_direction == "LONG":
        behind, ahead = nearest_support, nearest_resistance
        behind_label, ahead_label = "support", "resistance"
    else:
        behind, ahead = nearest_resistance, nearest_support
        behind_label, ahead_label = "resistance", "support"

    score = 0.0
    reasons: List[str] = []
    at_key_level = False
    blocked_ahead = False

    if behind is not None:
        dist = abs(close - behind) / atr14
        if dist <= 0.3:
            score += 25
            at_key_level = True
            reasons.append(f"AT {behind_label} {behind:.5f} (entry at structure)")
        elif dist <= 1.0:
            score += 15
            reasons.append(f"Near {behind_label} {behind:.5f}")

    if ahead is not None:
        dist = abs(ahead - close) / atr14
        # Target sits ~2.25×ATR out (1.5×RR on a 1.5×ATR stop), so a level inside
        # 1×ATR means the trade is very unlikely to reach target unimpeded.
        if dist <= 1.0:
            score -= 25
            blocked_ahead = True
            reasons.append(f"BLOCKED by {ahead_label} {ahead:.5f} ({dist:.1f}×ATR ahead)")
        elif dist <= 1.5:
            score -= 10
            reasons.append(f"{ahead_label.capitalize()} {ahead:.5f} close ahead ({dist:.1f}×ATR)")

    return (
        round(max(-25.0, min(score, 25.0)), 1),
        "; ".join(reasons),
        at_key_level,
        blocked_ahead,
        nearest_support,
        nearest_resistance,
    )


# Regime thresholds on ADX: above TREND → momentum playbook, below RANGE → reversion.
_ADX_TREND = 25.0
_ADX_RANGE = 18.0

# ATR multiples for suggested stop/target. Reward:risk stays fixed — the target is
# derived from the final stop distance, so widening the stop widens the target too.
_STOP_ATR_MULT = 1.5
_RR = 1.5
# Noise floor for the stop: 1×ATR on M5 is often just 2-4 pips, which sits inside
# ordinary spread noise and gets tagged within minutes. Never risk less than this.
_MIN_STOP_PIPS = 8.0

# Transaction cost is the dominant term at this timeframe. Measured over 910
# recorded outcomes: an average 2.23-pip spread against a 9.39-pip stop burned
# 0.237R per round trip — more than any plausible edge in the score. The stop must
# therefore scale with the spread so the cost ratio stays bounded.
_SPREAD_STOP_MULT = 8.0   # stop ≥ 8× spread ⇒ cost ≤ 12.5% of risk
_MAX_COST_RATIO = 0.15    # hard veto above this; nothing actionable survives it

# How far above cost-adjusted breakeven a modelled probability must sit before the
# trade is worth taking. Trading at exactly breakeven just donates the spread to
# the broker while adding variance, so demand a real cushion.
_PROB_MARGIN = 0.04


def breakeven_win_rate(rr: float = _RR, cost_ratio: float = 0.0) -> float:
    """
    Win rate at which a bracket exactly breaks even, including round-trip cost.

    expectancy = p·(rr − c) − (1 − p)·(1 + c) = 0  ⇒  p = (1 + c) / (1 + rr)

    with everything expressed in units of the stop distance. At rr=1.5 and zero
    cost this is exactly 0.400 — which is why an edgeless system lands on ~40%
    and why the observed 39.56% was indistinguishable from random entry.
    """
    return (1.0 + cost_ratio) / (1.0 + rr)


def _regime_weights(adx14: Optional[float]) -> Tuple[float, float, str]:
    """
    Decide how much to trust momentum vs mean-reversion given trend strength.
    Returns (momentum_weight, reversion_weight, regime_label). Blends linearly
    between the range/trend thresholds to avoid hard flip-flopping.
    """
    if adx14 is None:
        return 1.0, 1.0, "UNKNOWN"
    if adx14 >= _ADX_TREND:
        return 1.0, 0.0, "TREND"
    if adx14 <= _ADX_RANGE:
        return 0.0, 1.0, "RANGE"
    t = (adx14 - _ADX_RANGE) / (_ADX_TREND - _ADX_RANGE)
    return round(t, 3), round(1 - t, 3), "MIXED"


def _trade_levels(
    direction: str,
    entry: Optional[float],
    atr14: Optional[float],
    pair: str,
    spread_pips: Optional[float] = None,
) -> dict:
    """ATR-based stop/target/RR for an actionable direction. Empty dict if not computable."""
    if direction not in ("LONG", "SHORT") or not entry or not atr14 or atr14 <= 0:
        return {}
    pip = 0.01 if "JPY" in pair else 0.0001
    stop_dist = max(
        _STOP_ATR_MULT * atr14,
        _MIN_STOP_PIPS * pip,
        (spread_pips or 0.0) * _SPREAD_STOP_MULT * pip,
    )
    tgt_dist = _RR * stop_dist
    if direction == "LONG":
        stop = entry - stop_dist
        target = entry + tgt_dist
    else:
        stop = entry + stop_dist
        target = entry - tgt_dist
    return {
        "suggested_entry": round(entry, 6),
        "suggested_stop": round(stop, 6),
        "suggested_target": round(target, 6),
        "stop_pips": round(stop_dist / pip, 1),
        "target_pips": round(tgt_dist / pip, 1),
        "rr_ratio": round(_RR, 2),
    }


def score_pair(
    pair: str,
    bid: Optional[float],
    ask: Optional[float],
    spread_pips: Optional[float],
    indicators: dict,
    session: Optional[str] = None,
    max_spread_pips: float = 2.0,
    h1_direction: Optional[str] = None,
    h4_direction: Optional[str] = None,
    sr_levels: Optional[list] = None,
    model_prob: Optional[float] = None,
    strength_bonus: float = 0.0,
) -> dict:
    """
    Compute all signal scores and produce final trade_signal.
    Returns a dict merging into ForexSnapshot.

    ``model_prob`` is the trained model's P(target before stop) for this setup, when
    one is available. It acts as a veto, never as a promoter: the rule-based score
    still has to propose the setup, and the model decides whether the measured odds
    justify paying the spread.
    """
    close = indicators.get("close")
    rsi14 = indicators.get("rsi14")
    ema9 = indicators.get("ema9")
    ema20 = indicators.get("ema20")
    macd = indicators.get("macd")
    macd_hist = indicators.get("macd_histogram")
    atr14 = indicators.get("atr14")
    adx14 = indicators.get("adx14")
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    bb_middle = indicators.get("bb_middle")
    session_high = indicators.get("session_high")
    session_low = indicators.get("session_low")

    entry_px = round((bid + ask) / 2, 6) if bid and ask else close

    risk_notes = []
    if spread_pips is not None and spread_pips > max_spread_pips:
        risk_notes.append(f"Wide spread {spread_pips:.1f} pips (max {max_spread_pips})")

    mom_raw, mom_signal, mom_reason = _momentum(ema9, ema20, macd_hist, macd, rsi14, atr14)
    rev_raw, rev_signal, rev_reason = _mean_reversion(
        rsi14, close, bb_upper, bb_lower, bb_middle, session_high, session_low
    )
    sess_score, sess_signal, sess_reason = _session_breakout(
        close, session_high, session_low, atr14, session
    )

    # Regime gate: in trends trust momentum, in ranges trust reversion. These are
    # opposite playbooks — weighting (instead of summing both) stops them cancelling.
    w_mom, w_rev, regime = _regime_weights(adx14)
    mom_score = round(mom_raw * w_mom, 1)
    rev_score = round(rev_raw * w_rev, 1)

    # Weighted dominant direction — a suppressed playbook gets no vote.
    long_w = sum(sc for sig, sc in (
        (mom_signal, mom_score), (rev_signal, rev_score), (sess_signal, sess_score)
    ) if "LONG" in sig)
    short_w = sum(sc for sig, sc in (
        (mom_signal, mom_score), (rev_signal, rev_score), (sess_signal, sess_score)
    ) if "SHORT" in sig)
    dominant = "LONG" if long_w > short_w else (
        "SHORT" if short_w > long_w else "NEUTRAL"
    )

    # MTF confluence bonus (0-30 pts) — requires real higher-timeframe confirmation
    mtf_bonus, mtf_confluence = _mtf_confluence(dominant, h1_direction, h4_direction)

    # S/R structure, direction-aware (-25 to +25 pts)
    (
        sr_bonus, sr_reason, at_key_level, blocked_ahead,
        nearest_support, nearest_resistance,
    ) = _sr_proximity(close, atr14, sr_levels or [], dominant)

    # Currency-strength alignment is folded in here rather than bolted onto
    # total_score after the fact, so the score that drives the decision, the score
    # shown on the dashboard, and the score the model is trained on are the same number.
    total = mom_score + rev_score + sess_score + mtf_bonus + sr_bonus + strength_bonus

    # Penalize spread
    if spread_pips is not None and spread_pips > max_spread_pips:
        total = max(0, total - 20)

    # Cost ratio: spread as a fraction of the risk being taken. Computed from the
    # stop we would actually use, so it reflects the real drag on expectancy.
    provisional = _trade_levels(dominant, entry_px, atr14, pair, spread_pips)
    prov_stop_pips = provisional.get("stop_pips") or 0.0
    cost_ratio = round(spread_pips / prov_stop_pips, 4) if (
        spread_pips is not None and prov_stop_pips > 0
    ) else None
    cost_veto = cost_ratio is not None and cost_ratio > _MAX_COST_RATIO
    if cost_veto:
        risk_notes.append(
            f"Cost {cost_ratio:.0%} of risk (spread {spread_pips:.1f}p vs "
            f"{prov_stop_pips:.0f}p stop) — above {_MAX_COST_RATIO:.0%} limit"
        )
    ahead_block_label = "resistance" if dominant == "LONG" else "support"
    if blocked_ahead:
        risk_notes.append(f"Nearby {ahead_block_label} blocks the path to target")

    # H1 alignment gate: forward-tested outcomes showed candidates firing against the
    # H1 trend were the biggest loss bucket. An actionable signal must not fight H1.
    h1_opposes = (
        dominant in ("LONG", "SHORT")
        and h1_direction in ("LONG", "SHORT")
        and h1_direction != dominant
    )

    # Probability gate. When a trained model is available, an actionable signal must
    # clear the cost-adjusted breakeven win rate by a margin — this is the whole
    # point of the learning loop: the score proposes, the measured model disposes.
    be_p = breakeven_win_rate(_RR, cost_ratio or 0.0)
    required_p = round(be_p + _PROB_MARGIN, 4)
    prob_veto = model_prob is not None and model_prob < required_p

    if spread_pips is not None and spread_pips > max_spread_pips * 2:
        trade_signal = "AVOID"
        reason = f"Spread too wide ({spread_pips:.1f} pips)"
    elif cost_veto:
        trade_signal = "AVOID"
        reason = f"Cost {cost_ratio:.0%} of risk exceeds {_MAX_COST_RATIO:.0%} limit"
    elif h1_opposes and total >= 45:
        trade_signal = "WATCH_ONLY"
        reason = f"{dominant} setup ({total:.0f}pts) but H1 trend is {h1_direction} — countertrend"
    elif blocked_ahead and total >= 45:
        trade_signal = "WATCH_ONLY"
        reason = f"{dominant} setup ({total:.0f}pts) but {ahead_block_label} blocks the target"
    elif prob_veto and total >= 45:
        trade_signal = "WATCH_ONLY"
        reason = (
            f"{dominant} setup ({total:.0f}pts) but model P(win)={model_prob:.0%} "
            f"< {required_p:.0%} required"
        )
    elif total >= 70 and dominant == "LONG" and mtf_bonus >= 15:
        trade_signal = "STRONG_BUY"
        reason = f"Strong long setup ({total:.0f}pts, MTF:{mtf_confluence})"
    elif total >= 70 and dominant == "SHORT" and mtf_bonus >= 15:
        trade_signal = "STRONG_SHORT"
        reason = f"Strong short setup ({total:.0f}pts, MTF:{mtf_confluence})"
    elif total >= 45 and dominant == "LONG":
        trade_signal = "BUY_CANDIDATE"
        reason = f"Long candidate ({total:.0f}pts)"
    elif total >= 45 and dominant == "SHORT":
        trade_signal = "SHORT_CANDIDATE"
        reason = f"Short candidate ({total:.0f}pts)"
    elif total >= 25:
        trade_signal = "WATCH_ONLY"
        reason = f"Mixed signals ({total:.0f}pts)"
    else:
        trade_signal = "AVOID"
        reason = f"No clear setup ({total:.0f}pts)"

    if model_prob is not None and trade_signal not in ("AVOID", "WATCH_ONLY"):
        reason += f" — P(win) {model_prob:.0%} vs {required_p:.0%} needed"

    # ATR-based stop/target/RR for actionable directions
    entry = entry_px
    levels = _trade_levels(dominant, entry, atr14, pair, spread_pips) if trade_signal not in (
        "AVOID", "WATCH_ONLY"
    ) else {}

    signal_parts = []
    if mom_reason:
        signal_parts.append(f"Momentum: {mom_reason}")
    if rev_reason:
        signal_parts.append(f"Reversion: {rev_reason}")
    if sess_reason:
        signal_parts.append(f"Session: {sess_reason}")
    if mtf_confluence != "NONE":
        signal_parts.append(f"MTF: {mtf_confluence} ({h1_direction or '?'}/{h4_direction or '?'})")
    if sr_reason:
        signal_parts.append(f"S/R: {sr_reason}")

    return {
        "momentum_score": mom_score,
        "reversion_score": rev_score,
        "session_score": sess_score,
        "adx14": adx14,
        "regime": regime,
        "suggested_entry": levels.get("suggested_entry"),
        "suggested_stop": levels.get("suggested_stop"),
        "suggested_target": levels.get("suggested_target"),
        "stop_pips": levels.get("stop_pips"),
        "target_pips": levels.get("target_pips"),
        "rr_ratio": levels.get("rr_ratio"),
        "mtf_score": mtf_bonus,
        "mtf_confluence": mtf_confluence,
        "sr_score": sr_bonus,
        "at_key_level": at_key_level,
        "blocked_ahead": blocked_ahead,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "sr_levels_json": json.dumps(sr_levels[:5]) if sr_levels else None,
        "cost_ratio": cost_ratio,
        "model_prob": model_prob,
        "required_prob": required_p,
        "dominant": dominant,
        # Levels the trade *would* use, exposed even when the signal is not actionable
        # so feature extraction always sees a real stop size rather than a zero.
        "prov_stop_pips": provisional.get("stop_pips"),
        "prov_target_pips": provisional.get("target_pips"),
        "total_score": round(total, 1),
        "trade_signal": trade_signal,
        "signal_reason": reason,
        "risk_notes": "; ".join(risk_notes + signal_parts),
        "current_session": session,
    }
