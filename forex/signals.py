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
    m5_dir: Optional[str],
    h1_dir: Optional[str],
    h4_dir: Optional[str],
) -> Tuple[float, str]:
    """
    Score multi-timeframe alignment.
    Full (all 3 agree): +30 pts, "FULL"
    Two of three agree: +15 pts, "PARTIAL"
    Conflict/missing: 0 pts, "NONE"
    """
    dirs = [d for d in (m5_dir, h1_dir, h4_dir) if d and d != "NEUTRAL"]
    if not dirs:
        return 0.0, "NONE"
    long_count = dirs.count("LONG")
    short_count = dirs.count("SHORT")
    total = len(dirs)
    if long_count == total or short_count == total:
        return 30.0, "FULL"
    elif long_count >= 2 or short_count >= 2:
        return 15.0, "PARTIAL"
    return 0.0, "NONE"


def _sr_proximity(
    close: Optional[float],
    atr14: Optional[float],
    sr_levels: list,
    dominant_direction: str,
) -> Tuple[float, str, bool, Optional[float], Optional[float]]:
    """
    Score proximity to key S/R levels.
    Returns (score, reason, at_key_level, nearest_support, nearest_resistance).
    """
    if not sr_levels or not close or not atr14 or atr14 == 0:
        return 0.0, "", False, None, None

    supports = [lv["price"] for lv in sr_levels if lv["type"] == "S" and lv["price"] <= close]
    resistances = [lv["price"] for lv in sr_levels if lv["type"] == "R" and lv["price"] >= close]

    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    score = 0.0
    reasons: List[str] = []
    at_key_level = False

    if nearest_support is not None:
        dist = (close - nearest_support) / atr14
        if dist <= 0.3:
            score += 25
            at_key_level = True
            reasons.append(f"AT support {nearest_support:.5f}")
        elif dist <= 1.0 and dominant_direction == "LONG":
            score += 15
            reasons.append(f"Near support {nearest_support:.5f}")

    if nearest_resistance is not None:
        dist = (nearest_resistance - close) / atr14
        if dist <= 0.3:
            score += 25
            at_key_level = True
            reasons.append(f"AT resistance {nearest_resistance:.5f}")
        elif dist <= 1.0 and dominant_direction == "SHORT":
            score += 15
            reasons.append(f"Near resistance {nearest_resistance:.5f}")

    return round(min(score, 25), 1), "; ".join(reasons), at_key_level, nearest_support, nearest_resistance


# Regime thresholds on ADX: above TREND → momentum playbook, below RANGE → reversion.
_ADX_TREND = 25.0
_ADX_RANGE = 18.0

# ATR multiples for suggested stop/target.
_STOP_ATR_MULT = 1.0
_TARGET_ATR_MULT = 1.5


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
) -> dict:
    """ATR-based stop/target/RR for an actionable direction. Empty dict if not computable."""
    if direction not in ("LONG", "SHORT") or not entry or not atr14 or atr14 <= 0:
        return {}
    pip = 0.01 if "JPY" in pair else 0.0001
    stop_dist = _STOP_ATR_MULT * atr14
    tgt_dist = _TARGET_ATR_MULT * atr14
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
        "rr_ratio": round(_TARGET_ATR_MULT / _STOP_ATR_MULT, 2),
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
) -> dict:
    """
    Compute all signal scores and produce final trade_signal.
    Returns a dict merging into ForexSnapshot.
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

    # MTF confluence bonus (0-30 pts)
    mtf_bonus, mtf_confluence = _mtf_confluence(dominant, h1_direction, h4_direction)

    # S/R proximity bonus (0-25 pts)
    sr_bonus, sr_reason, at_key_level, nearest_support, nearest_resistance = _sr_proximity(
        close, atr14, sr_levels or [], dominant
    )

    total = mom_score + rev_score + sess_score + mtf_bonus + sr_bonus

    # Penalize spread
    if spread_pips is not None and spread_pips > max_spread_pips:
        total = max(0, total - 20)

    if spread_pips is not None and spread_pips > max_spread_pips * 2:
        trade_signal = "AVOID"
        reason = f"Spread too wide ({spread_pips:.1f} pips)"
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

    # ATR-based stop/target/RR for actionable directions
    entry = round((bid + ask) / 2, 6) if bid and ask else close
    levels = _trade_levels(dominant, entry, atr14, pair) if trade_signal not in (
        "AVOID", "WATCH_ONLY"
    ) else {}
    if levels and spread_pips is not None and levels["target_pips"] < spread_pips * 3:
        risk_notes.append(
            f"Target {levels['target_pips']:.1f}p < 3× spread — thin edge"
        )

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
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "sr_levels_json": json.dumps(sr_levels[:5]) if sr_levels else None,
        "total_score": round(total, 1),
        "trade_signal": trade_signal,
        "signal_reason": reason,
        "risk_notes": "; ".join(risk_notes + signal_parts),
        "current_session": session,
    }
