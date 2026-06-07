from __future__ import annotations
from typing import Dict, List, Optional, Tuple

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]

# (base_currency, quote_currency)
# When day_change_pct > 0: base strengthened, quote weakened
PAIR_CURRENCY_MAP: Dict[str, Tuple[str, str]] = {
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "USD_JPY": ("USD", "JPY"),
    "USD_CHF": ("USD", "CHF"),
    "AUD_USD": ("AUD", "USD"),
    "USD_CAD": ("USD", "CAD"),
    "NZD_USD": ("NZD", "USD"),
    "EUR_GBP": ("EUR", "GBP"),
    "EUR_JPY": ("EUR", "JPY"),
    "GBP_JPY": ("GBP", "JPY"),
    "EUR_CHF": ("EUR", "CHF"),
    "AUD_JPY": ("AUD", "JPY"),
    "EUR_AUD": ("EUR", "AUD"),
    "GBP_CHF": ("GBP", "CHF"),
}


def calculate_strength(snapshots: List[dict]) -> Dict[str, float]:
    """
    Compute relative currency strength from snapshot day_change_pct values.
    Returns {currency: strength_score} normalized 0-100.
    """
    raw: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
    counts: Dict[str, int] = {c: 0 for c in CURRENCIES}

    for snap in snapshots:
        pair = snap.get("pair", "")
        pct = snap.get("day_change_pct")
        if pct is None or pair not in PAIR_CURRENCY_MAP:
            continue
        base, quote = PAIR_CURRENCY_MAP[pair]
        raw[base] += pct
        counts[base] = counts.get(base, 0) + 1
        raw[quote] -= pct
        counts[quote] = counts.get(quote, 0) + 1

    averaged: Dict[str, float] = {}
    for c in CURRENCIES:
        if counts.get(c, 0) > 0:
            averaged[c] = raw[c] / counts[c]
        else:
            averaged[c] = 0.0

    values = list(averaged.values())
    min_v, max_v = min(values), max(values)
    rng = max_v - min_v
    if rng > 0:
        return {c: round((v - min_v) / rng * 100, 1) for c, v in averaged.items()}
    return {c: 50.0 for c in CURRENCIES}


def get_strength_for_pair(
    pair: str,
    strength_scores: Dict[str, float],
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Return (base_strength, quote_strength, assessment) for a pair.
    assessment: "STRONG_BASE_WEAK_QUOTE" | "WEAK_BASE_STRONG_QUOTE" | "NEUTRAL"
    """
    if pair not in PAIR_CURRENCY_MAP:
        return None, None, "NEUTRAL"

    base, quote = PAIR_CURRENCY_MAP[pair]
    base_str = strength_scores.get(base)
    quote_str = strength_scores.get(quote)

    if base_str is None or quote_str is None:
        return base_str, quote_str, "NEUTRAL"

    diff = base_str - quote_str
    if diff >= 30:
        assessment = "STRONG_BASE_WEAK_QUOTE"
    elif diff <= -30:
        assessment = "WEAK_BASE_STRONG_QUOTE"
    else:
        assessment = "NEUTRAL"

    return base_str, quote_str, assessment


def strength_bonus(assessment: str, trade_signal: str) -> float:
    """
    Returns score adjustment based on currency strength alignment with signal direction.
    Aligned: +10 pts. Conflicting: -5 pts. Neutral: 0.
    """
    is_long = trade_signal in ("STRONG_BUY", "BUY_CANDIDATE")
    is_short = trade_signal in ("STRONG_SHORT", "SHORT_CANDIDATE")

    if assessment == "STRONG_BASE_WEAK_QUOTE" and is_long:
        return 10.0
    elif assessment == "WEAK_BASE_STRONG_QUOTE" and is_short:
        return 10.0
    elif assessment == "STRONG_BASE_WEAK_QUOTE" and is_short:
        return -5.0
    elif assessment == "WEAK_BASE_STRONG_QUOTE" and is_long:
        return -5.0
    return 0.0
