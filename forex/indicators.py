from __future__ import annotations
from typing import List, Optional, Tuple


def _ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _ema_series(values: List[float], period: int) -> List[Optional[float]]:
    if len(values) < period:
        return [None] * len(values)
    k = 2.0 / (period + 1)
    result: List[Optional[float]] = [None] * (period - 1)
    ema = sum(values[:period]) / period
    result.append(ema)
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
        result.append(ema)
    return result


def calculate_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def calculate_ema(closes: List[float], period: int) -> Optional[float]:
    return _ema(closes, period)


def calculate_macd(
    closes: List[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < 26:
        return None, None, None
    fast = _ema_series(closes, 12)
    slow = _ema_series(closes, 26)
    macd_line = [
        (f - s) if f is not None and s is not None else None
        for f, s in zip(fast, slow)
    ]
    valid = [v for v in macd_line if v is not None]
    if len(valid) < 9:
        return None, None, None
    signal = _ema(valid, 9)
    last_macd = valid[-1]
    histogram = (last_macd - signal) if signal is not None else None
    return (
        round(last_macd, 6),
        round(signal, 6) if signal else None,
        round(histogram, 6) if histogram is not None else None,
    )


def calculate_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 6)


def calculate_bollinger_bands(
    closes: List[float],
    period: int = 20,
    std_mult: float = 2.0,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Returns (upper, middle, lower, width_pct)."""
    if len(closes) < period:
        return None, None, None, None
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((v - middle) ** 2 for v in window) / period
    std = variance ** 0.5
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    width_pct = ((upper - lower) / middle * 100) if middle else None
    return (
        round(upper, 6),
        round(middle, 6),
        round(lower, 6),
        round(width_pct, 2) if width_pct is not None else None,
    )


def calculate_session_hl(
    bars: List[dict],
    session_start_iso: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (session_high, session_low) for bars at or after session_start_iso."""
    session_bars = [b for b in bars if b.get("timestamp", "") >= session_start_iso]
    if not session_bars:
        return None, None
    highs = [b["high"] for b in session_bars]
    lows = [b["low"] for b in session_bars]
    return max(highs), min(lows)


def compute_all(bars: List[ForexBarDict]) -> dict:
    """
    Compute all indicators from a list of bar dicts.
    Returns a dict of indicator values ready to merge into ForexSnapshot.
    """
    if not bars:
        return {}
    closes = [b["close"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]

    rsi = calculate_rsi(closes)
    ema9 = calculate_ema(closes, 9)
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    macd, macd_sig, macd_hist = calculate_macd(closes)
    atr = calculate_atr(highs, lows, closes)
    bb_upper, bb_mid, bb_lower, bb_width = calculate_bollinger_bands(closes)

    last = bars[-1]
    prev_close = bars[-2]["close"] if len(bars) >= 2 else None
    day_change_pct = (
        round((last["close"] - prev_close) / prev_close * 100, 3)
        if prev_close
        else None
    )

    return {
        "open": last["open"],
        "high": last["high"],
        "low": last["low"],
        "close": last["close"],
        "day_change_pct": day_change_pct,
        "rsi14": rsi,
        "ema9": ema9,
        "ema20": ema20,
        "ema50": ema50,
        "macd": macd,
        "macd_signal": macd_sig,
        "macd_histogram": macd_hist,
        "atr14": atr,
        "bb_upper": bb_upper,
        "bb_middle": bb_mid,
        "bb_lower": bb_lower,
        "bb_width_pct": bb_width,
    }


# Type alias for readability
ForexBarDict = dict
