from __future__ import annotations
from datetime import datetime, timezone
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


def calculate_adx(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> Optional[float]:
    """
    Wilder's ADX (trend-strength). Returns a value ~0-100 or None if insufficient bars.
    Rule of thumb: >25 trending, <18 ranging. Used to choose momentum vs reversion.
    """
    n = len(closes)
    if n < period * 2 + 1:
        return None

    plus_dm: List[float] = []
    minus_dm: List[float] = []
    trs: List[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    if len(trs) < period:
        return None

    # Wilder-smoothed running sums seeded with the first `period` values
    atr = sum(trs[:period])
    pdm = sum(plus_dm[:period])
    mdm = sum(minus_dm[:period])
    dxs: List[float] = []
    for i in range(period, len(trs)):
        atr = atr - atr / period + trs[i]
        pdm = pdm - pdm / period + plus_dm[i]
        mdm = mdm - mdm / period + minus_dm[i]
        if atr == 0:
            continue
        pdi = 100 * pdm / atr
        mdi = 100 * mdm / atr
        denom = pdi + mdi
        dx = 100 * abs(pdi - mdi) / denom if denom else 0.0
        dxs.append(dx)

    if not dxs:
        return None
    if len(dxs) < period:
        return round(sum(dxs) / len(dxs), 1)
    adx = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        adx = (adx * (period - 1) + dx) / period
    return round(adx, 1)


def _parse_oanda_ts(ts: str) -> Optional[datetime]:
    """Parse OANDA RFC3339 timestamps (which carry 9-digit nanosecond fractions)."""
    if not ts:
        return None
    try:
        if "." in ts:
            head, frac = ts.split(".", 1)
            frac = frac.rstrip("Z")[:6]  # fromisoformat handles <= microseconds
            iso = f"{head}.{frac}+00:00"
        else:
            iso = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return None


def session_high_low(
    bars: List[dict],
    start_dt: Optional[datetime],
) -> Tuple[Optional[float], Optional[float]]:
    """High/low of all bars whose timestamp is at or after ``start_dt`` (a tz-aware UTC datetime)."""
    if start_dt is None:
        return None, None
    selected = [
        b for b in bars
        if (ts := _parse_oanda_ts(b.get("timestamp", ""))) is not None and ts >= start_dt
    ]
    if not selected:
        return None, None
    return max(b["high"] for b in selected), min(b["low"] for b in selected)


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


def compute_trend_direction(bars: List[dict]) -> str:
    """
    Classify higher-timeframe trend direction (for H1/H4 bars).
    Returns 'LONG', 'SHORT', or 'NEUTRAL'.
    Uses 2-vote consensus: EMA9/EMA20 cross + MACD histogram sign.
    """
    if len(bars) < 26:
        return "NEUTRAL"
    closes = [b["close"] for b in bars]
    ema9 = calculate_ema(closes, 9)
    ema20 = calculate_ema(closes, 20)
    _, _, macd_hist = calculate_macd(closes)

    long_votes = 0
    short_votes = 0
    if ema9 is not None and ema20 is not None:
        if ema9 > ema20:
            long_votes += 1
        elif ema9 < ema20:
            short_votes += 1
    if macd_hist is not None:
        if macd_hist > 0:
            long_votes += 1
        elif macd_hist < 0:
            short_votes += 1

    if long_votes > short_votes:
        return "LONG"
    elif short_votes > long_votes:
        return "SHORT"
    return "NEUTRAL"


def detect_sr_levels(bars: List[dict], lookback: int = 50, n_pivot: int = 3) -> List[dict]:
    """
    Detect support/resistance levels using pivot high/low method.
    Returns list of {"price", "type", "touches", "strength"} sorted by strength desc.
    """
    if len(bars) < n_pivot * 2 + 1:
        return []
    recent = bars[-lookback:] if len(bars) > lookback else bars
    closes = [b["close"] for b in recent]
    highs = [b["high"] for b in recent]
    lows = [b["low"] for b in recent]

    atr = calculate_atr(highs, lows, closes, period=min(14, len(recent) - 1)) or 0.001
    tolerance = atr * 0.5

    pivot_highs: List[float] = []
    pivot_lows: List[float] = []
    n = len(recent)
    for i in range(n_pivot, n - n_pivot):
        lh = highs[i - n_pivot:i]
        rh = highs[i + 1:i + n_pivot + 1]
        if lh and rh and highs[i] > max(lh) and highs[i] > max(rh):
            pivot_highs.append(highs[i])
        ll = lows[i - n_pivot:i]
        rl = lows[i + 1:i + n_pivot + 1]
        if ll and rl and lows[i] < min(ll) and lows[i] < min(rl):
            pivot_lows.append(lows[i])

    def _cluster(prices: List[float], level_type: str) -> List[dict]:
        if not prices:
            return []
        sorted_prices = sorted(prices)
        clusters: List[List[float]] = [[sorted_prices[0]]]
        for p in sorted_prices[1:]:
            if p - clusters[-1][-1] <= tolerance:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        result = []
        for cluster in clusters:
            price = sum(cluster) / len(cluster)
            touches = sum(
                1 for b in recent
                if (level_type == "R" and abs(b["high"] - price) <= tolerance * 2)
                or (level_type == "S" and abs(b["low"] - price) <= tolerance * 2)
            )
            result.append({
                "price": round(price, 6),
                "type": level_type,
                "touches": touches,
                "strength": float(max(len(cluster), touches)),
            })
        return result

    levels = _cluster(pivot_highs, "R") + _cluster(pivot_lows, "S")
    levels.sort(key=lambda x: x["strength"], reverse=True)
    return levels[:8]


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
    adx = calculate_adx(highs, lows, closes)
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
        "adx14": adx,
        "bb_upper": bb_upper,
        "bb_middle": bb_mid,
        "bb_lower": bb_lower,
        "bb_width_pct": bb_width,
    }


# Type alias for readability
ForexBarDict = dict
