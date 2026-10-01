from typing import List, Optional

MAJOR_PAIRS: List[str] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
]

MINOR_PAIRS: List[str] = [
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "EUR_CHF",
    "AUD_JPY", "EUR_AUD", "GBP_CHF",
]

EXOTIC_PAIRS: List[str] = [
    "USD_MXN", "USD_SGD", "USD_NOK", "USD_SEK",
    "USD_DKK", "USD_HKD",
]

# Pairs whose spread is small enough that a bracket can amortise it. Measured median
# spreads: all are <= 1.8 pips, against the 2.3-3.5 pips of everything excluded. At a
# ~27 pip M15 stop that is 5-7% of risk rather than 9-13%, which is the difference
# between a breakeven system and a losing one. Backtested over 9 months, restricting
# to these lifted expectancy from -0.002R (all 14 pairs) to +0.024R.
TIGHT_SPREAD_PAIRS: List[str] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "USD_CAD", "AUD_USD", "EUR_GBP",
]

UNIVERSE_MAP = {
    "Tight spread (recommended)": TIGHT_SPREAD_PAIRS,
    "Majors": MAJOR_PAIRS,
    "Majors + Minors": MAJOR_PAIRS + MINOR_PAIRS,
    "All": MAJOR_PAIRS + MINOR_PAIRS + EXOTIC_PAIRS,
}


def spread_to_pips(pair: str, spread: float) -> float:
    """Convert raw price spread to pip value."""
    if "JPY" in pair:
        return round(spread / 0.01, 1)
    return round(spread / 0.0001, 1)


def format_pair(pair: str) -> str:
    """Convert EUR_USD to EUR/USD for display."""
    return pair.replace("_", "/")


def tradingview_url(pair: str, interval: Optional[str] = None) -> str:
    """
    TradingView chart link for a pair, on OANDA's own feed.

    The display form (``EUR/USD``) rides along as the URL fragment: TradingView
    ignores it, and the dashboard's link columns read it back as the cell text,
    so a grid can show the pair while the cell itself is the link.
    """
    oanda = to_oanda_pair(pair)
    url = f"https://www.tradingview.com/chart/?symbol=OANDA%3A{oanda.replace('_', '')}"
    if interval:
        url += f"&interval={interval}"
    return f"{url}#{format_pair(oanda)}"


def pip_value(pair: str) -> float:
    """Price move worth one pip for this pair."""
    return 0.01 if "JPY" in pair.upper() else 0.0001


def to_oanda_pair(symbol: str) -> str:
    """
    Normalise an external pair symbol to OANDA's ``EUR_USD`` form.

    The daily ML repo stores pairs unseparated (``EURUSD``); other sources use
    ``EUR/USD`` or ``EUR-USD``. Anything that is not a recognisable 6-character
    pair is upper-cased and returned as-is rather than mangled, so an unexpected
    symbol shows up in the UI instead of silently matching the wrong instrument.
    """
    raw = (symbol or "").strip().upper()
    compact = raw.replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
    if len(compact) == 6 and compact.isalpha():
        return f"{compact[:3]}_{compact[3:]}"
    return raw
