from typing import List

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

UNIVERSE_MAP = {
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
