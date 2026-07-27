"""
Timestamp parsing shared by the indicator, storage and feature layers.

OANDA emits RFC3339 with 9-digit nanosecond fractions ("...T08:00:00.000000000Z"),
which ``datetime.fromisoformat`` rejects — it accepts at most microseconds. Three
modules had grown their own near-miss version of this fix, each mishandling a
different edge (numeric UTC offsets, or bare SQLite ``CURRENT_TIMESTAMP`` strings).
A silent None here is expensive: it is what made trade durations unmeasurable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    """
    Parse an OANDA RFC3339 timestamp or a SQLite ``CURRENT_TIMESTAMP`` string into a
    timezone-aware UTC datetime. Returns None when the value cannot be parsed.

    Naive inputs are assumed UTC, which is correct for both sources here.
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        if "." in text:
            head, frac = text.split(".", 1)
            # Leading digits are the fraction; whatever follows is the zone suffix.
            i = 0
            while i < len(frac) and frac[i].isdigit():
                i += 1
            digits = frac[:i][:6].ljust(6, "0")
            rest = frac[i:]
            tz = "+00:00" if rest in ("", "Z", "z") else rest
            text = f"{head}.{digits}{tz}"
        elif text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def hour_utc(value: Optional[str]) -> Optional[float]:
    """Fractional UTC hour (0-24) from a timestamp, or None if unparseable."""
    dt = parse_ts(value)
    if dt is None:
        return None
    dt = dt.astimezone(timezone.utc)
    return dt.hour + dt.minute / 60.0
