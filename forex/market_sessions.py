from datetime import datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

US_EASTERN = ZoneInfo("America/New_York")

# Session windows in US Eastern time
# Asian spans midnight so handled specially
_SESSION_WINDOWS = {
    "London": (time(3, 0), time(12, 0)),
    "New_York": (time(8, 0), time(17, 0)),
    "London_NY_Overlap": (time(8, 0), time(12, 0)),
}

_ASIAN_START = time(17, 0)  # Sydney opens ~5pm ET (forex convention)
_ASIAN_END = time(4, 0)


def _now_eastern(now: datetime = None) -> datetime:
    t = now or datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(US_EASTERN)


def current_session(now: datetime = None) -> str:
    """Return the most specific active session name or 'Off_Hours'."""
    local = _now_eastern(now)
    t = local.time()

    # Weekend check: forex closes Fri 5pm ET, opens Sun 5pm ET
    if not is_forex_market_open(now):
        return "Off_Hours"

    # London/NY overlap is most specific — check first
    lo_start, lo_end = _SESSION_WINDOWS["London_NY_Overlap"]
    if lo_start <= t <= lo_end:
        return "London_NY_Overlap"

    # New York
    ny_start, ny_end = _SESSION_WINDOWS["New_York"]
    if ny_start <= t <= ny_end:
        return "New_York"

    # London
    lon_start, lon_end = _SESSION_WINDOWS["London"]
    if lon_start <= t <= lon_end:
        return "London"

    # Asian spans midnight
    if t >= _ASIAN_START or t <= _ASIAN_END:
        return "Asian"

    return "Off_Hours"


def current_session_start_utc(now: datetime = None) -> Optional[datetime]:
    """
    UTC datetime of the active session's open, for keying session high/low.
    Returns None during Off_Hours (no meaningful session range to break).
    """
    session = current_session(now)
    local = _now_eastern(now)

    if session in ("London", "London_NY_Overlap"):
        start_t = _SESSION_WINDOWS["London"][0]      # London open 03:00 ET
    elif session == "New_York":
        start_t = _SESSION_WINDOWS["New_York"][0]    # 08:00 ET
    elif session == "Asian":
        start_t = _ASIAN_START                       # 17:00 ET
    else:
        return None

    anchor = local.replace(
        hour=start_t.hour, minute=start_t.minute, second=0, microsecond=0
    )
    # Asian spans midnight: if we're in the early-morning tail, it opened yesterday evening.
    if session == "Asian" and local.time() <= _ASIAN_END:
        anchor -= timedelta(days=1)
    return anchor.astimezone(timezone.utc)


def is_forex_market_open(now: datetime = None) -> bool:
    """Forex is open Mon 5pm ET through Fri 5pm ET (approx)."""
    local = _now_eastern(now)
    weekday = local.weekday()  # 0=Mon ... 6=Sun
    t = local.time()

    if weekday == 5:  # Saturday — always closed
        return False
    if weekday == 6:  # Sunday — open after 5pm ET
        return t >= time(17, 0)
    if weekday == 4:  # Friday — closed after 5pm ET
        return t <= time(17, 0)
    return True  # Mon–Thu always open


def session_badge_color(session: str) -> str:
    """Return a color string for Streamlit display."""
    colors = {
        "London_NY_Overlap": "🟢",
        "New_York": "🔵",
        "London": "🟡",
        "Asian": "🟠",
        "Off_Hours": "⚫",
    }
    return colors.get(session, "⚫")
