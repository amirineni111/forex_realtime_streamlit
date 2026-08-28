"""
Mirror the daily ML predictions into SQLite, then score them at the NY close.

Two jobs, both deliberately stingy with external calls because the dashboard
re-runs every 60 seconds:

  1. ``sync_predictions`` — SQL Server → local SQLite. The source only changes
     once per weekday, so a sync is attempted only when the mirror is provably
     behind that day's run, and then at most once per throttle window. Failed
     attempts are logged too, otherwise an unreachable server would be dialled
     every single refresh.

  2. ``reconcile_session`` — score cached predictions for a completed New York
     session against OANDA daily candles. Already-scored (date, pair) rows are
     skipped, so the OANDA cost of a fully reconciled day is zero.

Why OANDA and not SQL Server for the actual outcome: the D1 candle closes at
17:00 ET, exactly the NY session close the reconcile is defined against, and it
is available the moment the session ends — whereas the SQL Server outcome
columns are backfilled a day or two later. Their closes agree to the last digit,
so the two feeds are interchangeable in accuracy but not in latency.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional
from zoneinfo import ZoneInfo

from .config import AppSettings
from .pairs import pip_value
from .sqlserver import MLSourceError, default_since, fetch_predictions
from .timeutil import parse_ts

US_EASTERN = ZoneInfo("America/New_York")

# The daily job runs at 20:55 ET and takes a few seconds; give it a margin before
# we consider that day's predictions "due".
DAILY_RUN_READY_ET = time(21, 15)

# OANDA D1 candles are aligned to 17:00 ET, which is the NY session close.
NY_SESSION_CLOSE_ET = time(17, 0)

SYNC_KIND_PREDICTIONS = "predictions"
SYNC_KIND_RECONCILE = "reconcile"

# A day's session can only be reconciled once the D1 candle exists. If it does
# not yet (broker lag, holiday), back off instead of retrying every refresh.
RECONCILE_RETRY_MINUTES = 15

# A pair whose newest daily candle predates the session by more than this is not
# merely late — the broker has stopped quoting it. OANDA still lists USD_INR but
# its last candle is from October 2022, while the daily model keeps predicting
# it; left as "pending" it would burn one request every retry window, forever.
# Such rows are recorded as NO_DATA so they stop being outstanding.
STALE_INSTRUMENT_DAYS = 5

OUTCOME_NO_DATA = "NO_DATA"


# ── Calendar helpers ────────────────────────────────────────────────────────

def _eastern(now: Optional[datetime] = None) -> datetime:
    t = now or datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(US_EASTERN)


def expected_prediction_date(now: Optional[datetime] = None) -> date:
    """
    The most recent weekday whose EOD prediction run should have finished.

    This is the mirror's target: if the newest cached ``prediction_date`` equals
    this, there is nothing on SQL Server we do not already have, and we make no
    connection at all.
    """
    local = _eastern(now)
    candidate = local.date()
    if local.time() < DAILY_RUN_READY_ET:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:  # Sat/Sun — no run
        candidate -= timedelta(days=1)
    return candidate


def last_completed_session_date(now: Optional[datetime] = None) -> date:
    """
    Trading date of the most recently closed New York session.

    Forex trading days run 17:00 ET → 17:00 ET and are labelled by the day they
    close on, which is also how OANDA labels its D1 candles.
    """
    local = _eastern(now)
    candidate = local.date()
    if local.time() < NY_SESSION_CLOSE_ET:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:  # Sat/Sun close on Friday's session
        candidate -= timedelta(days=1)
    return candidate


def candle_trading_date(timestamp: str) -> Optional[date]:
    """
    Map an OANDA D1 candle start to the trading date it closes on.

    The candle starting 2026-07-29T21:00Z is 17:00 ET on the 29th and closes at
    17:00 ET on the 30th, so it is the 30th's session. Converting to Eastern
    first keeps this correct across DST, when the same candle starts at 22:00Z.
    """
    parsed = parse_ts(timestamp)
    if parsed is None:
        return None
    return parsed.astimezone(US_EASTERN).date() + timedelta(days=1)


# ── Prediction sync ─────────────────────────────────────────────────────────

def sync_due(
    storage,
    settings: AppSettings,
    now: Optional[datetime] = None,
) -> tuple:
    """
    Decide whether to reach out to SQL Server. Returns ``(due, reason)``.

    Not due is the common case and costs nothing — no socket, no driver load.
    """
    if not settings.ml_source_configured:
        return False, "SQL Server is not configured"

    cached_max = storage.load_ml_max_prediction_date()
    expected = expected_prediction_date(now).isoformat()
    if cached_max and cached_max >= expected:
        return False, f"up to date ({cached_max})"

    last_attempt = storage.load_ml_last_sync(kind=SYNC_KIND_PREDICTIONS)
    if last_attempt:
        attempted = parse_ts(last_attempt.get("attempted_at"))
        if attempted is not None:
            reference = now or datetime.now(timezone.utc)
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=timezone.utc)
            age_minutes = (reference - attempted).total_seconds() / 60
            if age_minutes < settings.ml_sync_min_interval_minutes:
                wait = settings.ml_sync_min_interval_minutes - age_minutes
                return False, f"throttled, retry in {wait:.0f} min"

    return True, f"mirror behind (have {cached_max or 'nothing'}, expect {expected})"


def sync_predictions(
    storage,
    settings: AppSettings,
    force: bool = False,
    now: Optional[datetime] = None,
) -> dict:
    """
    Pull new prediction rows into SQLite if the mirror is behind (or ``force``).

    Always returns a status dict rather than raising: a dashboard refresh must
    not die because a database on another machine is asleep.
    """
    if force and not settings.ml_source_configured:
        return {"status": "skipped", "reason": "SQL Server is not configured", "rows": 0}

    if not force:
        due, reason = sync_due(storage, settings, now=now)
        if not due:
            return {"status": "skipped", "reason": reason, "rows": 0}

    cached_max = storage.load_ml_max_prediction_date()
    since = default_since(settings, cached_max)
    try:
        rows = fetch_predictions(settings, since)
    except MLSourceError as exc:
        storage.log_ml_sync(SYNC_KIND_PREDICTIONS, "error", 0, None, str(exc))
        return {"status": "error", "reason": str(exc), "rows": 0}

    saved = storage.save_ml_predictions(rows)
    new_max = storage.load_ml_max_prediction_date()
    storage.log_ml_sync(
        SYNC_KIND_PREDICTIONS, "ok", saved, new_max,
        f"since={since.isoformat()}",
    )
    return {
        "status": "ok",
        "reason": f"synced from {since.isoformat()}",
        "rows": saved,
        "max_prediction_date": new_max,
    }


# ── Reconciliation ──────────────────────────────────────────────────────────

def implied_direction(prob_buy: Optional[float], prob_sell: Optional[float]) -> Optional[str]:
    """
    The model's directional lean, independent of the abstain gate.

    HOLD rows are abstentions, not neutral forecasts — roughly 70% of the daily
    output is HOLD, so scoring only BUY/SELL would leave most days unmeasured.
    The underlying binary probabilities still carry a direction and are what make
    those rows scoreable.
    """
    if prob_buy is None or prob_sell is None:
        return None
    if prob_buy > prob_sell:
        return "UP"
    if prob_sell > prob_buy:
        return "DOWN"
    return None


def score_prediction(prediction: dict, base_close: float, actual_close: float) -> dict:
    """
    Score one cached prediction against the session's realised close.

    ``base_close`` is the previous session's close and ``actual_close`` the
    target session's, both from OANDA — the same close-to-close basis the daily
    model was trained on.
    """
    pair = prediction["pair"]
    signal = (prediction.get("predicted_signal") or "").upper()
    ret_pct = ((actual_close - base_close) / base_close * 100) if base_close else 0.0
    move_pips = (actual_close - base_close) / pip_value(pair)

    if actual_close > base_close:
        actual = "UP"
    elif actual_close < base_close:
        actual = "DOWN"
    else:
        actual = "FLAT"

    lean = implied_direction(prediction.get("prob_buy"), prediction.get("prob_sell"))

    if signal == "BUY":
        signal_outcome = "HIT" if actual == "UP" else "MISS"
    elif signal == "SELL":
        signal_outcome = "HIT" if actual == "DOWN" else "MISS"
    else:
        # HOLD is the gate abstaining; there was no call to be right or wrong about.
        signal_outcome = "ABSTAIN"

    if lean is None or actual == "FLAT":
        lean_outcome = "N/A"
    else:
        lean_outcome = "HIT" if lean == actual else "MISS"

    return {
        "target_date": prediction["target_date"],
        "pair": pair,
        "prediction_date": prediction["prediction_date"],
        "predicted_signal": signal or None,
        "implied_direction": lean,
        "signal_confidence": prediction.get("signal_confidence"),
        "prob_buy": prediction.get("prob_buy"),
        "prob_sell": prediction.get("prob_sell"),
        "sql_base_close": prediction.get("base_close"),
        "base_close": base_close,
        "actual_close": actual_close,
        "actual_return_pct": round(ret_pct, 4),
        "actual_pips": round(move_pips, 1),
        "actual_direction": actual,
        "signal_outcome": signal_outcome,
        "implied_outcome": lean_outcome,
    }


def _as_float(value) -> Optional[float]:
    """Coerce to float, treating None, NaN and non-numerics alike as missing."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def live_progress(prediction: dict, current_price) -> dict:
    """
    Where an open session currently sits against the ML call.

    This is the same base-to-close comparison ``score_prediction`` will make at
    the 17:00 ET close, with the live price standing in for the close that has
    not been published yet — so the verdict is a running preview of the row the
    reconcile will write, not a second, differently-defined metric. It stays a
    preview: until the session closes the direction can still flip.

    ``base_close`` is the source close the model itself predicted from, which is
    also what makes the two comparable; broker and source closes agree to the
    last digit (see module docstring).
    """
    base = _as_float(prediction.get("base_close"))
    price = _as_float(current_price)
    if base is None or price is None or base == 0:
        return {
            "current_price": price, "current_direction": None,
            "move_pips": None, "move_pct": None,
            "signal_status": None, "lean_status": None,
        }

    scored = score_prediction(prediction, base, price)
    return {
        "current_price": price,
        "current_direction": scored["actual_direction"],
        "move_pips": scored["actual_pips"],
        "move_pct": scored["actual_return_pct"],
        "signal_status": scored["signal_outcome"],
        "lean_status": scored["implied_outcome"],
    }


def unscoreable_row(prediction: dict, reason: str) -> dict:
    """
    A placeholder for a prediction that can never be scored against this broker.

    Written rather than skipped so the pair stops counting as outstanding. It is
    excluded from every hit-rate calculation — an instrument with no price data
    must not be silently folded in as a miss.
    """
    return {
        "target_date": prediction["target_date"],
        "pair": prediction["pair"],
        "prediction_date": prediction["prediction_date"],
        "predicted_signal": (prediction.get("predicted_signal") or "").upper() or None,
        "implied_direction": implied_direction(
            prediction.get("prob_buy"), prediction.get("prob_sell")
        ),
        "signal_confidence": prediction.get("signal_confidence"),
        "prob_buy": prediction.get("prob_buy"),
        "prob_sell": prediction.get("prob_sell"),
        "sql_base_close": prediction.get("base_close"),
        "base_close": None,
        "actual_close": None,
        "actual_return_pct": None,
        "actual_pips": None,
        "actual_direction": None,
        "signal_outcome": OUTCOME_NO_DATA,
        "implied_outcome": reason,
    }


def _closes_by_trading_date(bars: List) -> dict:
    out = {}
    for bar in bars:
        trading_date = candle_trading_date(getattr(bar, "timestamp", None))
        if trading_date is not None:
            out[trading_date] = float(bar.close)
    return out


def _resolve_closes(closes: dict, target_date: date) -> tuple:
    """
    Pick the session close and its base from a pair's candle series.

    Returns ``(base_close, actual_close, state)`` where state is ``ok``,
    ``pending`` (candle should arrive, try again later) or ``stale`` (the broker
    has stopped quoting this pair — waiting will not help).
    """
    if not closes:
        return None, None, "stale"

    actual_close = closes.get(target_date)
    if actual_close is None:
        if (target_date - max(closes)).days > STALE_INSTRUMENT_DAYS:
            return None, None, "stale"
        return None, None, "pending"

    # Previous *available* session close — steps over weekends and holidays
    # rather than assuming target_date - 1 exists.
    prior_dates = sorted(d for d in closes if d < target_date)
    if not prior_dates:
        return None, None, "pending"  # base close falls outside the fetched window
    return closes[prior_dates[-1]], actual_close, "ok"


def reconcile_session(
    storage,
    client,
    target_date: Optional[date] = None,
    force: bool = False,
    now: Optional[datetime] = None,
    candle_count: Optional[int] = None,
) -> dict:
    """
    Score every cached prediction for ``target_date`` against OANDA D1 closes.

    Pairs already scored for that date are skipped, so calling this on every
    dashboard refresh settles to zero OANDA calls once the session is fully
    reconciled. Pairs whose candle has not appeared yet are simply left for the
    next attempt.
    """
    target_date = target_date or last_completed_session_date(now)
    target_iso = target_date.isoformat()

    # OANDA returns the *most recent* ``count`` candles, so an older session
    # needs a window wide enough to still contain it — a fixed count would make
    # re-evaluating any past date silently return nothing.
    if candle_count is None:
        lookback_days = max(0, (last_completed_session_date(now) - target_date).days)
        candle_count = min(500, 12 + int(lookback_days * 5 / 7))

    predictions = storage.load_ml_predictions_for_target(target_iso)
    if not predictions:
        return {
            "status": "skipped",
            "reason": f"no cached predictions targeting {target_iso}",
            "target_date": target_iso, "scored": 0, "pending": 0,
        }

    if not force:
        already = storage.load_ml_reconciled_pairs(target_iso)
        predictions = [p for p in predictions if p["pair"] not in already]
        if not predictions:
            return {
                "status": "up_to_date",
                "reason": f"{target_iso} already reconciled",
                "target_date": target_iso, "scored": 0, "pending": 0,
            }

    scored, unscoreable, pending, errors = [], [], [], []
    for prediction in predictions:
        pair = prediction["pair"]
        try:
            bars = client.get_candles(pair, granularity="D", count=candle_count)
        except Exception as exc:
            errors.append(f"{pair}: {exc}")
            continue

        base_close, actual_close, state = _resolve_closes(
            _closes_by_trading_date(bars), target_date
        )
        if state == "ok":
            scored.append(score_prediction(prediction, base_close, actual_close))
        elif state == "stale":
            unscoreable.append(unscoreable_row(prediction, "no broker data"))
        else:
            pending.append(pair)  # candle has not been published yet

    written = scored + unscoreable
    if written:
        storage.save_ml_reconciliation(written)

    message_bits = []
    if unscoreable:
        message_bits.append(
            f"no broker data: {', '.join(r['pair'] for r in unscoreable)}"
        )
    if pending:
        message_bits.append(f"pending: {', '.join(pending)}")
    if errors:
        message_bits.append(f"errors: {'; '.join(errors)}")
    message = " | ".join([f"scored {len(scored)}"] + message_bits)

    if scored or unscoreable:
        status = "ok"
    elif errors and not pending:
        status = "error"
    else:
        status = "pending"
    storage.log_ml_sync(SYNC_KIND_RECONCILE, status, len(scored), target_iso, message)

    return {
        "status": status,
        "reason": message,
        "target_date": target_iso,
        "scored": len(scored),
        "unscoreable": len(unscoreable),
        "pending": len(pending),
        "errors": errors,
    }


def backfill_reconciliation(
    storage,
    client,
    days: int = 60,
    now: Optional[datetime] = None,
    force: bool = False,
) -> dict:
    """
    Score every unscored past session in one pass.

    Deliberately *not* a loop over ``reconcile_session``: that would cost one
    OANDA request per (session, pair) — ~840 for a 60-day window. Here each pair
    is fetched once with a window wide enough to cover every session, so the cost
    is one request per pair regardless of how far back the backfill reaches.
    """
    cutoff = last_completed_session_date(now)
    earliest = cutoff - timedelta(days=days)

    targets = [
        d for d in storage.load_ml_target_dates(limit=days * 2)
        if d and earliest.isoformat() <= d <= cutoff.isoformat()
    ]
    if not targets:
        return {"status": "skipped", "reason": "no past sessions to score", "scored": 0}

    by_pair: dict = {}
    for target_iso in targets:
        already = set() if force else storage.load_ml_reconciled_pairs(target_iso)
        for prediction in storage.load_ml_predictions_for_target(target_iso):
            if prediction["pair"] in already:
                continue
            by_pair.setdefault(prediction["pair"], []).append(prediction)

    if not by_pair:
        return {"status": "up_to_date", "reason": "all sessions already scored", "scored": 0}

    # Weekdays in the window, plus headroom for the base close and holidays.
    candle_count = min(500, int(days * 5 / 7) + 10)

    scored, unscoreable, errors, pending = [], [], [], 0
    for pair, predictions in by_pair.items():
        try:
            bars = client.get_candles(pair, granularity="D", count=candle_count)
        except Exception as exc:
            errors.append(f"{pair}: {exc}")
            continue
        closes = _closes_by_trading_date(bars)
        for prediction in predictions:
            try:
                target = date.fromisoformat(prediction["target_date"])
            except (TypeError, ValueError):
                continue
            base_close, actual_close, state = _resolve_closes(closes, target)
            if state == "ok":
                scored.append(score_prediction(prediction, base_close, actual_close))
            elif state == "stale":
                unscoreable.append(unscoreable_row(prediction, "no broker data"))
            else:
                pending += 1

    written = scored + unscoreable
    if written:
        storage.save_ml_reconciliation(written)

    message = f"scored {len(scored)} across {len(targets)} session(s)"
    if unscoreable:
        message += f", {len(unscoreable)} with no broker data"
    if pending:
        message += f", {pending} awaiting candles"
    if errors:
        message += f" | errors: {'; '.join(errors)}"
    status = "ok" if written else ("error" if errors else "pending")
    storage.log_ml_sync(SYNC_KIND_RECONCILE, status, len(scored), cutoff.isoformat(), message)

    return {
        "status": status, "reason": message, "scored": len(scored),
        "sessions": len(targets), "unscoreable": len(unscoreable),
        "pending": pending, "errors": errors,
    }


def reconcile_due(
    storage,
    target_date: date,
    now: Optional[datetime] = None,
) -> tuple:
    """
    Whether an automatic reconcile attempt is worth making. Returns ``(due, reason)``.

    Guards the auto-refresh path only; the tab's explicit button bypasses it.
    """
    target_iso = target_date.isoformat()
    predictions = storage.load_ml_predictions_for_target(target_iso)
    if not predictions:
        return False, f"no predictions targeting {target_iso}"

    outstanding = {p["pair"] for p in predictions} - storage.load_ml_reconciled_pairs(target_iso)
    if not outstanding:
        return False, f"{target_iso} fully reconciled"

    last = storage.load_ml_last_sync(kind=SYNC_KIND_RECONCILE)
    if last:
        attempted = parse_ts(last.get("attempted_at"))
        if attempted is not None:
            reference = now or datetime.now(timezone.utc)
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=timezone.utc)
            age_minutes = (reference - attempted).total_seconds() / 60
            if age_minutes < RECONCILE_RETRY_MINUTES:
                return False, f"throttled, retry in {RECONCILE_RETRY_MINUTES - age_minutes:.0f} min"

    return True, f"{len(outstanding)} pair(s) outstanding for {target_iso}"
