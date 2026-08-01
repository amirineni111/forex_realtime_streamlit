"""
Read-only reader for the daily ML prediction table in SQL Server.

The predictions are produced once per weekday (~20:55 ET) by the
``sqlserver_copilot_forex`` repo and written to ``forex_ml_predictions`` in
``stockdata_db``. This module is the *only* place that talks to SQL Server; the
scanner reads everything else from the local SQLite mirror, which is what keeps
a 1-minute auto-refresh from hammering a database that changes once a day.

Column semantics of the source table (verified against live rows):

  prediction_date  the EOD run timestamp — i.e. the close the model predicted FROM
  date_time        the target trading date the prediction is FOR (next business day)
  close_price      the closing price at ``prediction_date`` that the model saw
  currency_pair    unseparated, e.g. ``EURUSD``
  predicted_signal BUY / SELL / HOLD, where HOLD means the confidence gate
                   ABSTAINED rather than "no move expected"
  prob_buy/sell    the underlying binary probabilities, still meaningful on HOLD
                   rows — this is the only directional read available for them
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List, Optional

from .config import AppSettings
from .pairs import to_oanda_pair

# Only identifiers we build ourselves are interpolated into SQL; the table name
# comes from config and is validated here so it can never carry an injection.
_SAFE_TABLE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.[]")


class MLSourceError(RuntimeError):
    """Raised when the SQL Server prediction source cannot be read."""


def _validated_table(name: str) -> str:
    if not name or any(ch not in _SAFE_TABLE_CHARS for ch in name):
        raise MLSourceError(f"Unsafe ML predictions table name: {name!r}")
    return name


def _connection_string(settings: AppSettings) -> str:
    parts = [
        f"DRIVER={{{settings.sql_driver}}}",
        f"SERVER={settings.sql_server}",
        f"DATABASE={settings.sql_database}",
    ]
    if settings.sql_trusted_connection:
        parts.append("Trusted_Connection=yes")
    else:
        parts.append(f"UID={settings.sql_username}")
        parts.append(f"PWD={settings.sql_password}")
    return ";".join(parts) + ";"


def _connect(settings: AppSettings):
    try:
        import pyodbc  # imported lazily: the dashboard runs fine without the driver
    except ImportError as exc:  # pragma: no cover - depends on host install
        raise MLSourceError(
            "pyodbc is not installed. Run: pip install pyodbc "
            "(and install the Microsoft ODBC Driver for SQL Server)."
        ) from exc

    if not settings.ml_source_configured:
        raise MLSourceError(
            "SQL Server is not configured. Set SQL_SERVER / SQL_DATABASE and either "
            "SQL_TRUSTED_CONNECTION=yes or SQL_USERNAME / SQL_PASSWORD in .env."
        )
    try:
        return pyodbc.connect(
            _connection_string(settings), timeout=settings.sql_login_timeout_seconds
        )
    except Exception as exc:
        raise MLSourceError(f"Could not connect to SQL Server: {exc}") from exc


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_iso_date(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def fetch_predictions(settings: AppSettings, since: date) -> List[dict]:
    """
    Fetch prediction rows with ``prediction_date >= since``, newest run wins.

    Returns rows already normalised to this app's vocabulary: OANDA pair codes,
    ISO date strings, plain floats. The source occasionally holds more than one
    row per (date, pair) when the daily job is re-run, so rows are de-duplicated
    on ``created_at``.
    """
    table = _validated_table(settings.ml_predictions_table)
    sql = f"""
        SELECT CAST(prediction_date AS DATE) AS prediction_date,
               currency_pair,
               CAST(date_time AS DATE)       AS target_date,
               predicted_signal,
               signal_confidence,
               prob_buy,
               prob_sell,
               prob_hold,
               close_price,
               model_name,
               model_version,
               created_at
        FROM {table}
        WHERE prediction_date >= ?
        ORDER BY prediction_date, currency_pair, created_at
    """

    conn = _connect(settings)
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, since)
            raw = cursor.fetchall()
        finally:
            cursor.close()
    except MLSourceError:
        raise
    except Exception as exc:
        raise MLSourceError(f"Query against {table} failed: {exc}") from exc
    finally:
        conn.close()

    # ORDER BY created_at means a later duplicate simply overwrites the earlier one.
    deduped: dict = {}
    for row in raw:
        pair = to_oanda_pair(row[1])
        prediction_date = _as_iso_date(row[0])
        deduped[(prediction_date, pair)] = {
            "prediction_date": prediction_date,
            "pair": pair,
            "sql_pair": (row[1] or "").strip(),
            "target_date": _as_iso_date(row[2]),
            "predicted_signal": (row[3] or "").strip().upper() or None,
            "signal_confidence": _as_float(row[4]),
            "prob_buy": _as_float(row[5]),
            "prob_sell": _as_float(row[6]),
            "prob_hold": _as_float(row[7]),
            "base_close": _as_float(row[8]),
            "model_name": row[9],
            "model_version": row[10],
            "source_created_at": row[11].isoformat() if isinstance(row[11], datetime) else row[11],
        }
    return sorted(deduped.values(), key=lambda r: (r["prediction_date"], r["pair"]))


def test_connection(settings: AppSettings) -> dict:
    """Probe the source and report what is there. Used by the Settings tab."""
    table = _validated_table(settings.ml_predictions_table)
    conn = _connect(settings)
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"SELECT MAX(CAST(prediction_date AS DATE)), COUNT(*) FROM {table}"
            )
            max_date, total = cursor.fetchone()
        finally:
            cursor.close()
    except MLSourceError:
        raise
    except Exception as exc:
        raise MLSourceError(f"Query against {table} failed: {exc}") from exc
    finally:
        conn.close()
    return {
        "server": settings.sql_server,
        "database": settings.sql_database,
        "table": table,
        "max_prediction_date": _as_iso_date(max_date),
        "total_rows": int(total or 0),
    }


def default_since(settings: AppSettings, cached_max: Optional[str]) -> date:
    """
    Lower bound for the next fetch.

    First sync pulls a history window so the Reconcile tab has something to score
    immediately; afterwards it pulls only what the mirror is missing. The one-day
    overlap re-reads the newest cached day, which costs ~14 rows and covers the
    case where the daily job re-ran after our last sync.
    """
    if not cached_max:
        return date.today() - timedelta(days=settings.ml_initial_sync_days)
    try:
        return date.fromisoformat(cached_max) - timedelta(days=1)
    except ValueError:
        return date.today() - timedelta(days=settings.ml_initial_sync_days)
