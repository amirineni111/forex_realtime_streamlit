from __future__ import annotations
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from forex.config import AppSettings, get_settings
from forex.features import FEATURE_VERSION
from forex.market_sessions import current_session, is_forex_market_open, session_badge_color
from forex.ml_sync import (
    backfill_reconciliation, expected_prediction_date, implied_direction,
    last_completed_session_date, live_progress, reconcile_due, reconcile_session,
    sync_due, sync_predictions,
)
from forex.alerts import make_sink
from forex.model import MIN_TRAIN_SAMPLES
from forex.models import ScanRequest
from forex.oanda import OandaClient
from forex.pairs import UNIVERSE_MAP, format_pair
from forex.scanner import run_scan
from forex.storage import Storage
from forex.strength import calculate_strength, CURRENCIES
from forex.training import evaluate_and_fit, gate_summary, save_candidate

st.set_page_config(
    page_title="Forex Trading Dashboard",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Persistence helpers ──────────────────────────────────────────────────────

PREFS_PATH = Path("data/app_preferences.json")


def _load_prefs() -> dict:
    if PREFS_PATH.exists():
        try:
            return json.loads(PREFS_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_prefs(d: dict) -> None:
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFS_PATH.write_text(json.dumps(d, indent=2))


# ── Grid sizing ──────────────────────────────────────────────────────────────

# Streamlit's grid draws a ~38px header and ~35px per row, and defaults to a
# height that fits about ten rows — anything longer gets an inner scrollbar.
# Sizing the widget to the data removes that scrollbar; the sidebar setting caps
# how far a grid may grow before it is allowed to scroll again.
GRID_HEADER_PX = 38
GRID_ROW_PX = 35
DEFAULT_TABLE_ROWS = 20   # covers every built-in universe (All = 20 pairs)
MIN_TABLE_ROWS = 5
MAX_TABLE_ROWS = 50


def _grid_height(row_count: int, max_rows: int | None = None) -> int:
    """
    Pixel height that shows ``row_count`` rows without an inner scrollbar.

    Grows only to the data's own length, so a short table does not leave a band
    of empty grid below it; past the configured cap the grid scrolls as before.
    """
    limit = max_rows or st.session_state.get("table_rows", DEFAULT_TABLE_ROWS)
    visible = max(1, min(int(row_count or 1), int(limit)))
    return GRID_HEADER_PX + visible * GRID_ROW_PX


# ── Session state init ───────────────────────────────────────────────────────

def _init_state() -> None:
    prefs = _load_prefs()
    defaults = {
        "oanda_api_key": prefs.get("oanda_api_key", ""),
        "oanda_account_id": prefs.get("oanda_account_id", ""),
        "oanda_env": prefs.get("oanda_env", "practice"),
        "auto_refresh": prefs.get("auto_refresh", False),
        "refresh_seconds": prefs.get("refresh_seconds", 60),
        "universe_choice": prefs.get("universe_choice", "Tight spread (recommended)"),
        "custom_pairs_raw": prefs.get("custom_pairs_raw", ""),
        "table_rows": prefs.get("table_rows", DEFAULT_TABLE_ROWS),
        "signal_timeframe": prefs.get("signal_timeframe", "M15"),
        "alerts_enabled": prefs.get("alerts_enabled", True),
        "alert_webhook": prefs.get("alert_webhook", ""),
        "auto_refresh_count_last": 0,
        "quotes_auto_refresh_count_last": 0,
        # Candidate model report, held across reruns so the retrain result survives
        # until it is promoted or a new retrain replaces it.
        "model_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Settings builder ─────────────────────────────────────────────────────────

def _build_settings() -> AppSettings:
    env_settings = get_settings()
    # Start from the environment so the SQL Server / ML block carries through,
    # then override only the OANDA fields the sidebar owns.
    return env_settings.model_copy(update={
        "oanda_api_key": st.session_state.oanda_api_key or env_settings.oanda_api_key,
        "oanda_account_id": st.session_state.oanda_account_id or env_settings.oanda_account_id,
        "oanda_env": st.session_state.oanda_env,
    })


# ── Signal color ─────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "STRONG_BUY": "🟢",
    "BUY_CANDIDATE": "🔵",
    "WATCH_ONLY": "🟡",
    "AVOID": "⚫",
    "SHORT_CANDIDATE": "🟠",
    "STRONG_SHORT": "🔴",
}


def _signal_badge(signal: str) -> str:
    return f"{SIGNAL_COLORS.get(signal, '⚫')} {signal}"


# ── Daily ML predictions (SQL Server mirror) ─────────────────────────────────

ML_SIGNAL_COLORS = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
_DIRECTION_ARROWS = {"UP": "▲ UP", "DOWN": "▼ DOWN", "FLAT": "= FLAT"}


def _ml_signal_badge(signal: str) -> str:
    signal = (signal or "").upper()
    return f"{ML_SIGNAL_COLORS.get(signal, '⚫')} {signal or '—'}"


def _direction_label(direction) -> str:
    return _DIRECTION_ARROWS.get(direction, "—")


def _outcome_badge(outcome) -> str:
    return {
        "HIT": "✅ HIT", "MISS": "❌ MISS", "ABSTAIN": "➖ ABSTAIN",
        "NO_DATA": "⚠️ NO DATA", "no broker data": "⚠️ NO DATA",
    }.get(outcome, "—")


def _fmt_pct1(value) -> str:
    try:
        return f"{value:.2f}%" if value is not None and value == value else ""
    except (TypeError, ValueError):
        return ""


def _hit_rate(hits, calls) -> str:
    if not calls:
        return "—"
    return f"{hits / calls:.0%} ({int(hits)}/{int(calls)})"


_TRACK_BADGES = {
    "HIT": "✅ ON TRACK",
    "MISS": "❌ OFF TRACK",
    "ABSTAIN": "➖ NO CALL",
    "N/A": "—",
}


def _track_badge(status) -> str:
    """Live wording for a running verdict — deliberately not the settled HIT/MISS."""
    return _TRACK_BADGES.get(status, "—")


# Streamlit re-runs the whole script on every widget interaction, not only on the
# refresh timer, so the live quote is cached for a fraction of the shortest
# refresh interval (30s): clicks reuse the last price, the timer always gets a
# fresh one.
LIVE_PRICE_TTL_SECONDS = 15


@st.cache_data(ttl=LIVE_PRICE_TTL_SECONDS, show_spinner=False)
def _fetch_live_mids(pairs: tuple, api_key: str, account_id: str, env: str) -> dict:
    """
    Current mid per pair, plus any pair this account cannot quote.

    One pricing request covers the whole list, but OANDA rejects the *entire*
    batch when a single instrument is unknown to the account — and the daily
    model predicts instruments OANDA no longer quotes (the same ones reconcile
    records as NO_DATA). A failed batch therefore falls back to one request per
    pair, which salvages the rest and names the offenders so the caller can stop
    asking for them.
    """
    settings = get_settings().model_copy(update={
        "oanda_api_key": api_key,
        "oanda_account_id": account_id or None,
        "oanda_env": env,
    })
    client = OandaClient(settings)

    def _mids(batch) -> dict:
        return {
            q.pair: {"mid": (q.bid + q.ask) / 2, "as_of": q.as_of}
            for q in client.get_pricing(list(batch))
        }

    try:
        return {"prices": _mids(pairs), "unavailable": [], "error": None}
    except Exception as batch_exc:
        prices, unavailable = {}, []
        for pair in pairs:
            try:
                prices.update(_mids([pair]))
            except Exception:
                unavailable.append(pair)
        if not prices:
            # Nothing came back at all — this is a connection/auth problem, not a
            # bad instrument, so report it instead of blaming every pair.
            return {"prices": {}, "unavailable": [], "error": str(batch_exc)}
        return {"prices": prices, "unavailable": unavailable, "error": None}


def _live_ml_prices(df: pd.DataFrame, settings: AppSettings) -> tuple:
    """Live mids for the predicted pairs; returns ``(prices, error)``."""
    if not settings.oanda_api_key or df.empty:
        return {}, None

    skip = set(st.session_state.get("ml_live_price_skip", ()))
    wanted = tuple(sorted(p for p in df["pair"].unique() if p not in skip))
    if not wanted:
        return {}, None

    fetched = _fetch_live_mids(
        wanted, settings.oanda_api_key, settings.oanda_account_id or "", settings.oanda_env,
    )
    if fetched["unavailable"]:
        st.session_state.ml_live_price_skip = sorted(skip.union(fetched["unavailable"]))
    return fetched["prices"], fetched["error"]


def _auto_sync_ml(settings: AppSettings, storage: Storage) -> dict:
    """
    Refresh the SQLite mirror if — and only if — it is behind the day's run.

    Called on every dashboard render, including auto-refreshes; the throttle and
    the up-to-date check live in ``ml_sync`` so the common path never opens a
    connection to SQL Server.
    """
    try:
        return sync_predictions(storage, settings)
    except Exception as exc:  # a data-source problem must not blank the page
        return {"status": "error", "reason": str(exc), "rows": 0}


def _render_ml_prediction_box(settings: AppSettings, storage: Storage) -> None:
    """Home-page box: yesterday's close-based ML direction for every pair in SQL Server."""
    result = _auto_sync_ml(settings, storage)
    rows = storage.load_ml_latest_predictions()

    prediction_date = rows[0]["prediction_date"] if rows else None
    target_date = rows[0].get("target_date") if rows else None
    title = "🤖 Daily ML Direction (SQL Server)"
    if prediction_date:
        title += f" — from {prediction_date} close, for {target_date}"

    with st.expander(title, expanded=True):
        if not settings.ml_source_configured:
            st.info(
                "SQL Server is not configured. Add SQL_SERVER / SQL_DATABASE and "
                "credentials to `.env` to pull the daily ML predictions."
            )
            return

        head, actions = st.columns([5, 1])
        with actions:
            if st.button("↻ Sync now", key="ml_sync_now", use_container_width=True):
                with st.spinner("Reading SQL Server…"):
                    result = sync_predictions(storage, settings, force=True)
                rows = storage.load_ml_latest_predictions()
                if result["status"] == "error":
                    st.error(result["reason"])
                else:
                    st.success(f"Synced {result['rows']} rows")

        if not rows:
            with head:
                if result["status"] == "error":
                    st.error(f"Sync failed: {result['reason']}")
                else:
                    st.info("No ML predictions mirrored yet — click **Sync now**.")
            return

        df = pd.DataFrame(rows)
        df["Direction"] = df["predicted_signal"].apply(_ml_signal_badge)
        df["Lean"] = df.apply(
            lambda r: _direction_label(implied_direction(r["prob_buy"], r["prob_sell"])), axis=1
        )
        df["Pair"] = df["pair"].apply(format_pair)

        # Live leg: where price sits *now* against the close the model predicted
        # from. Refreshed with the page, so the static morning call and the
        # running market read sit on the same row.
        live_prices, live_error = _live_ml_prices(df, settings)
        progress = [
            live_progress(row, (live_prices.get(row["pair"]) or {}).get("mid"))
            for row in df.to_dict("records")
        ]
        df["Now"] = [p["current_price"] for p in progress]
        df["Now Dir"] = [_direction_label(p["current_direction"]) for p in progress]
        df["Move (pips)"] = [p["move_pips"] for p in progress]
        df["Move %"] = [p["move_pct"] for p in progress]
        df["vs Signal"] = [_track_badge(p["signal_status"]) for p in progress]
        df["vs Lean"] = [_track_badge(p["lean_status"]) for p in progress]

        counts = df["predicted_signal"].value_counts().to_dict()
        called = [p for p in progress if p["signal_status"] in ("HIT", "MISS")]
        on_track = sum(1 for p in called if p["signal_status"] == "HIT")
        with head:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Pairs", len(df))
            c2.metric("🟢 BUY", counts.get("BUY", 0))
            c3.metric("🔴 SELL", counts.get("SELL", 0))
            c4.metric("⚪ HOLD", counts.get("HOLD", 0))
            c5.metric(
                "✅ On track",
                f"{on_track}/{len(called)}" if called else "—",
                help="BUY/SELL calls whose direction the live price currently agrees with.",
            )

        live_cols = ["Now", "Move (pips)", "Move %", "Now Dir", "vs Signal", "vs Lean"]
        display = df[[
            "Pair", "Direction", "signal_confidence", "Lean", "base_close",
            *(live_cols if live_prices else []),
            "prob_buy", "prob_sell",
        ]].rename(columns={
            "signal_confidence": "Confidence",
            "prob_buy": "P(up)",
            "prob_sell": "P(down)",
            "base_close": "Close",
        })

        def _track_row(row):
            verdict = row.get("vs Signal", "")
            if verdict.endswith("ON TRACK"):
                return ["background-color: #d8f3d8; color: #000000"] * len(row)
            if verdict.endswith("OFF TRACK"):
                return ["background-color: #f8d7da; color: #000000"] * len(row)
            return [""] * len(row)

        styled = display.style.format({
            "Confidence": "{:.1%}", "P(up)": "{:.1%}", "P(down)": "{:.1%}",
            "Close": "{:.5f}", "Now": "{:.5f}",
            "Move (pips)": "{:+.1f}", "Move %": "{:+.2f}%",
        }, na_rep="")
        if live_prices:
            styled = styled.apply(_track_row, axis=1)

        st.dataframe(
            styled, use_container_width=True, hide_index=True,
            height=_grid_height(len(display)),
        )

        last_sync = storage.load_ml_last_sync(kind="predictions")
        synced_at = last_sync.get("attempted_at") if last_sync else "never"
        _, reason = sync_due(storage, settings)
        st.caption(
            f"Static daily snapshot — model `{df['model_name'].iloc[0]}` "
            f"v`{df['model_version'].iloc[0]}`. Mirrored to SQLite; SQL Server was "
            f"last contacted at {synced_at} UTC. Next check: {reason}. "
            "**HOLD** means the model's confidence gate abstained, not a flat forecast — "
            "the *Lean* column shows the direction underneath it."
        )

        if live_prices:
            quoted_at = next(
                (q.get("as_of") for q in live_prices.values() if q.get("as_of")), "—"
            )
            st.caption(
                f"**Now** is the live OANDA mid ({len(live_prices)} pairs, quoted "
                f"{quoted_at}), and the move is measured from the same "
                f"{prediction_date} close the model predicted from — so *vs Signal* / "
                f"*vs Lean* preview the verdict the {target_date} reconcile will record "
                "at the 17:00 ET close. Until that close it can still flip."
                + ("" if is_forex_market_open() else " Market is closed — prices are last-traded.")
            )
        elif not settings.oanda_api_key:
            st.caption("Add an OANDA API key in the sidebar to compare the live price against each call.")
        elif live_error:
            st.caption(f"Live prices unavailable: {live_error}")

        skipped = st.session_state.get("ml_live_price_skip", [])
        if skipped:
            st.caption(
                "No live quote from this OANDA account for: "
                + ", ".join(format_pair(p) for p in skipped)
            )

        if result["status"] == "error":
            st.warning(f"Last sync attempt failed: {result['reason']}")


def _render_reconcile_tab(settings: AppSettings, storage: Storage) -> None:
    """
    Score each New York session's realised move against that morning's ML call.

    The prediction is made from the previous session's close and targets the next
    one, so the natural checkpoint is 17:00 ET — the NY close, which is also where
    OANDA's daily candle boundary sits.
    """
    st.subheader("ML Prediction Reconciliation")
    st.caption(
        "Each closed New York session (17:00 ET) scored against the ML direction "
        "predicted from the previous session's close."
    )

    if not storage.load_ml_max_prediction_date():
        st.info(
            "No ML predictions mirrored yet. Open the **Results** tab and use "
            "**Sync now** in the Daily ML Direction box."
        )
        return

    latest_session = last_completed_session_date()
    reconciled_dates = storage.load_ml_reconciled_dates()
    known_dates = storage.load_ml_target_dates()
    # Only sessions that have actually closed can be scored.
    selectable = [d for d in known_dates if d <= latest_session.isoformat()]
    if not selectable:
        st.info(f"No predictions target a session on or before {latest_session}.")
        return

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 3])
    chosen = ctrl1.selectbox(
        "Session (NY close)", selectable, index=0,
        format_func=lambda d: f"{d}{'' if d in reconciled_dates else '  · not scored'}",
    )
    manual_eval = ctrl2.button("▶ Evaluate", use_container_width=True)
    run_eval = manual_eval

    api_ok = bool(settings.oanda_api_key)
    if not api_ok:
        ctrl3.warning("OANDA API key required to fetch the realised close.")

    # Auto-evaluate the newest closed session, throttled so a session whose
    # candles have not landed yet is retried on a timer, not every refresh.
    if api_ok and not manual_eval and chosen == latest_session.isoformat():
        due, _ = reconcile_due(storage, latest_session)
        if due:
            run_eval = True

    if run_eval and api_ok:
        with st.spinner(f"Scoring {chosen} against OANDA daily closes…"):
            try:
                # A manual click re-scores everything; the automatic pass only
                # fills in pairs that are still outstanding.
                outcome = reconcile_session(
                    storage, OandaClient(settings),
                    target_date=date.fromisoformat(chosen), force=manual_eval,
                )
            except Exception as exc:
                outcome = {"status": "error", "reason": str(exc), "scored": 0}
        if outcome["status"] == "error":
            st.error(f"Reconcile failed: {outcome['reason']}")
        elif outcome["scored"]:
            st.success(f"Scored {outcome['scored']} pair(s) for {chosen}")
        elif outcome["status"] == "pending":
            st.info(f"Waiting on daily candles for {chosen}: {outcome['reason']}")

    rows = storage.load_ml_reconciliation(target_date=chosen)
    if not rows:
        st.info(f"{chosen} has not been scored yet — click **Evaluate**.")
    else:
        df = pd.DataFrame(rows)

        signal_calls = int((df["signal_outcome"].isin(["HIT", "MISS"])).sum())
        signal_hits = int((df["signal_outcome"] == "HIT").sum())
        lean_calls = int((df["implied_outcome"].isin(["HIT", "MISS"])).sum())
        lean_hits = int((df["implied_outcome"] == "HIT").sum())

        scoreable = int((df["signal_outcome"] != "NO_DATA").sum())
        avg_move = df["actual_pips"].abs().mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Actioned signals", f"{signal_calls}/{scoreable}",
                  help="BUY or SELL calls. The rest were HOLD — the gate abstained.")
        m2.metric("Signal accuracy", _hit_rate(signal_hits, signal_calls))
        m3.metric("Lean accuracy", _hit_rate(lean_hits, lean_calls),
                  help="Direction implied by P(up) vs P(down), scored on every pair "
                       "including abstained ones.")
        m4.metric("Avg |move|", f"{avg_move:.1f} pips" if avg_move == avg_move else "—")

        no_data = df.loc[df["signal_outcome"] == "NO_DATA", "pair"].tolist()
        if no_data:
            st.caption(
                f"⚠️ No broker price data for {', '.join(format_pair(p) for p in no_data)} — "
                "predicted daily but not quoted by OANDA, so held out of the rates above."
            )

        display = pd.DataFrame({
            "Pair": df["pair"].apply(format_pair),
            "Predicted": df["predicted_signal"].apply(_ml_signal_badge),
            "Conf": df["signal_confidence"],
            "Lean": df["implied_direction"].apply(_direction_label),
            "Actual": df["actual_direction"].apply(_direction_label),
            "Move %": df["actual_return_pct"],
            "Pips": df["actual_pips"],
            "Signal": df["signal_outcome"].apply(_outcome_badge),
            "Lean result": df["implied_outcome"].apply(_outcome_badge),
            "Prev close": df["base_close"],
            "Close": df["actual_close"],
        })

        def _shade(row):
            if row["Signal"].endswith("HIT"):
                return ["background-color: #d4edda; color: #000000"] * len(row)
            if row["Signal"].endswith("MISS"):
                return ["background-color: #f8d7da; color: #000000"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display.style
            .format({"Conf": "{:.1%}", "Move %": "{:+.2f}%", "Pips": "{:+.1f}",
                     "Prev close": "{:.5f}", "Close": "{:.5f}"}, na_rep="")
            .apply(_shade, axis=1),
            use_container_width=True, hide_index=True,
            height=_grid_height(len(display)),
        )

        # Feed cross-check: the mirror's base close comes from SQL Server's own
        # price history, the realised move from OANDA. A wide gap means the two
        # feeds have diverged and the scoring below it is not comparable.
        drift = (df["sql_base_close"] - df["base_close"]).abs()
        worst = drift.max() if len(drift) else 0
        if worst and worst > 0:
            rel = (drift / df["base_close"]).max()
            if rel > 0.002:
                st.warning(
                    f"SQL Server and OANDA disagree on the previous close by up to "
                    f"{rel:.2%} — treat these results as indicative."
                )

    st.divider()
    hist_head, hist_action = st.columns([4, 1])
    hist_head.subheader("History")
    if hist_action.button("⏮ Backfill", key="ml_backfill", disabled=not api_ok,
                          use_container_width=True,
                          help="Score every past session already mirrored. Costs one "
                               "OANDA request per pair, not per session."):
        with st.spinner("Scoring past sessions…"):
            try:
                filled = backfill_reconciliation(storage, OandaClient(settings))
            except Exception as exc:
                filled = {"status": "error", "reason": str(exc), "scored": 0}
        if filled["status"] == "error":
            st.error(f"Backfill failed: {filled['reason']}")
        else:
            st.success(f"Backfill: {filled['reason']}")

    history = storage.load_ml_reconciliation_by_date()
    if not history:
        st.info("No sessions scored yet — use **Backfill** to score the mirrored history.")
        return

    hist_df = pd.DataFrame(history)
    total_signal_calls = int(hist_df["signal_calls"].sum())
    total_signal_hits = int(hist_df["signal_hits"].sum())
    total_lean_calls = int(hist_df["lean_calls"].sum())
    total_lean_hits = int(hist_df["lean_hits"].sum())

    h1, h2, h3 = st.columns(3)
    h1.metric("Sessions scored", len(hist_df))
    h2.metric("Signal accuracy (all)", _hit_rate(total_signal_hits, total_signal_calls))
    h3.metric("Lean accuracy (all)", _hit_rate(total_lean_hits, total_lean_calls))
    if total_lean_calls:
        st.caption(
            "A directional coin flip is 50%. Lean accuracy is the honest read on the "
            "daily model since it scores every pair; signal accuracy covers only the "
            "days the gate took a side."
        )

    hist_display = pd.DataFrame({
        "Session": hist_df["target_date"],
        "Pairs": hist_df["pairs"],
        "Signals": hist_df["signal_calls"],
        "Signal hit rate": hist_df.apply(
            lambda r: r["signal_hits"] / r["signal_calls"] if r["signal_calls"] else None, axis=1),
        "Lean hit rate": hist_df.apply(
            lambda r: r["lean_hits"] / r["lean_calls"] if r["lean_calls"] else None, axis=1),
        "Avg move %": hist_df["avg_return_pct"],
    })
    st.dataframe(
        hist_display.style.format(
            {"Signal hit rate": "{:.0%}", "Lean hit rate": "{:.0%}", "Avg move %": "{:+.2f}%"},
            na_rep="—",
        ),
        use_container_width=True, hide_index=True,
    )

    with st.expander("By pair"):
        by_pair = storage.load_ml_reconciliation_by_pair()
        if by_pair:
            pair_df = pd.DataFrame(by_pair)
            st.dataframe(
                pd.DataFrame({
                    "Pair": pair_df["pair"].apply(format_pair),
                    "Sessions": pair_df["sessions"],
                    "Signals": pair_df["signal_calls"],
                    "Signal hit rate": pair_df.apply(
                        lambda r: r["signal_hits"] / r["signal_calls"] if r["signal_calls"] else None, axis=1),
                    "Lean hit rate": pair_df.apply(
                        lambda r: r["lean_hits"] / r["lean_calls"] if r["lean_calls"] else None, axis=1),
                }).style.format(
                    {"Signal hit rate": "{:.0%}", "Lean hit rate": "{:.0%}"}, na_rep="—"
                ),
                use_container_width=True, hide_index=True,
            )


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple:
    st.sidebar.title("💱 Forex Dashboard")

    # Session status
    session = current_session()
    market_open = is_forex_market_open()
    badge = session_badge_color(session)
    st.sidebar.markdown(f"**Session:** {badge} {session.replace('_', ' ')}")
    if not market_open:
        st.sidebar.warning("Forex market is closed (weekend).")

    st.sidebar.divider()

    # API credentials
    st.sidebar.subheader("OANDA Credentials")
    api_key = st.sidebar.text_input(
        "API Key",
        value=st.session_state.oanda_api_key,
        type="password",
        key="input_api_key",
    )
    account_id = st.sidebar.text_input(
        "Account ID (optional — auto-fetched)",
        value=st.session_state.oanda_account_id,
        key="input_account_id",
    )
    oanda_env = st.sidebar.radio(
        "Environment",
        ["practice", "live"],
        index=0 if st.session_state.oanda_env == "practice" else 1,
        horizontal=True,
        key="input_env",
    )

    if api_key != st.session_state.oanda_api_key:
        st.session_state.oanda_api_key = api_key
    if account_id != st.session_state.oanda_account_id:
        st.session_state.oanda_account_id = account_id
    if oanda_env != st.session_state.oanda_env:
        st.session_state.oanda_env = oanda_env

    api_ok = bool(api_key)
    if api_ok:
        st.sidebar.success("API key set")
    else:
        st.sidebar.error("Enter your OANDA API key")

    st.sidebar.divider()

    # Pair universe
    st.sidebar.subheader("Pair Universe")
    universe_options = list(UNIVERSE_MAP.keys()) + ["Custom"]
    saved_choice = st.session_state.universe_choice
    universe_choice = st.sidebar.radio(
        "Universe",
        universe_options,
        index=universe_options.index(saved_choice) if saved_choice in universe_options else 0,
        horizontal=False,
    )
    st.session_state.universe_choice = universe_choice
    if universe_choice == "Custom":
        custom_raw = st.sidebar.text_area(
            "Custom pairs (one per line, e.g. EUR_USD)",
            value=st.session_state.custom_pairs_raw,
            height=120,
            placeholder="EUR_USD\nGBP_USD\nUSD_JPY",
        )
        st.session_state.custom_pairs_raw = custom_raw
        selected_pairs = [
            p.strip().upper()
            for p in custom_raw.replace(",", "\n").splitlines()
            if p.strip()
        ]
    else:
        selected_pairs = UNIVERSE_MAP[universe_choice]
        st.sidebar.caption(f"{len(selected_pairs)} pairs: {', '.join(format_pair(p) for p in selected_pairs)}")

    st.sidebar.divider()

    # Scan settings
    st.sidebar.subheader("Scan Settings")
    max_spread = st.sidebar.slider("Max spread (pips)", 0.5, 5.0, 2.0, 0.1)
    signal_mode = st.sidebar.selectbox(
        "Signal mode",
        ["All", "Momentum", "Mean Reversion", "Session Breakout"],
    )

    st.sidebar.divider()

    # Auto-refresh
    st.sidebar.subheader("Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Auto-refresh scanner", value=st.session_state.auto_refresh)
    refresh_secs = st.sidebar.number_input(
        "Interval (seconds)",
        min_value=30,
        max_value=1800,
        value=st.session_state.refresh_seconds,
        step=30,
    )
    st.session_state.auto_refresh = auto_refresh
    st.session_state.refresh_seconds = refresh_secs

    st.sidebar.divider()

    # Display — how tall the main grids are allowed to grow before scrolling.
    st.sidebar.subheader("Display")
    table_rows = st.sidebar.slider(
        "Table height (rows)",
        min_value=MIN_TABLE_ROWS,
        max_value=MAX_TABLE_ROWS,
        value=int(st.session_state.table_rows),
        step=1,
        help="Rows shown before a grid starts scrolling. Grids never grow taller "
             "than their own data, so raising this only removes scrollbars.",
    )
    st.session_state.table_rows = table_rows
    st.sidebar.caption(
        f"Scanner, ML direction, Reconcile and Live Quotes grids fit up to "
        f"{table_rows} rows (~{_grid_height(table_rows)}px)."
    )

    # ── Signal timeframe ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Signal Timeframe")
    tf_options = ["M15", "H1", "M5"]
    saved_tf = st.session_state.get("signal_timeframe", "M15")
    signal_timeframe = st.sidebar.radio(
        "Bars the signal is computed on",
        tf_options,
        index=tf_options.index(saved_tf) if saved_tf in tf_options else 0,
        help="M15 is the validated default. M5 ATR is 2-4 pips, so a stop wide "
             "enough to cover the spread sits far outside the signal's own horizon "
             "— backtested, every M5 setup is vetoed on cost. H1 measured worse "
             "than M15 (-0.115R vs -0.001R).",
    )
    st.session_state.signal_timeframe = signal_timeframe
    if signal_timeframe == "M5":
        st.sidebar.warning(
            "M5 is retained for comparison only. Its brackets cannot amortise the "
            "spread, and the cost veto rejects effectively all of them."
        )

    # ── Alerts ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Alerts")
    alerts_enabled = st.sidebar.checkbox(
        "Push alerts on qualifying setups",
        value=st.session_state.get("alerts_enabled", True),
        help="Only setups that clear the entry-quality gate raise an alert — "
             "about a fifth of what the scanner arms.",
    )
    webhook_url = st.sidebar.text_input(
        "Webhook URL (optional)",
        value=st.session_state.get("alert_webhook", ""),
        type="password",
        placeholder="https://ntfy.sh/your-private-topic",
        help="ntfy topic URL (phone push via the ntfy app), Slack, Discord or any "
             "endpoint accepting JSON. Blank = FOREX_ALERT_WEBHOOK_URL from .env, "
             "or dashboard-only alerts if that is unset too.",
    )
    st.session_state.alerts_enabled = alerts_enabled
    st.session_state.alert_webhook = webhook_url

    # Save prefs
    _save_prefs({
        "oanda_api_key": api_key,
        "oanda_account_id": account_id,
        "oanda_env": oanda_env,
        "auto_refresh": auto_refresh,
        "refresh_seconds": refresh_secs,
        "universe_choice": st.session_state.universe_choice,
        "custom_pairs_raw": st.session_state.custom_pairs_raw,
        "table_rows": table_rows,
        "signal_timeframe": signal_timeframe,
        "alerts_enabled": alerts_enabled,
        "alert_webhook": webhook_url,
    })

    return api_key, selected_pairs, max_spread, signal_mode, auto_refresh, refresh_secs


# ── Alerts ───────────────────────────────────────────────────────────────────

def _alert_sinks() -> list:
    """
    Delivery destinations for this session.

    The dashboard feed is not a sink — alerts are persisted by the scanner before
    any sink runs, so the feed shows everything that was raised whether or not the
    webhook was reachable.

    A blank sidebar URL falls back to FOREX_ALERT_WEBHOOK_URL: the dashboard and
    the headless runner share the alert dedupe, so an alert the dashboard raised
    without pushing would never reach the phone from the runner either.
    """
    if not st.session_state.get("alerts_enabled", True):
        return []
    url = (st.session_state.get("alert_webhook") or "").strip() or get_settings().alert_webhook_url
    sink = make_sink(url)
    return [sink] if sink else []


def _render_alerts_tab(storage: Storage) -> None:
    st.subheader("Alerts")
    st.caption(
        "Setups that cleared the entry-quality gate. Measured over 1,003 trades "
        "carrying features, gated entries won 52.3% against the 43.9% that RR 1.5 "
        "needs at a 10% cost ratio — versus 39.2% ungated. The gate keeps roughly "
        "one setup in twelve, so a quiet feed is the system working."
    )

    window = st.selectbox(
        "Window", [("Last 24 hours", 1440), ("Last 3 days", 4320),
                   ("Last week", 10080), ("Everything", None)],
        format_func=lambda opt: opt[0], index=0,
    )
    rows = storage.load_alerts(limit=200, since_minutes=window[1])
    if not rows:
        st.info("No alerts in this window.")
        return

    undelivered = [r for r in rows if r["delivery_error"]]
    if undelivered:
        st.warning(
            f"{len(undelivered)} alert(s) failed webhook delivery — they are listed "
            f"below regardless. Most recent error: {undelivered[0]['delivery_error']}"
        )

    df = pd.DataFrame(rows)
    df["When"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["Side"] = df["direction"].map(_direction_label)
    df["Risk"] = df["stop_pips"].map(lambda v: f"{v:.0f}p" if pd.notna(v) else "—")
    df["Reward"] = df["target_pips"].map(lambda v: f"{v:.0f}p" if pd.notna(v) else "—")
    df["Cost"] = df["cost_ratio"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    view = df[[
        "When", "urgency", "pair", "Side", "entry", "stop", "target",
        "Risk", "Reward", "Cost", "total_score", "regime", "session", "reason",
    ]].rename(columns={
        "urgency": "Urgency", "pair": "Pair", "entry": "Entry", "stop": "Stop",
        "target": "Target", "total_score": "Score", "regime": "Regime",
        "session": "Session", "reason": "Notes",
    })
    st.dataframe(
        view, width="stretch", hide_index=True,
        height=_grid_height(len(view), st.session_state.table_rows),
        column_config={
            "When": st.column_config.DatetimeColumn(format="MMM DD HH:mm"),
            "Entry": st.column_config.NumberColumn(format="%.5f"),
            "Stop": st.column_config.NumberColumn(format="%.5f"),
            "Target": st.column_config.NumberColumn(format="%.5f"),
            "Score": st.column_config.NumberColumn(format="%.0f"),
        },
    )


# ── Scanner page ─────────────────────────────────────────────────────────────

def _page_scanner(
    settings: AppSettings,
    storage: Storage,
    selected_pairs: list,
    max_spread: float,
    signal_mode: str,
    auto_refresh: bool,
    refresh_secs: int,
) -> None:
    st.title("Forex Scanner")

    api_ok = bool(settings.oanda_api_key)

    # Auto-refresh wiring
    auto_count = None
    if auto_refresh and api_ok:
        auto_count = st_autorefresh(interval=refresh_secs * 1000, key="scanner_autorefresh")

    auto_due = (
        auto_refresh
        and api_ok
        and bool(selected_pairs)
        and auto_count is not None
        and auto_count != st.session_state.auto_refresh_count_last
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        run_now = st.button("▶ Run Scan", type="primary", disabled=not api_ok or not selected_pairs)
    with col2:
        if not api_ok:
            st.warning("Enter your OANDA API key in the sidebar.")
        elif not selected_pairs:
            st.warning("Select or enter forex pairs.")

    if run_now or auto_due:
        request = ScanRequest(
            pairs=selected_pairs,
            max_spread_pips=max_spread,
            signal_mode=signal_mode,
            signal_timeframe=st.session_state.get("signal_timeframe", "M15"),
        )
        with st.spinner(f"Scanning {len(selected_pairs)} pairs…"):
            try:
                summary = run_scan(settings, storage, request, alert_sinks=_alert_sinks())
                st.session_state.auto_refresh_count_last = auto_count or 0
                st.success(
                    f"Scan complete — {summary.pairs_scanned} pairs, "
                    f"{summary.signals_found} signals, "
                    f"{summary.alerts_raised} alerts, {summary.errors} errors"
                )
                # Surface new alerts immediately rather than waiting for the operator
                # to notice a tab badge — this is the "near real-time" half of the job.
                for row in storage.load_alerts(limit=summary.alerts_raised or 0,
                                               since_minutes=5):
                    st.toast(
                        f"{'🔴' if row['urgency'] == 'HIGH' else '🟡'} "
                        f"{_direction_label(row['direction'])} {row['pair']} "
                        f"@ {row['entry']:g}",
                        icon="🔔",
                    )
            except Exception as exc:
                st.error(f"Scan failed: {exc}")

    (tab_results, tab_alerts, tab_reconcile, tab_watchlist, tab_perf,
     tab_model, tab_logs, tab_settings) = st.tabs(
        ["Results", "Alerts", "Reconcile", "Watchlist", "Performance", "Model",
         "Scan Logs", "Settings"]
    )

    with tab_alerts:
        _render_alerts_tab(storage)

    # ── Results tab ──────────────────────────────────────────────────────────
    with tab_results:
        # Daily ML direction from SQL Server — rendered before the scan results so
        # it is visible on a cold start, when there is no scan to show yet.
        _render_ml_prediction_box(settings, storage)

        rows = storage.load_latest_snapshots()
        if not rows:
            st.info("No scan results yet. Click 'Run Scan' to start.")
        else:
            df = pd.DataFrame(rows)

            # Currency Strength Meter
            with st.expander("Currency Strength Meter", expanded=True):
                strength_scores = calculate_strength(rows)
                strength_df = pd.DataFrame(
                    [{"Currency": c, "Strength": strength_scores.get(c, 50.0)} for c in CURRENCIES]
                ).sort_values("Strength", ascending=False)
                st.bar_chart(strength_df.set_index("Currency"), height=220)
                st.caption("Strength normalized 0–100 from each pair's ~24h change. Higher = relatively stronger currency.")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            last_run = storage.load_latest_scan_run()
            last_ts = last_run.get("finished_at", "—") if last_run else "—"
            m1.metric("Pairs Scanned", len(df))
            m2.metric("Actionable", len(df[df["trade_signal"].isin(["STRONG_BUY", "BUY_CANDIDATE", "STRONG_SHORT", "SHORT_CANDIDATE"])]))
            m3.metric("Avg Score", f"{df['total_score'].mean():.1f}")
            m4.metric("Last Scan", last_ts)

            # Filters
            fc1, fc2, fc3 = st.columns(3)
            all_pairs = sorted(df["pair"].unique().tolist())
            all_signals = sorted(df["trade_signal"].unique().tolist())
            pair_filter = fc1.multiselect("Filter pairs", all_pairs, default=[])
            signal_filter = fc2.multiselect("Filter signals", all_signals, default=[])
            mtf_filter = fc3.selectbox("MTF Confluence", ["All", "Full Confluence Only", "Partial+"])

            filtered = df.copy()
            if pair_filter:
                filtered = filtered[filtered["pair"].isin(pair_filter)]
            if signal_filter:
                filtered = filtered[filtered["trade_signal"].isin(signal_filter)]
            if mtf_filter == "Full Confluence Only" and "mtf_confluence" in filtered.columns:
                filtered = filtered[filtered["mtf_confluence"] == "FULL"]
            elif mtf_filter == "Partial+" and "mtf_confluence" in filtered.columns:
                filtered = filtered[filtered["mtf_confluence"].isin(["FULL", "PARTIAL"])]

            # Apply signal mode filter
            if signal_mode == "Momentum":
                filtered = filtered[filtered["momentum_score"] >= filtered["reversion_score"]]
            elif signal_mode == "Mean Reversion":
                filtered = filtered[filtered["reversion_score"] >= filtered["momentum_score"]]
            elif signal_mode == "Session Breakout":
                filtered = filtered[filtered["session_score"] > 0]

            # Format display columns
            display_cols = [
                "pair", "trade_signal", "total_score", "regime",
                "model_prob", "required_prob",
                "momentum_score", "reversion_score", "session_score",
                "mtf_score", "mtf_confluence",
                "bid", "ask",
                "suggested_entry", "suggested_stop", "suggested_target",
                "stop_pips", "target_pips",
                "h1_direction", "h4_direction",
                "sr_score", "at_key_level", "blocked_ahead",
                "nearest_support", "nearest_resistance",
                "strength_assessment",
                "rr_ratio", "spread_pips", "cost_ratio",
                "close", "day_change_pct",
                "rsi14", "adx14", "macd_histogram", "ema9", "ema20",
                "atr14", "bb_width_pct",
                "current_session", "signal_reason", "as_of",
            ]
            display = filtered[[c for c in display_cols if c in filtered.columns]].copy()
            display["pair"] = display["pair"].apply(format_pair)
            display["trade_signal"] = display["trade_signal"].apply(_signal_badge)

            def _fmt5(x):
                try:
                    return f"{x:.5f}" if x is not None and x == x else ""
                except (TypeError, ValueError):
                    return ""

            price_cols = {c: _fmt5 for c in [
                "bid", "ask", "close", "ema9", "ema20",
                "nearest_support", "nearest_resistance",
                "suggested_entry", "suggested_stop", "suggested_target",
            ] if c in display.columns}

            def _fmt_pct(x):
                try:
                    return f"{x:.0%}" if x is not None and x == x else ""
                except (TypeError, ValueError):
                    return ""

            price_cols.update({
                c: _fmt_pct for c in ["model_prob", "required_prob", "cost_ratio"]
                if c in display.columns
            })

            def _highlight_row(row):
                # Structure blocking the target is the more actionable warning of the
                # two, so it wins the row colour when both apply.
                if row.get("blocked_ahead"):
                    return ["background-color: #f8d7da; color: #000000"] * len(row)
                if row.get("at_key_level"):
                    return ["background-color: #fff3b0; color: #000000"] * len(row)
                return [""] * len(row)

            styled = display.style.format(price_cols, na_rep="")
            if "at_key_level" in display.columns or "blocked_ahead" in display.columns:
                styled = styled.apply(_highlight_row, axis=1)

            st.dataframe(
                styled, use_container_width=True, hide_index=True,
                height=_grid_height(len(display)),
            )

            # Column guide
            with st.expander("Column Guide"):
                st.markdown("""
| Column | Description |
|--------|-------------|
| pair | Currency pair (e.g. EUR/USD) |
| trade_signal | Overall signal: STRONG_BUY, BUY_CANDIDATE, SHORT_CANDIDATE, STRONG_SHORT, WATCH_ONLY, AVOID |
| total_score | Combined score (momentum/reversion 0–40 + session 0–20 + MTF 0–30 + S/R −25…+25 + strength ±10) |
| model_prob | Trained model's P(target before stop). Blank until a model is activated |
| required_prob | Cost-adjusted breakeven win rate + margin. A signal must clear this to stay actionable |
| regime | ADX-based: TREND (momentum weighted), RANGE (reversion weighted), MIXED, UNKNOWN |
| momentum_score | EMA/MACD/RSI momentum component (0–40), weighted by regime |
| reversion_score | RSI extreme/Bollinger Band component (0–40), weighted by regime |
| session_score | Session quality/breakout component (0–20) |
| suggested_entry / stop / target | ATR-based levels (stop max of 1.5×ATR, 8 pips, 8× spread; target 1.5× the stop) |
| stop_pips / target_pips / rr_ratio | Risk in pips, reward in pips, reward:risk (1.5) |
| cost_ratio | Spread as a fraction of the stop. Above 15% the setup is vetoed — cost outruns any edge |
| adx14 | Trend strength: >25 trending, <18 ranging |
| mtf_score | Higher-timeframe confirmation bonus (0=none, 15=PARTIAL, 30=FULL) |
| mtf_confluence | FULL=H1 and H4 both agree, PARTIAL=one agrees and the other is flat, UNCONFIRMED=both flat, CONFLICT/OPPOSED=they disagree |
| h1_direction | H1 trend direction (LONG/SHORT/NEUTRAL) |
| h4_direction | H4 trend direction (LONG/SHORT/NEUTRAL) |
| sr_score | Structure relative to the trade: +25 entering at a level behind you, −25 when one blocks the path ahead |
| at_key_level | Entry sits at favourable structure — support for a long, resistance for a short (yellow row) |
| blocked_ahead | A level sits within 1×ATR of the path to target, so the move is unlikely to complete (red row) |
| nearest_support | Closest support level below current price |
| nearest_resistance | Closest resistance level above current price |
| strength_assessment | Currency strength: STRONG_BASE_WEAK_QUOTE / WEAK_BASE_STRONG_QUOTE / NEUTRAL |
| spread_pips | Bid-ask spread in pips |
| rsi14 | RSI(14): <30 oversold, >70 overbought |
| macd_histogram | MACD histogram (positive = bullish momentum) |
| atr14 | Average True Range — daily volatility measure |
| bb_width_pct | Bollinger Band width % — squeeze detection |
| current_session | Active market session |
                """)

            st.download_button(
                "Export CSV",
                display.to_csv(index=False).encode(),
                file_name="forex_scan.csv",
                mime="text/csv",
            )

    # ── Watchlist tab ────────────────────────────────────────────────────────
    with tab_watchlist:
        st.subheader("Add to Watchlist")
        with st.form("add_watchlist"):
            wc1, wc2, wc3 = st.columns(3)
            w_pair = wc1.text_input("Pair (e.g. EUR_USD)")
            w_signal = wc2.selectbox(
                "Signal",
                ["STRONG_BUY", "BUY_CANDIDATE", "SHORT_CANDIDATE", "STRONG_SHORT", "WATCH_ONLY"],
            )
            w_entry = wc3.number_input("Entry price", min_value=0.0, format="%.5f")
            wc4, wc5, wc6 = st.columns(3)
            w_target = wc4.number_input("Target price", min_value=0.0, format="%.5f")
            w_stop = wc5.number_input("Stop price", min_value=0.0, format="%.5f")
            w_notes = wc6.text_input("Notes")
            w_stop_pips = st.number_input("Stop (pips)", min_value=0.0, value=20.0)
            w_target_pips = st.number_input("Target (pips)", min_value=0.0, value=40.0)
            submitted = st.form_submit_button("Add")
            if submitted and w_pair:
                storage.add_watchlist(
                    w_pair.strip().upper(), w_signal, w_entry,
                    w_target, w_stop, w_stop_pips, w_target_pips, w_notes,
                )
                st.success(f"Added {format_pair(w_pair)} to watchlist.")

        st.subheader("Active Watches")
        watching = storage.load_watchlist("watching")
        if not watching:
            st.info("No active watches.")
        else:
            wdf = pd.DataFrame(watching)
            wdf["pair"] = wdf["pair"].apply(format_pair)
            if "signal" in wdf.columns:
                wdf["signal"] = wdf["signal"].apply(lambda s: _signal_badge(s) if s else s)
            st.dataframe(wdf, use_container_width=True, hide_index=True)

            cc1, cc2 = st.columns(2)
            close_id = cc1.number_input("Close watch ID", min_value=0, step=1, value=0)
            exit_price = cc2.number_input("Exit price (0 = skip outcome)", min_value=0.0, format="%.5f")
            if st.button("Close Watch") and close_id > 0:
                if exit_price > 0:
                    storage.close_watchlist_with_outcome(int(close_id), float(exit_price))
                    st.success(f"Closed watch ID {close_id} — outcome recorded.")
                else:
                    storage.close_watchlist(int(close_id))
                    st.success(f"Closed watch ID {close_id}.")
                st.rerun()

        with st.expander("Closed Watches"):
            closed = storage.load_watchlist("closed")
            if closed:
                cdf = pd.DataFrame(closed)
                cdf["pair"] = cdf["pair"].apply(format_pair)
                st.dataframe(cdf, use_container_width=True, hide_index=True)

    # ── Performance tab ───────────────────────────────────────────────────────
    with tab_perf:
        outcomes = storage.load_trade_outcomes(limit=10000)
        if not outcomes:
            st.info(
                "No trade outcomes yet. Outcomes accrue automatically as each scan forward-tests "
                "actionable signals against their ATR stop/target — or close watchlist entries with "
                "an exit price to log manual trades."
            )
        else:
            odf = pd.DataFrame(outcomes)
            odf["date"] = pd.to_datetime(odf["created_at"]).dt.date.astype(str)

            # Date filter
            all_dates = sorted(odf["date"].unique().tolist(), reverse=True)
            date_choice = st.selectbox("Filter by date", ["All dates"] + all_dates)
            fdf = odf if date_choice == "All dates" else odf[odf["date"] == date_choice]

            # Overall metrics (respect date filter)
            total_trades = len(fdf)
            wins = int((fdf["outcome"] == "WIN").sum())
            losses = int((fdf["outcome"] == "LOSS").sum())
            win_rate = wins / total_trades if total_trades else 0
            r_all = fdf["r_multiple"].dropna()
            r_wins = fdf.loc[fdf["outcome"] == "WIN", "r_multiple"].dropna()
            avg_win_r = r_wins.mean() if len(r_wins) else 0.0
            expectancy = r_all.mean() if len(r_all) else 0.0

            pm1, pm2, pm3, pm4, pm5 = st.columns(5)
            pm1.metric("Predictions", total_trades)
            pm2.metric("Wins / Losses", f"{wins} / {losses}")
            pm3.metric("Win Rate", f"{win_rate*100:.1f}%")
            pm4.metric("Avg Win R", f"{avg_win_r:.2f}R")
            pm5.metric("Expectancy", f"{expectancy:.3f}R/trade")
            if date_choice != "All dates":
                st.caption(f"{total_trades} predictions on {date_choice}: {wins} won, {losses} lost.")

            # Daily results summary
            st.subheader("Results by Day")
            daily = (
                odf.assign(
                    win=(odf["outcome"] == "WIN").astype(int),
                    loss=(odf["outcome"] == "LOSS").astype(int),
                )
                .groupby("date")
                .agg(Predictions=("outcome", "size"), Wins=("win", "sum"), Losses=("loss", "sum"), R=("r_multiple", "sum"))
                .reset_index()
                .sort_values("date", ascending=False)
            )
            daily["Win Rate"] = (daily["Wins"] / daily["Predictions"] * 100).map("{:.1f}%".format)
            daily["R"] = daily["R"].round(1)
            daily.columns = ["Date", "Predictions", "Wins", "Losses", "Net R", "Win Rate"]
            st.dataframe(
                daily[["Date", "Predictions", "Wins", "Losses", "Win Rate", "Net R"]],
                use_container_width=True, hide_index=True,
            )

            # Equity curve (cumulative R, respects date filter)
            if "r_multiple" in fdf.columns and len(fdf):
                st.subheader("Equity Curve (Cumulative R)")
                cum_r = fdf.sort_values("created_at")["r_multiple"].fillna(0).cumsum().reset_index(drop=True)
                st.line_chart(cum_r)

            # Win rate breakdowns (computed from the filtered set)
            def _dim_table(df: pd.DataFrame, col: str, label: str) -> None:
                if col not in df.columns or not len(df):
                    return
                grp = (
                    df.assign(win=(df["outcome"] == "WIN").astype(int))
                    .groupby(col)
                    .agg(Trades=("outcome", "size"), Wins=("win", "sum"), Expectancy=("r_multiple", "mean"))
                    .reset_index()
                    .sort_values("Trades", ascending=False)
                )
                grp["Win Rate"] = (grp["Wins"] / grp["Trades"] * 100).map("{:.1f}%".format)
                grp["Expectancy"] = grp["Expectancy"].round(3)
                grp.columns = [label, "Trades", "Wins", "Expectancy", "Win Rate"]
                st.subheader(f"Win Rate by {label}")
                st.dataframe(
                    grp[[label, "Trades", "Wins", "Win Rate", "Expectancy"]],
                    use_container_width=True, hide_index=True,
                )

            fdf_display = fdf.copy()
            fdf_display["pair"] = fdf_display["pair"].apply(format_pair)
            _dim_table(fdf_display, "pair", "Pair")
            _dim_table(fdf_display, "signal", "Signal")

            # Recent trade log
            with st.expander("Trade History"):
                st.dataframe(fdf_display, use_container_width=True, hide_index=True)

        # Open auto-tracked signals (forward-testing in progress)
        open_tracked = storage.load_tracked_signals("open")
        if open_tracked:
            with st.expander(f"Open Tracked Signals ({len(open_tracked)})"):
                tdf = pd.DataFrame(open_tracked)
                tdf["pair"] = tdf["pair"].apply(format_pair)
                keep = [c for c in [
                    "pair", "signal", "model_prob", "required_prob", "total_score",
                    "entry_price", "stop_price", "target_price",
                    "stop_pips", "target_pips", "cost_ratio", "spread_pips",
                    "session", "regime", "entry_ts", "created_at",
                ] if c in tdf.columns]
                st.dataframe(tdf[keep], use_container_width=True, hide_index=True)
                st.caption("Each scan checks these against their ATR stop/target; a touch records a WIN/LOSS outcome.")

    # ── Model tab ────────────────────────────────────────────────────────────
    with tab_model:
        st.subheader("Direction Model")
        st.caption(
            "The rules-based score proposes a setup; this model decides whether the "
            "measured odds justify paying the spread. Until one is trained and "
            "activated, the scanner runs rules-only and simply logs features."
        )

        training_rows = storage.load_training_rows(feature_version=FEATURE_VERSION)
        open_n = len(storage.load_tracked_signals("open"))
        needed = max(0, MIN_TRAIN_SAMPLES - len(training_rows))

        c1, c2, c3 = st.columns(3)
        c1.metric("Trainable trades", len(training_rows))
        c2.metric("Awaiting resolution", open_n)
        c3.metric("Needed to train", needed if needed else "ready")

        if needed:
            st.progress(min(1.0, len(training_rows) / MIN_TRAIN_SAMPLES))
            st.info(
                f"{needed} more resolved trades needed before the first train "
                f"(minimum {MIN_TRAIN_SAMPLES}). Every actionable signal now stores its "
                "feature vector, so this fills up as signals resolve."
            )

        models = storage.load_models()
        active = next((m for m in models if m["is_active"]), None)

        if not models:
            st.warning("No model trained yet — the scanner is running rules-only.")
        elif active:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("OOS AUC", f"{active['auc']:.4f}" if active["auc"] else "—")
            a2.metric("Top-decile precision",
                      f"{active['top_decile_prec']:.3f}" if active["top_decile_prec"] else "—")
            a3.metric("Brier", f"{active['brier']:.4f}" if active["brier"] else "—")
            a4.metric("Trained on", active["n_train"] or "—")
            st.success(f"Model #{active['id']} is active — signals are being probability-gated.")
            if active["auc"] is not None and active["auc"] < 0.53:
                st.warning(
                    f"Active model's out-of-sample AUC is {active['auc']:.4f} — "
                    "close to the 0.50 no-skill line. Treat its vetoes as weak evidence."
                )
        else:
            st.info("Models exist but none is active. The scanner is running rules-only.")

        # ── When to retrain ─────────────────────────────────────────────────
        with st.expander("When to retrain, and how to read the result"):
            st.markdown(f"""
**Required**

- The first time you cross **{MIN_TRAIN_SAMPLES}** resolved trades.
- After editing `FEATURE_NAMES` in `forex/features.py` — bump `FEATURE_VERSION` and the
  scanner ignores the stale model (fails closed) until you retrain.
- After changing `_RR`, `_STOP_ATR_MULT`, `_MIN_STOP_PIPS` or `_SPREAD_STOP_MULT` in
  `forex/signals.py`. Those change the stop/target geometry, so older labels describe a
  different bracket and the model would be fitting the wrong thing.

**Worth doing**

- Every ~50–100 newly resolved trades once past the first train.
- If the realised win rate on gated trades sits below the model's predicted rate for
  several weeks — that is calibration drift.

**Don't** retrain on a handful of new rows. With ~29 features you would be chasing
noise, and each retrain shifts the veto threshold under your live signals.

---

**Reading the result — three numbers, in priority order**

1. **Pooled OOS AUC** — 0.50 is no skill. Below ~0.53 the model has nothing and its
   vetoes are noise. This is the pass/fail.
2. **Expectancy at the decision threshold** — the money number. AUC can look
   respectable while expectancy stays negative. Compare the gated rows against the
   ungated baseline; if gating does not beat it, the model is not earning its keep.
3. **Calibration** — predicted vs actual should track. If it predicts 0.60 and delivers
   0.40, the probability gate is comparing against a breakeven number that means
   nothing, even with good AUC.

The gate enforces (1) and (2) automatically. Check (3) yourself — it is the one that
can be quietly wrong.

Also watch **fold AUC std dev**. One fold at 0.78 and the rest near 0.52 is a model
fitted to one regime, not an edge — more informative than the pooled number at this
sample size.

> Your effective sample is smaller than the row count: simultaneous longs across
> EUR/GBP/AUD/NZD are largely one USD bet. Treat a first-train AUC of 0.53–0.58 as
> encouraging but provisional.
""")

        # ── Retrain ─────────────────────────────────────────────────────────
        st.markdown("### Retrain")
        ready = len(training_rows) >= MIN_TRAIN_SAMPLES

        with st.expander("Advanced settings"):
            adv1, adv2, adv3 = st.columns(3)
            l2 = adv1.number_input("L2 penalty", 0.01, 100.0, 1.0, step=0.5,
                                   help="Higher = more shrinkage. Raise if coefficients "
                                        "swing wildly between folds.")
            folds = adv2.number_input("Walk-forward folds", 2, 12, 5, step=1)
            min_auc = adv3.number_input("Minimum AUC to pass", 0.50, 0.80, 0.53, step=0.01)
            notes = st.text_input("Note stored with the model", "",
                                  placeholder="e.g. after widening stops")

        if st.button("Run Retrain", type="primary", disabled=not ready,
                     help=None if ready else
                     f"Needs {MIN_TRAIN_SAMPLES} resolved trades; you have {len(training_rows)}."):
            with st.spinner("Walk-forward evaluating and fitting…"):
                try:
                    report = evaluate_and_fit(storage, l2=float(l2), folds=int(folds),
                                              min_auc=float(min_auc))
                    if report["error"]:
                        st.session_state.model_report = None
                        st.error(report["error"])
                    else:
                        # Saved inactive — promotion is a separate, deliberate click.
                        new_id = save_candidate(storage, report, notes=notes)
                        st.session_state.model_report = {
                            "id": new_id,
                            "passes": report["passes"],
                            "summary": gate_summary(report),
                            "metrics": report["metrics"],
                            "gated": report["gated"],
                            "calibration": report["calibration"],
                            "folds": report["folds"],
                            "coefficients": report["coefficients"],
                            "baseline": report["baseline_expectancy_r"],
                            "fold_auc_std": report["fold_auc_std"],
                        }
                except Exception as exc:
                    st.session_state.model_report = None
                    st.error(f"Retrain failed: {exc}")

        # ── Candidate review + promote ──────────────────────────────────────
        rep = st.session_state.get("model_report")
        if rep:
            st.markdown("---")
            st.markdown(f"### Candidate model #{rep['id']}")
            if rep["passes"]:
                st.success(rep["summary"])
            else:
                st.error(rep["summary"])

            m = rep["metrics"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("OOS AUC", f"{m['auc']:.4f}" if m["auc"] is not None else "—",
                      delta=f"{m['auc'] - 0.5:+.4f} vs no-skill" if m["auc"] is not None else None)
            r2.metric("Fold AUC std", f"{rep['fold_auc_std']:.4f}" if rep["fold_auc_std"] else "—")
            r3.metric("Top-decile prec",
                      f"{m['top_decile_prec']:.3f}" if m["top_decile_prec"] is not None else "—")
            r4.metric("Brier", f"{m['brier']:.4f}" if m["brier"] is not None else "—")

            st.markdown("**Expectancy at each decision threshold** "
                        f"(ungated baseline: `{rep['baseline']:+.4f}R`)")
            gdf = pd.DataFrame(rep["gated"])
            if not gdf.empty:
                gdf = gdf.rename(columns={
                    "threshold": "P(win) >=", "trades_taken": "Taken",
                    "trades_available": "Available", "selectivity": "Selectivity",
                    "expectancy_r": "Expectancy (R)", "total_r": "Total (R)",
                })
                keep = [c for c in ["P(win) >=", "Taken", "Available", "Selectivity",
                                    "Expectancy (R)", "Total (R)"] if c in gdf.columns]
                st.dataframe(gdf[keep], use_container_width=True, hide_index=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Calibration** (predicted vs actual)")
                cdf = pd.DataFrame(rep["calibration"])
                if not cdf.empty:
                    st.dataframe(cdf, use_container_width=True, hide_index=True)
                    st.caption("These two columns should track each other.")
            with cc2:
                st.markdown("**Per-fold stability**")
                fdf2 = pd.DataFrame(rep["folds"])
                if not fdf2.empty:
                    st.dataframe(fdf2, use_container_width=True, hide_index=True)
                    st.caption("One strong fold among weak ones = regime-fitted, not an edge.")

            with st.expander("Largest standardised coefficients (what carries the edge)"):
                st.dataframe(
                    pd.DataFrame(rep["coefficients"], columns=["feature", "coefficient"]),
                    use_container_width=True, hide_index=True,
                )

            st.markdown("#### Run in shadow (recommended first step)")
            st.caption(
                "Shadow scores every setup and logs the probability, but never vetoes. "
                "It is the only way to find out what the model would do to the trades it "
                "wants to block — once it is gating, those trades stop happening and stop "
                "being measurable. Check progress with `python scripts/model_report.py`."
            )
            if st.button(f"Run model #{rep['id']} in shadow"):
                if storage.shadow_model(rep["id"]):
                    st.success(
                        f"Model #{rep['id']} is now shadowing. Nothing is gated; "
                        f"probabilities are logged against every tracked trade."
                    )
                    st.session_state.model_report = None
                    st.rerun()
                else:
                    st.error("Could not set shadow — model id not found.")

            st.markdown("#### Promote")
            if rep["passes"]:
                st.caption("This model clears the gate. Promoting makes it veto live signals "
                           "on the next scan.")
                if st.button(f"Promote model #{rep['id']} to active", type="primary"):
                    if storage.activate_model(rep["id"]):
                        st.success(f"Model #{rep['id']} is now active.")
                        st.session_state.model_report = None
                        st.rerun()
                    else:
                        st.error("Could not activate — model id not found.")
            else:
                st.warning(
                    "This model failed the quality gate. Promoting it would let a model "
                    "with no demonstrated edge suppress real setups. The usual answer is "
                    "to collect more resolved trades and retrain."
                )
                override = st.checkbox("I understand, promote it anyway")
                if st.button(f"Force-promote model #{rep['id']}", disabled=not override):
                    if storage.activate_model(rep["id"]):
                        st.warning(f"Model #{rep['id']} force-promoted despite failing the gate.")
                        st.session_state.model_report = None
                        st.rerun()

        # ── History / rollback ──────────────────────────────────────────────
        if models:
            st.markdown("---")
            st.markdown("### Model history")
            st.dataframe(pd.DataFrame(models), use_container_width=True, hide_index=True)

            def _label(i: int) -> str:
                m = next((x for x in models if x["id"] == i), {})
                tag = " (active)" if m.get("is_active") else (
                    " (shadow)" if m.get("is_shadow") else "")
                return f"#{i}{tag}"

            h1, h2, h3 = st.columns([3, 1, 1])
            options = [m["id"] for m in models]
            pick = h1.selectbox("Select a model", options, format_func=_label)
            if h2.button("Activate", key="rollback_activate"):
                if storage.activate_model(int(pick)):
                    st.success(f"Model #{pick} is now active and gating.")
                    st.rerun()
            if h3.button("Shadow", key="rollback_shadow"):
                if storage.shadow_model(int(pick)):
                    st.success(f"Model #{pick} is now shadowing (logging only).")
                    st.rerun()

            if active and st.button("Disable model (revert to rules-only)"):
                storage.deactivate_all_models()
                st.info("All models deactivated — the scanner is rules-only again.")
                st.rerun()

            if any(m.get("is_shadow") for m in models) and st.button("Stop shadowing"):
                storage.clear_shadow_model()
                st.info("Shadow cleared — no model is scoring.")
                st.rerun()

        st.caption(
            "Equivalent CLI, if you prefer it: `python scripts/train_model.py` to evaluate, "
            "`--activate` to promote in one step."
        )

    # ── Scan Logs tab ─────────────────────────────────────────────────────────
    with tab_logs:
        logs = storage.load_scan_logs()
        if not logs:
            st.info("No scan logs yet.")
        else:
            ldf = pd.DataFrame(logs)
            st.dataframe(ldf, use_container_width=True, hide_index=True)

    # ── Settings tab ──────────────────────────────────────────────────────────
    # ── Reconcile tab ────────────────────────────────────────────────────────
    with tab_reconcile:
        _render_reconcile_tab(settings, storage)

    with tab_settings:
        last_run = storage.load_latest_scan_run()
        if last_run:
            st.json(last_run)
        else:
            st.info("No scan runs recorded yet.")
        st.subheader("Current Request")
        st.json({
            "pairs": selected_pairs,
            "max_spread_pips": max_spread,
            "signal_mode": signal_mode,
            "oanda_env": settings.oanda_env,
        })

        st.subheader("Daily ML Source (SQL Server)")
        st.json({
            "configured": settings.ml_source_configured,
            "server": settings.sql_server or "—",
            "database": settings.sql_database or "—",
            "table": settings.ml_predictions_table,
            "auth": "windows" if settings.sql_trusted_connection else "sql login",
            "min_sync_interval_minutes": settings.ml_sync_min_interval_minutes,
            "mirrored_through": storage.load_ml_max_prediction_date() or "—",
            "expected_latest_run": expected_prediction_date().isoformat(),
        })
        if st.button("Test SQL Server connection", key="ml_test_conn"):
            from forex.sqlserver import MLSourceError, test_connection
            try:
                probe = test_connection(settings)
                st.success("Connected to SQL Server")
                st.json(probe)
            except MLSourceError as exc:
                st.error(str(exc))

        sync_log = storage.load_ml_sync_log(limit=15)
        if sync_log:
            st.caption("Recent ML sync / reconcile attempts")
            st.dataframe(
                pd.DataFrame(sync_log)[
                    ["attempted_at", "kind", "status", "rows_synced",
                     "max_prediction_date", "message"]
                ],
                use_container_width=True, hide_index=True,
            )


# ── Live Quotes page ─────────────────────────────────────────────────────────

def _page_live_quotes(settings: AppSettings, storage: Storage, selected_pairs: list) -> None:
    st.title("Live Quotes")

    api_ok = bool(settings.oanda_api_key)

    # Auto-refresh every 30s on this page
    auto_count = st_autorefresh(interval=30_000, key="quotes_autorefresh")

    auto_due = (
        api_ok
        and bool(selected_pairs)
        and auto_count != st.session_state.quotes_auto_refresh_count_last
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        fetch_now = st.button("Fetch Quotes", disabled=not api_ok)
    with col2:
        session = current_session()
        badge = session_badge_color(session)
        st.markdown(f"**Session:** {badge} {session.replace('_', ' ')} — auto-refreshes every 30s")

    if fetch_now or auto_due:
        if api_ok and selected_pairs:
            from forex.oanda import OandaClient
            client = OandaClient(settings)
            try:
                quotes = client.get_pricing(selected_pairs)
                storage.save_quotes(quotes)
                st.session_state.quotes_auto_refresh_count_last = auto_count
            except Exception as exc:
                st.error(f"Failed to fetch quotes: {exc}")

    quotes = storage.load_latest_quotes()
    if not quotes:
        st.info("No live quotes yet. Click 'Fetch Quotes' or wait for auto-refresh.")
    else:
        qdf = pd.DataFrame(quotes)
        qdf["pair"] = qdf["pair"].apply(format_pair)
        qdf["mid"] = ((qdf["bid"] + qdf["ask"]) / 2).round(6)

        # Highlight London/NY overlap rows
        session = current_session()

        def _row_color(row):
            if session == "London_NY_Overlap":
                return ["background-color: #d8f3d8; color: #000000"] * len(row)
            elif session in ("London", "New_York"):
                return ["background-color: #d8e8f3; color: #000000"] * len(row)
            return [""] * len(row)

        display_cols = ["pair", "bid", "ask", "mid", "spread_pips", "as_of"]
        display = qdf[[c for c in display_cols if c in qdf.columns]]
        price_fmt = {c: "{:.5f}" for c in ["bid", "ask", "mid"] if c in display.columns}
        if "spread_pips" in display.columns:
            price_fmt["spread_pips"] = "{:.1f}"
        st.dataframe(
            display.style.apply(_row_color, axis=1).format(price_fmt),
            use_container_width=True,
            hide_index=True,
            height=_grid_height(len(display)),
        )

        st.caption(
            f"Session: {session.replace('_', ' ')} | "
            f"Market open: {'Yes' if is_forex_market_open() else 'No (weekend)'} | "
            f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key, selected_pairs, max_spread, signal_mode, auto_refresh, refresh_secs = _render_sidebar()

    settings = _build_settings()
    storage = Storage(settings.db_path)

    page = st.sidebar.radio(
        "Page",
        ["Forex Scanner", "Live Quotes"],
        label_visibility="collapsed",
    )

    if page == "Forex Scanner":
        _page_scanner(
            settings, storage, selected_pairs,
            max_spread, signal_mode, auto_refresh, refresh_secs,
        )
    else:
        _page_live_quotes(settings, storage, selected_pairs)


if __name__ == "__main__":
    main()
