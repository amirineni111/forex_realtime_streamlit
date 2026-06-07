from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from forex.config import AppSettings, get_settings
from forex.market_sessions import current_session, is_forex_market_open, session_badge_color
from forex.models import ScanRequest
from forex.pairs import UNIVERSE_MAP, format_pair
from forex.scanner import run_scan
from forex.storage import Storage
from forex.strength import calculate_strength, CURRENCIES

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


# ── Session state init ───────────────────────────────────────────────────────

def _init_state() -> None:
    prefs = _load_prefs()
    defaults = {
        "oanda_api_key": prefs.get("oanda_api_key", ""),
        "oanda_account_id": prefs.get("oanda_account_id", ""),
        "oanda_env": prefs.get("oanda_env", "practice"),
        "auto_refresh": prefs.get("auto_refresh", False),
        "refresh_seconds": prefs.get("refresh_seconds", 60),
        "auto_refresh_count_last": 0,
        "quotes_auto_refresh_count_last": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Settings builder ─────────────────────────────────────────────────────────

def _build_settings() -> AppSettings:
    env_settings = get_settings()
    return AppSettings(
        oanda_api_key=st.session_state.oanda_api_key or env_settings.oanda_api_key,
        oanda_account_id=st.session_state.oanda_account_id or env_settings.oanda_account_id,
        oanda_env=st.session_state.oanda_env,
        db_path=env_settings.db_path,
    )


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
    universe_choice = st.sidebar.radio(
        "Universe",
        list(UNIVERSE_MAP.keys()) + ["Custom"],
        horizontal=False,
    )
    if universe_choice == "Custom":
        custom_raw = st.sidebar.text_area(
            "Custom pairs (one per line, e.g. EUR_USD)",
            height=120,
            placeholder="EUR_USD\nGBP_USD\nUSD_JPY",
        )
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

    # Save prefs
    _save_prefs({
        "oanda_api_key": api_key,
        "oanda_account_id": account_id,
        "oanda_env": oanda_env,
        "auto_refresh": auto_refresh,
        "refresh_seconds": refresh_secs,
    })

    return api_key, selected_pairs, max_spread, signal_mode, auto_refresh, refresh_secs


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
        )
        with st.spinner(f"Scanning {len(selected_pairs)} pairs…"):
            try:
                summary = run_scan(settings, storage, request)
                st.session_state.auto_refresh_count_last = auto_count or 0
                st.success(
                    f"Scan complete — {summary.pairs_scanned} pairs, "
                    f"{summary.signals_found} signals, {summary.errors} errors"
                )
            except Exception as exc:
                st.error(f"Scan failed: {exc}")

    tab_results, tab_watchlist, tab_perf, tab_logs, tab_settings = st.tabs(
        ["Results", "Watchlist", "Performance", "Scan Logs", "Settings"]
    )

    # ── Results tab ──────────────────────────────────────────────────────────
    with tab_results:
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
                st.caption("Strength normalized 0–100 from pair day_change_pct. Higher = relatively stronger currency.")

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
                "pair", "trade_signal", "total_score",
                "momentum_score", "reversion_score", "session_score",
                "mtf_score", "mtf_confluence", "h1_direction", "h4_direction",
                "sr_score", "at_key_level", "nearest_support", "nearest_resistance",
                "strength_assessment",
                "bid", "ask", "spread_pips",
                "close", "day_change_pct",
                "rsi14", "macd_histogram", "ema9", "ema20",
                "atr14", "bb_width_pct",
                "current_session", "signal_reason", "as_of",
            ]
            display = filtered[[c for c in display_cols if c in filtered.columns]].copy()
            display["pair"] = display["pair"].apply(format_pair)
            display["trade_signal"] = display["trade_signal"].apply(_signal_badge)

            price_cols = {c: "{:.5f}" for c in ["bid", "ask", "close", "ema9", "ema20", "nearest_support", "nearest_resistance"] if c in display.columns}

            def _highlight_key_level(row):
                if row.get("at_key_level"):
                    return ["background-color: #2a2a10"] * len(row)
                return [""] * len(row)

            styled = display.style.format(price_cols)
            if "at_key_level" in display.columns:
                styled = styled.apply(_highlight_key_level, axis=1)

            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Column guide
            with st.expander("Column Guide"):
                st.markdown("""
| Column | Description |
|--------|-------------|
| pair | Currency pair (e.g. EUR/USD) |
| trade_signal | Overall signal: STRONG_BUY, BUY_CANDIDATE, SHORT_CANDIDATE, STRONG_SHORT, WATCH_ONLY, AVOID |
| total_score | Combined score (base 0–100 + MTF 0–30 + S/R 0–25 + strength ±10) |
| momentum_score | EMA/MACD/RSI momentum component (0–40) |
| reversion_score | RSI extreme/Bollinger Band component (0–40) |
| session_score | Session quality/breakout component (0–20) |
| mtf_score | Multi-timeframe confluence bonus (0=NONE, 15=PARTIAL, 30=FULL) |
| mtf_confluence | FULL=all 3 TFs agree, PARTIAL=2 agree, NONE=conflict |
| h1_direction | H1 trend direction (LONG/SHORT/NEUTRAL) |
| h4_direction | H4 trend direction (LONG/SHORT/NEUTRAL) |
| sr_score | Support/Resistance proximity bonus (0–25) |
| at_key_level | Price is AT a key S/R level (highlighted row) |
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
            wdf["trade_signal"] = wdf["trade_signal"].apply(_signal_badge)
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
        outcomes = storage.load_trade_outcomes()
        if not outcomes:
            st.info("No trade outcomes yet. Close watchlist entries with an exit price to start tracking performance.")
        else:
            odf = pd.DataFrame(outcomes)

            # Overall metrics
            total_trades = len(odf)
            wins = (odf["outcome"] == "WIN").sum()
            win_rate = wins / total_trades if total_trades else 0
            r_vals = odf["r_multiple"].dropna()
            avg_r = r_vals.mean() if len(r_vals) else 0.0
            expectancy = avg_r * win_rate - (1 - win_rate) if len(r_vals) else 0.0

            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Total Trades", total_trades)
            pm2.metric("Win Rate", f"{win_rate*100:.1f}%")
            pm3.metric("Avg R:R", f"{avg_r:.2f}R")
            pm4.metric("Expectancy", f"{expectancy:.3f}R")

            # Equity curve (cumulative R)
            if "r_multiple" in odf.columns:
                st.subheader("Equity Curve (Cumulative R)")
                cum_r = odf.sort_values("created_at")["r_multiple"].fillna(0).cumsum().reset_index(drop=True)
                st.line_chart(cum_r)

            # Win rate by pair
            pair_stats = storage.load_performance_by_dimension("pair")
            if pair_stats:
                st.subheader("Win Rate by Pair")
                pair_df = pd.DataFrame(pair_stats)[["dimension_value", "trades", "wins", "win_rate", "avg_r", "expectancy"]]
                pair_df.columns = ["Pair", "Trades", "Wins", "Win Rate", "Avg R", "Expectancy"]
                pair_df["Win Rate"] = pair_df["Win Rate"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(pair_df, use_container_width=True, hide_index=True)

            # Win rate by signal type
            sig_stats = storage.load_performance_by_dimension("signal")
            if sig_stats:
                st.subheader("Win Rate by Signal")
                sig_df = pd.DataFrame(sig_stats)[["dimension_value", "trades", "wins", "win_rate", "avg_r", "expectancy"]]
                sig_df.columns = ["Signal", "Trades", "Wins", "Win Rate", "Avg R", "Expectancy"]
                sig_df["Win Rate"] = sig_df["Win Rate"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(sig_df, use_container_width=True, hide_index=True)

            # Recent trade log
            with st.expander("Trade History"):
                odf["pair"] = odf["pair"].apply(format_pair)
                st.dataframe(odf, use_container_width=True, hide_index=True)

    # ── Scan Logs tab ─────────────────────────────────────────────────────────
    with tab_logs:
        logs = storage.load_scan_logs()
        if not logs:
            st.info("No scan logs yet.")
        else:
            ldf = pd.DataFrame(logs)
            st.dataframe(ldf, use_container_width=True, hide_index=True)

    # ── Settings tab ──────────────────────────────────────────────────────────
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
                return ["background-color: #1a3a1a"] * len(row)
            elif session in ("London", "New_York"):
                return ["background-color: #1a2a3a"] * len(row)
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
