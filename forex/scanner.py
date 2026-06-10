from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List

from .config import AppSettings
from .models import ForexSnapshot, ScanRequest, ScanSummary
from .oanda import OandaClient
from .storage import Storage
from .indicators import (
    compute_all,
    compute_trend_direction,
    detect_sr_levels,
    session_high_low,
)
from .signals import score_pair
from .market_sessions import current_session, current_session_start_utc
from .strength import calculate_strength, get_strength_for_pair, strength_bonus


def run_scan(
    settings: AppSettings,
    storage: Storage,
    request: ScanRequest,
) -> ScanSummary:
    client = OandaClient(settings)
    summary = ScanSummary()
    scan_id = storage.start_scan()

    # 1. Fetch live pricing for all pairs in one call
    quotes_by_pair: dict = {}
    try:
        quotes = client.get_pricing(request.pairs)
        quotes_by_pair = {q.pair: q for q in quotes}
    except Exception as exc:
        storage.log_pair(scan_id, "ALL", None, f"Pricing fetch failed: {exc}")

    # 2. Fetch candles per pair in parallel
    def _process_pair(pair: str):
        try:
            bars = client.get_candles(pair, granularity="M5", count=200)
            if not bars:
                return pair, None, "No candle data", []

            bar_dicts = [b.model_dump() for b in bars]
            indicators = compute_all(bar_dicts)

            session = current_session()

            # Session context: high/low since the active session opened. Falls back to a
            # rolling 4h (48 M5 bars) window during Off_Hours or when no session bars exist.
            session_high, session_low = session_high_low(
                bar_dicts, current_session_start_utc()
            )
            if session_high is None or session_low is None:
                recent = bar_dicts[-48:] if len(bar_dicts) >= 48 else bar_dicts
                session_high = max(b["high"] for b in recent)
                session_low = min(b["low"] for b in recent)
            indicators["session_high"] = session_high
            indicators["session_low"] = session_low

            # Fetch H1/H4 bars for MTF and S/R analysis (optional — fail gracefully)
            h1_bar_dicts: List[dict] = []
            h4_bar_dicts: List[dict] = []
            try:
                h1_bars = client.get_candles(pair, granularity="H1", count=100)
                h1_bar_dicts = [b.model_dump() for b in h1_bars]
            except Exception:
                pass
            try:
                h4_bars = client.get_candles(pair, granularity="H4", count=60)
                h4_bar_dicts = [b.model_dump() for b in h4_bars]
            except Exception:
                pass

            h1_direction = compute_trend_direction(h1_bar_dicts) if h1_bar_dicts else None
            h4_direction = compute_trend_direction(h4_bar_dicts) if h4_bar_dicts else None

            # Replace the noisy 5-min "day change" with a real ~24h change off H1 closes.
            # This is what drives the currency-strength matrix, so the anchor matters.
            if len(h1_bar_dicts) >= 25 and indicators.get("close"):
                ref = h1_bar_dicts[-25]["close"]  # ~24 completed hours back
                if ref:
                    indicators["day_change_pct"] = round(
                        (indicators["close"] - ref) / ref * 100, 3
                    )

            # S/R levels from H4 (longer-term structure) + H1 (shorter-term)
            sr_levels: List[dict] = []
            if h4_bar_dicts:
                sr_levels += detect_sr_levels(h4_bar_dicts, lookback=50)
            if h1_bar_dicts:
                sr_levels += detect_sr_levels(h1_bar_dicts, lookback=30)
            sr_levels.sort(key=lambda x: x["strength"], reverse=True)

            quote = quotes_by_pair.get(pair)
            bid = quote.bid if quote else None
            ask = quote.ask if quote else None
            mid = round((bid + ask) / 2, 6) if bid and ask else indicators.get("close")
            spread_pips = quote.spread_pips if quote else None
            as_of = quote.as_of if quote else datetime.now(timezone.utc).isoformat()

            scoring = score_pair(
                pair=pair,
                bid=bid,
                ask=ask,
                spread_pips=spread_pips,
                indicators=indicators,
                session=session,
                max_spread_pips=request.max_spread_pips,
                h1_direction=h1_direction,
                h4_direction=h4_direction,
                sr_levels=sr_levels,
            )

            snapshot = ForexSnapshot(
                pair=pair,
                bid=bid,
                ask=ask,
                mid=mid,
                spread_pips=spread_pips,
                open=indicators.get("open"),
                high=indicators.get("high"),
                low=indicators.get("low"),
                close=indicators.get("close"),
                day_change_pct=indicators.get("day_change_pct"),
                rsi14=indicators.get("rsi14"),
                ema9=indicators.get("ema9"),
                ema20=indicators.get("ema20"),
                ema50=indicators.get("ema50"),
                macd=indicators.get("macd"),
                macd_signal=indicators.get("macd_signal"),
                macd_histogram=indicators.get("macd_histogram"),
                atr14=indicators.get("atr14"),
                adx14=indicators.get("adx14"),
                bb_upper=indicators.get("bb_upper"),
                bb_middle=indicators.get("bb_middle"),
                bb_lower=indicators.get("bb_lower"),
                bb_width_pct=indicators.get("bb_width_pct"),
                current_session=scoring.get("current_session"),
                session_high=session_high,
                session_low=session_low,
                momentum_score=scoring.get("momentum_score", 0.0),
                reversion_score=scoring.get("reversion_score", 0.0),
                session_score=scoring.get("session_score", 0.0),
                regime=scoring.get("regime"),
                total_score=scoring.get("total_score", 0.0),
                trade_signal=scoring.get("trade_signal", "AVOID"),
                signal_reason=scoring.get("signal_reason", ""),
                risk_notes=scoring.get("risk_notes", ""),
                as_of=as_of,
                suggested_entry=scoring.get("suggested_entry"),
                suggested_stop=scoring.get("suggested_stop"),
                suggested_target=scoring.get("suggested_target"),
                stop_pips=scoring.get("stop_pips"),
                target_pips=scoring.get("target_pips"),
                rr_ratio=scoring.get("rr_ratio"),
                # MTF confluence
                h1_direction=h1_direction,
                h4_direction=h4_direction,
                mtf_score=scoring.get("mtf_score", 0.0),
                mtf_confluence=scoring.get("mtf_confluence"),
                # Support/Resistance
                nearest_support=scoring.get("nearest_support"),
                nearest_resistance=scoring.get("nearest_resistance"),
                sr_score=scoring.get("sr_score", 0.0),
                at_key_level=scoring.get("at_key_level", False),
                sr_levels_json=scoring.get("sr_levels_json"),
            )
            return pair, snapshot, None, bar_dicts

        except Exception as exc:
            return pair, None, str(exc), []

    snapshots: List[ForexSnapshot] = []
    bars_by_pair: dict = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_pair, pair): pair for pair in request.pairs}
        for future in as_completed(futures):
            pair, snapshot, error, m5_bars = future.result()
            if error:
                summary.errors += 1
                storage.log_pair(scan_id, pair, None, error)
            else:
                snapshots.append(snapshot)
                bars_by_pair[pair] = m5_bars
                summary.pairs_scanned += 1
                storage.log_pair(scan_id, pair, snapshot.trade_signal, None)
                if snapshot.trade_signal not in ("AVOID", "WATCH_ONLY"):
                    summary.signals_found += 1

    # Forward-evaluate previously-tracked signals against this scan's fresh bars, then
    # arm any new actionable signals. All DB writes stay single-threaded here.
    _ACTIONABLE = ("STRONG_BUY", "BUY_CANDIDATE", "STRONG_SHORT", "SHORT_CANDIDATE")
    for s in snapshots:
        pair_bars = bars_by_pair.get(s.pair) or []
        if pair_bars:
            try:
                storage.evaluate_tracked_signals(s.pair, pair_bars)
            except Exception as exc:
                storage.log_pair(scan_id, s.pair, None, f"Tracking eval failed: {exc}")
        if (
            s.trade_signal in _ACTIONABLE
            and s.suggested_stop is not None
            and s.suggested_target is not None
            and pair_bars
        ):
            direction = -1 if "SHORT" in s.trade_signal else 1
            try:
                storage.record_tracked_signal(
                    pair=s.pair,
                    signal=s.trade_signal,
                    direction=direction,
                    entry=s.suggested_entry,
                    stop=s.suggested_stop,
                    target=s.suggested_target,
                    stop_pips=s.stop_pips or 0.0,
                    target_pips=s.target_pips or 0.0,
                    atr14=s.atr14 or 0.0,
                    entry_ts=pair_bars[-1]["timestamp"],
                )
            except Exception as exc:
                storage.log_pair(scan_id, s.pair, None, f"Tracking record failed: {exc}")

    # Post-scan: compute currency strength and adjust scores
    if snapshots:
        strength_scores = calculate_strength([s.model_dump() for s in snapshots])
        for s in snapshots:
            base_str, quote_str, assessment = get_strength_for_pair(s.pair, strength_scores)
            s.base_strength = base_str
            s.quote_strength = quote_str
            s.strength_assessment = assessment
            bonus = strength_bonus(assessment, s.trade_signal)
            s.total_score = round(s.total_score + bonus, 1)

    # Sort by score descending (after strength adjustment)
    snapshots.sort(key=lambda s: s.total_score, reverse=True)
    storage.save_snapshots(scan_id, snapshots)

    # Save live quotes too
    if quotes_by_pair:
        storage.save_quotes(list(quotes_by_pair.values()))

    storage.finish_scan(scan_id, summary)
    return summary
