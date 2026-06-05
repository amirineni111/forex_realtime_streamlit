from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .config import AppSettings
from .models import ForexSnapshot, ScanRequest, ScanSummary
from .oanda import OandaClient
from .storage import Storage
from .indicators import compute_all, calculate_session_hl
from .signals import score_pair
from .market_sessions import current_session


def _fetch_and_analyze(
    client: OandaClient,
    pair: str,
    request: ScanRequest,
) -> Tuple[Optional[ForexSnapshot], Optional[str]]:
    """Fetch candles for one pair, compute indicators + signals. Returns (snapshot, error)."""
    try:
        bars = client.get_candles(pair, granularity="M5", count=200)
        if not bars:
            return None, "No candle data returned"

        bar_dicts = [b.model_dump() for b in bars]
        indicators = compute_all(bar_dicts)
        if not indicators:
            return None, "Indicator computation returned empty"

        # Session high/low: use bars from current session start
        session = current_session()
        session_high, session_low = None, None
        if len(bar_dicts) >= 12:
            # Approximate session start as ~2 hours back (24 M5 bars = 2h)
            recent = bar_dicts[-48:]
            highs = [b["high"] for b in recent]
            lows = [b["low"] for b in recent]
            session_high = max(highs)
            session_low = min(lows)

        indicators["session_high"] = session_high
        indicators["session_low"] = session_low

        return None, None  # filled below (quote merged later)

    except Exception as exc:
        return None, str(exc)


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
                return pair, None, "No candle data"

            bar_dicts = [b.model_dump() for b in bars]
            indicators = compute_all(bar_dicts)

            # Session context: approx last 4 hours of M5 bars (48 bars)
            recent = bar_dicts[-48:] if len(bar_dicts) >= 48 else bar_dicts
            session_high = max(b["high"] for b in recent)
            session_low = min(b["low"] for b in recent)
            indicators["session_high"] = session_high
            indicators["session_low"] = session_low

            session = current_session()
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
                total_score=scoring.get("total_score", 0.0),
                trade_signal=scoring.get("trade_signal", "AVOID"),
                signal_reason=scoring.get("signal_reason", ""),
                risk_notes=scoring.get("risk_notes", ""),
                as_of=as_of,
            )
            return pair, snapshot, None

        except Exception as exc:
            return pair, None, str(exc)

    snapshots: List[ForexSnapshot] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_pair, pair): pair for pair in request.pairs}
        for future in as_completed(futures):
            pair, snapshot, error = future.result()
            if error:
                summary.errors += 1
                storage.log_pair(scan_id, pair, None, error)
            else:
                snapshots.append(snapshot)
                summary.pairs_scanned += 1
                storage.log_pair(scan_id, pair, snapshot.trade_signal, None)
                if snapshot.trade_signal not in ("AVOID", "WATCH_ONLY"):
                    summary.signals_found += 1

    # Sort by score descending
    snapshots.sort(key=lambda s: s.total_score, reverse=True)
    storage.save_snapshots(scan_id, snapshots)

    # Save live quotes too
    if quotes_by_pair:
        storage.save_quotes(list(quotes_by_pair.values()))

    storage.finish_scan(scan_id, summary)
    return summary
