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
from .features import build_features, FEATURE_VERSION
from .market_sessions import current_session, current_session_start_utc
from .strength import calculate_strength, get_strength_for_pair, strength_bonus


def _apply_scoring(s: ForexSnapshot, scoring: dict) -> None:
    """Copy a score_pair result onto a snapshot (used for the final scoring pass)."""
    for field in (
        "momentum_score", "reversion_score", "session_score", "regime",
        "total_score", "trade_signal", "signal_reason", "risk_notes",
        "suggested_entry", "suggested_stop", "suggested_target",
        "stop_pips", "target_pips", "rr_ratio",
        "mtf_score", "mtf_confluence", "sr_score", "at_key_level",
        "nearest_support", "nearest_resistance", "sr_levels_json",
        "blocked_ahead", "cost_ratio", "model_prob", "required_prob",
    ):
        if field in scoring:
            setattr(s, field, scoring[field])


def _load_model(storage: Storage):
    """Active direction model, or None when the loop has not been trained yet."""
    try:
        payload = storage.load_active_model_json()
        if not payload:
            return None
        from .model import ForexModel
        model = ForexModel.from_json(payload)
        # A model trained on a different feature contract would be silently
        # misaligned — refuse it rather than serve garbage probabilities.
        if model.feature_version != FEATURE_VERSION:
            return None
        return model
    except Exception:
        return None


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

            # Rules-only first pass. The model probability and the currency-strength
            # bonus both need information that is only available once every pair has
            # been fetched, so the final score is recomputed in the sequential phase
            # below with those inputs supplied.
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
            ctx = {
                "bid": bid, "ask": ask, "spread_pips": spread_pips,
                "indicators": indicators, "session": session,
                "h1_direction": h1_direction, "h4_direction": h4_direction,
                "sr_levels": sr_levels, "as_of": as_of,
            }

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
                blocked_ahead=scoring.get("blocked_ahead", False),
                cost_ratio=scoring.get("cost_ratio"),
            )
            return pair, snapshot, None, bar_dicts, ctx

        except Exception as exc:
            return pair, None, str(exc), [], None

    snapshots: List[ForexSnapshot] = []
    bars_by_pair: dict = {}
    ctx_by_pair: dict = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_pair, pair): pair for pair in request.pairs}
        for future in as_completed(futures):
            pair, snapshot, error, m5_bars, ctx = future.result()
            if error:
                summary.errors += 1
                storage.log_pair(scan_id, pair, None, error)
            else:
                snapshots.append(snapshot)
                bars_by_pair[pair] = m5_bars
                ctx_by_pair[pair] = ctx
                summary.pairs_scanned += 1

    # ── Sequential phase ────────────────────────────────────────────────────
    # Currency strength needs every pair's day change, and the model needs currency
    # strength, so both run here rather than inside the parallel fetch. All DB writes
    # are single-threaded in this block.
    model = _load_model(storage)

    strength_scores = calculate_strength([s.model_dump() for s in snapshots]) if snapshots else {}
    for s in snapshots:
        base_str, quote_str, assessment = get_strength_for_pair(s.pair, strength_scores)
        s.base_strength = base_str
        s.quote_strength = quote_str
        s.strength_assessment = assessment

    # Final scoring pass: strength bonus folded into total_score, then the model
    # probability applied as a veto on anything the rules proposed.
    features_by_pair: dict = {}
    for s in snapshots:
        ctx = ctx_by_pair.get(s.pair)
        if not ctx:
            continue

        def _rescore(prob, bonus):
            return score_pair(
                pair=s.pair, bid=ctx["bid"], ask=ctx["ask"],
                spread_pips=ctx["spread_pips"], indicators=ctx["indicators"],
                session=ctx["session"], max_spread_pips=request.max_spread_pips,
                h1_direction=ctx["h1_direction"], h4_direction=ctx["h4_direction"],
                sr_levels=ctx["sr_levels"], model_prob=prob, strength_bonus=bonus,
            )

        # Strength alignment is judged against the raw directional read, not against
        # the first-pass label — a countertrend setup downgraded to WATCH_ONLY still
        # has a direction, and keying off the label would silently zero the bonus.
        base = _rescore(None, 0.0)
        dom = base.get("dominant")
        bonus = strength_bonus(
            s.strength_assessment or "NEUTRAL",
            "STRONG_BUY" if dom == "LONG" else ("STRONG_SHORT" if dom == "SHORT" else "WATCH_ONLY"),
        )
        scoring = _rescore(None, bonus) if bonus else base

        # Features are built for every directional setup whether or not a model
        # exists yet. Building them only when a model was loaded would deadlock the
        # loop: no model means no logged features, which means no training data,
        # which means a model can never be trained.
        if scoring.get("dominant") in ("LONG", "SHORT"):
            direction = 1 if scoring["dominant"] == "LONG" else -1
            feat_snap = {
                **ctx["indicators"], **scoring,
                "spread_pips": ctx["spread_pips"],
                "stop_pips": scoring.get("prov_stop_pips"),
                "h1_direction": ctx["h1_direction"],
                "h4_direction": ctx["h4_direction"],
                "current_session": ctx["session"],
                "as_of": ctx["as_of"],
                "base_strength": s.base_strength,
                "quote_strength": s.quote_strength,
            }
            feats = build_features(feat_snap, direction)
            features_by_pair[s.pair] = feats

            if model is not None:
                try:
                    model_prob = round(model.predict_proba(feats), 4)
                except Exception as exc:
                    storage.log_pair(scan_id, s.pair, None, f"Model scoring failed: {exc}")
                    model_prob = None
                if model_prob is not None:
                    scoring = _rescore(model_prob, bonus)

        _apply_scoring(s, scoring)

    # Forward-evaluate previously-tracked signals against this scan's fresh bars,
    # then arm any new actionable signals.
    _ACTIONABLE = ("STRONG_BUY", "BUY_CANDIDATE", "STRONG_SHORT", "SHORT_CANDIDATE")
    for s in snapshots:
        pair_bars = bars_by_pair.get(s.pair) or []
        if pair_bars:
            try:
                storage.evaluate_tracked_signals(s.pair, pair_bars)
            except Exception as exc:
                storage.log_pair(scan_id, s.pair, None, f"Tracking eval failed: {exc}")

        storage.log_pair(scan_id, s.pair, s.trade_signal, None)
        if s.trade_signal not in ("AVOID", "WATCH_ONLY"):
            summary.signals_found += 1

        # Thin-edge gate: don't forward-test signals whose target cannot clear the
        # round-trip cost by a sensible margin. The old 3× bar still left a third of
        # the target being paid away in spread; the cost_ratio veto in score_pair now
        # carries most of this, and 6× is the belt-and-braces check on the target side.
        thin_edge = (
            s.spread_pips is not None
            and s.target_pips is not None
            and s.target_pips < s.spread_pips * 6
        )
        if thin_edge and s.trade_signal in _ACTIONABLE:
            storage.log_pair(scan_id, s.pair, None,
                             f"Skipped tracking: target {s.target_pips}p < 6× spread")
        if (
            s.trade_signal in _ACTIONABLE
            and s.suggested_stop is not None
            and s.suggested_target is not None
            and pair_bars
            and not thin_edge
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
                    # The feature vector as it stood when the trade was armed. This is
                    # the row the next retrain learns from.
                    features=features_by_pair.get(s.pair),
                    feature_version=FEATURE_VERSION,
                    model_prob=s.model_prob,
                    required_prob=s.required_prob,
                    cost_ratio=s.cost_ratio,
                    spread_pips=s.spread_pips,
                    total_score=s.total_score,
                    adx14=s.adx14,
                    regime=s.regime,
                    session=s.current_session,
                )
            except Exception as exc:
                storage.log_pair(scan_id, s.pair, None, f"Tracking record failed: {exc}")

    # Sort by score descending (after strength adjustment)
    snapshots.sort(key=lambda s: s.total_score, reverse=True)
    storage.save_snapshots(scan_id, snapshots)

    # Save live quotes too
    if quotes_by_pair:
        storage.save_quotes(list(quotes_by_pair.values()))

    storage.finish_scan(scan_id, summary)
    return summary
