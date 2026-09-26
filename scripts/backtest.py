"""
Timeframe-agnostic signal backtest.

What this is for
----------------
Everything else in this repo measures trades that were *already taken*. That can
tell you the scorer has no edge, but it cannot answer "would a different signal
timeframe have an edge", because no such trades exist. This harness generates
signals from scratch over historical bars, so a change to the signal timeframe can
be judged before it is shipped rather than after three months of live losses.

The question it was built to answer: the live system signals on M5, where ATR is
2-4 pips, while the spread needs a 20+ pip stop to amortise. Signal horizon and
bracket horizon are mismatched by 5-10x. Does moving the signal to M15 or H1 fix it?

Method
------
For each bar t (walking forward, never looking past it):

  1. indicators are computed on bars[:t+1] only — the same ``compute_all`` the live
     scanner uses, so there is no separate backtest implementation to drift;
  2. higher-timeframe direction and S/R come from HTF bars filtered to t, so MTF
     confluence is available without leaking the future;
  3. ``score_pair`` produces the signal, exactly as live;
  4. the entry-quality gate runs on the same ``build_features`` vector as live;
  5. an actionable, gate-passing setup arms a bracket at the bar close;
  6. the bracket resolves against subsequent bars, stop checked before target
     within a bar (the conservative tie-break ``Storage.evaluate_tracked_signals``
     uses), timing out at ``--max-hold`` bars.

Costs are charged as one full spread per round trip, from a per-pair spread table
measured from live quotes — candle data is mid, so a mid-to-mid bracket still pays
the spread getting in and out.

Two deliberate limitations, both stated rather than hidden:
  * spreads are a per-pair constant, not the spread that prevailed at that hour, so
    illiquid-hour costs are understated;
  * re-arm cooldown is applied per pair+direction, matching live, but position
    sizing and correlation across pairs are ignored — expectancy is per-trade R.

Usage:
    python scripts/backtest.py --granularity H1 M15 M5
    python scripts/backtest.py --granularity H1 --no-gate     # gate off, for contrast
    python scripts/backtest.py --granularity H1 --by-pair
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forex import quality as quality_gate
from forex.features import build_features
from forex.indicators import compute_all, compute_trend_direction, detect_sr_levels
from forex.market_sessions import current_session
from forex.signals import score_pair
from forex.timeutil import parse_ts

CACHE = Path("data/bar_cache")

# Median spread per pair, measured from live OANDA quotes on the tracked history.
SPREADS = {
    "AUD_USD": 1.3, "EUR_GBP": 1.3, "EUR_USD": 1.6, "USD_CHF": 1.6, "USD_JPY": 1.6,
    "USD_CAD": 1.8, "GBP_USD": 1.8, "EUR_CHF": 2.0, "NZD_USD": 2.3, "EUR_JPY": 2.7,
    "USD_SGD": 3.1, "GBP_JPY": 3.4, "AUD_NZD": 3.5, "USD_HKD": 3.5,
}

# Bars of higher-timeframe context per granularity, and how many lower-timeframe
# bars make one higher-timeframe bar.
HTF_OF = {"M5": ("H1", 12), "M15": ("H1", 4), "H1": ("H4", 4)}
HTF2_OF = {"M5": ("H4", 48), "M15": ("H4", 16), "H1": ("D", 24)}

WARMUP = 200          # bars of history before the first signal
SESSION_LOOKBACK = {"M5": 48, "M15": 16, "H1": 8}


def pip_value(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def load_bars(pair: str, gran: str) -> List[dict]:
    f = CACHE / f"{pair}_{gran}.json"
    if not f.exists():
        return []
    return json.loads(f.read_text()).get("bars", [])


def resample(bars: List[dict], factor: int) -> List[dict]:
    """
    Aggregate ``factor`` consecutive bars into one.

    Used to synthesise higher-timeframe context from the granularity already on
    disk, so the harness needs one cached series per timeframe rather than four.
    Bars are aligned to the *end* of each group, which is what keeps the last
    aggregated bar closed as of the current moment.
    """
    out = []
    for i in range(0, len(bars) - factor + 1, factor):
        chunk = bars[i:i + factor]
        out.append({
            "timestamp": chunk[-1]["timestamp"],
            "open": chunk[0].get("open", chunk[0]["close"]),
            "high": max(b["high"] for b in chunk),
            "low": min(b["low"] for b in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(b.get("volume", 0) for b in chunk),
        })
    return out


def backtest_pair(
    pair: str, gran: str, use_gate: bool, max_hold_bars: int,
    cooldown_bars: int, step: int,
) -> List[dict]:
    bars = load_bars(pair, gran)
    if len(bars) < WARMUP + 50:
        return []

    spread = SPREADS.get(pair, 2.0)
    pip = pip_value(pair)
    htf_gran, htf_factor = HTF_OF[gran]
    htf2_gran, htf2_factor = HTF2_OF[gran]
    sess_lb = SESSION_LOOKBACK[gran]

    trades: List[dict] = []
    # Last bar index at which this pair+direction armed, for the re-arm cooldown.
    last_armed: Dict[int, int] = {}

    for t in range(WARMUP, len(bars) - 1, step):
        window = bars[max(0, t - WARMUP + 1):t + 1]
        ind = compute_all(window)
        if not ind.get("close") or not ind.get("atr14"):
            continue

        recent = window[-sess_lb:]
        ind["session_high"] = max(b["high"] for b in recent)
        ind["session_low"] = min(b["low"] for b in recent)

        # Higher-timeframe context, built only from bars at or before t.
        htf = resample(window, htf_factor)
        htf2 = resample(window, htf2_factor)
        h1_dir = compute_trend_direction(htf) if len(htf) >= 20 else None
        h4_dir = compute_trend_direction(htf2) if len(htf2) >= 20 else None

        sr_levels = []
        if len(htf2) >= 20:
            sr_levels += detect_sr_levels(htf2, lookback=50)
        if len(htf) >= 20:
            sr_levels += detect_sr_levels(htf, lookback=30)
        sr_levels.sort(key=lambda x: x["strength"], reverse=True)

        ts = bars[t]["timestamp"]
        dt = parse_ts(ts)
        hour = dt.hour if dt else 12
        session = current_session(dt) if dt else "London"

        mid = ind["close"]
        half = spread * pip / 2
        scoring = score_pair(
            pair=pair, bid=mid - half, ask=mid + half, spread_pips=spread,
            indicators=ind, session=session, max_spread_pips=10.0,
            h1_direction=h1_dir, h4_direction=h4_dir, sr_levels=sr_levels,
        )
        dom = scoring.get("dominant")
        if dom not in ("LONG", "SHORT"):
            continue

        direction = 1 if dom == "LONG" else -1
        if t - last_armed.get(direction, -10**9) < cooldown_bars:
            continue

        feats = build_features({
            **ind, **scoring, "spread_pips": spread,
            "stop_pips": scoring.get("prov_stop_pips"),
            "h1_direction": h1_dir, "h4_direction": h4_dir,
            "current_session": session, "as_of": ts,
        }, direction)

        verdict = quality_gate.evaluate(
            feats, pair=pair, hour_utc=hour, cost_ratio=scoring.get("cost_ratio"),
        )
        signal = scoring.get("trade_signal")
        # With the gate off, judge the signal the rules alone produced.
        if use_gate:
            actionable = signal in ("STRONG_BUY", "BUY_CANDIDATE",
                                    "STRONG_SHORT", "SHORT_CANDIDATE") and verdict.passed
        else:
            actionable = (scoring.get("total_score", 0) >= 45
                          and not scoring.get("blocked_ahead"))
        if not actionable:
            continue

        entry = scoring.get("prov_entry") or mid
        stop = scoring.get("prov_stop")
        target = scoring.get("prov_target")
        stop_pips = scoring.get("prov_stop_pips")
        if not stop or not target or not stop_pips:
            continue

        # Resolve forward. Stop is checked before target within a bar.
        exit_px, reason = None, None
        for j in range(t + 1, min(t + 1 + max_hold_bars, len(bars))):
            b = bars[j]
            if direction == 1:
                if b["low"] <= stop:
                    exit_px, reason = stop, "STOP"; break
                if b["high"] >= target:
                    exit_px, reason = target, "TARGET"; break
            else:
                if b["high"] >= stop:
                    exit_px, reason = stop, "STOP"; break
                if b["low"] <= target:
                    exit_px, reason = target, "TARGET"; break
        if exit_px is None:
            last = min(t + max_hold_bars, len(bars) - 1)
            exit_px, reason = bars[last]["close"], "TIMEOUT"

        gross = (exit_px - entry) * direction / pip
        net = gross - spread
        last_armed[direction] = t
        trades.append({
            "pair": pair, "ts": ts, "hour": hour, "direction": direction,
            "signal": signal, "reason": reason, "gross_pips": gross,
            "net_pips": net, "stop_pips": stop_pips,
            "R": net / stop_pips, "gross_R": gross / stop_pips,
            "cost_R": spread / stop_pips, "win": 1 if net > 0 else 0,
            "gate_passed": verdict.passed, "score": scoring.get("total_score"),
        })

    return trades


def report(trades: List[dict], label: str, by_pair: bool = False) -> None:
    if not trades:
        print(f"  {label:<26} no trades")
        return
    n = len(trades)
    wr = sum(t["win"] for t in trades) / n
    rs = [t["R"] for t in trades]
    exp = sum(rs) / n
    gross = sum(t["gross_R"] for t in trades) / n
    cost = sum(t["cost_R"] for t in trades) / n
    total = sum(rs)
    med_stop = sorted(t["stop_pips"] for t in trades)[n // 2]
    # Standard error of the mean R, and the t-statistic against zero expectancy.
    # Printed on every line because a backtest expectancy without it invites
    # reading noise as edge — at these sample sizes |t| below ~2 means the result
    # is not distinguishable from break-even, however good the headline looks.
    var = sum((r - exp) ** 2 for r in rs) / (n - 1) if n > 1 else 0.0
    se = (var / n) ** 0.5 if n > 1 else float("inf")
    tstat = exp / se if se else 0.0
    print(f"  {label:<26} n={n:5d}  WR={wr:5.1%}  E={exp:+.4f}R  "
          f"gross={gross:+.4f}  cost={cost:.4f}  total={total:+8.1f}R  "
          f"stop~{med_stop:.0f}p  t={tstat:+.2f}")
    if by_pair:
        pairs = sorted({t["pair"] for t in trades})
        for p in pairs:
            sub = [t for t in trades if t["pair"] == p]
            if len(sub) < 5:
                continue
            swr = sum(x["win"] for x in sub) / len(sub)
            sexp = sum(x["R"] for x in sub) / len(sub)
            print(f"      {p:<10} n={len(sub):4d}  WR={swr:5.1%}  E={sexp:+.4f}R")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--granularity", nargs="+", default=["H1", "M15", "M5"])
    ap.add_argument("--pairs", nargs="+", default=list(SPREADS))
    ap.add_argument("--max-hold", type=int, default=24,
                    help="bars to hold before timing out (default 24)")
    ap.add_argument("--cooldown", type=int, default=9,
                    help="bars before the same pair+direction can re-arm")
    ap.add_argument("--step", type=int, default=1,
                    help="evaluate every Nth bar (speed/coverage trade-off)")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the entry-quality gate, to isolate its contribution")
    ap.add_argument("--allow-shorts", action="store_true",
                    help="lift _SUPPRESS_SHORT_CANDIDATE. That suppression was derived "
                         "from M5 outcomes, so leaving it on would silently answer the "
                         "timeframe question for shorts before it is asked.")
    ap.add_argument("--by-pair", action="store_true")
    ap.add_argument("--by-hour", action="store_true",
                    help="break results down by UTC hour — validates EXCLUDED_HOURS")
    ap.add_argument("--by-month", action="store_true",
                    help="break results down by month — the decay check")
    ap.add_argument("--stop-atr-mult", type=float,
                    help="override _STOP_ATR_MULT (bracket scale sweep)")
    ap.add_argument("--max-stop-atr-mult", type=float,
                    help="override _MAX_STOP_ATR_MULT (how far cost may widen the stop)")
    ap.add_argument("--rr", type=float, help="override _RR")
    ap.add_argument("--max-cost-ratio", type=float,
                    help="override the gate's MAX_COST_RATIO")
    args = ap.parse_args()

    from forex import signals as _sig
    if args.allow_shorts:
        _sig._SUPPRESS_SHORT_CANDIDATE = False
    if args.stop_atr_mult:
        _sig._STOP_ATR_MULT = args.stop_atr_mult
    if args.max_stop_atr_mult:
        _sig._MAX_STOP_ATR_MULT = args.max_stop_atr_mult
    if args.rr:
        _sig._RR = args.rr
    if args.max_cost_ratio:
        _sig._MAX_COST_RATIO = args.max_cost_ratio
        quality_gate.MAX_COST_RATIO = args.max_cost_ratio

    print("=" * 96)
    print("SIGNAL BACKTEST — signals generated from scratch, walking forward")
    print(f"max hold {args.max_hold} bars | re-arm cooldown {args.cooldown} bars | "
          f"step {args.step} | gate {'OFF' if args.no_gate else 'ON'}")
    print("=" * 96)

    for gran in args.granularity:
        all_trades: List[dict] = []
        for pair in args.pairs:
            all_trades += backtest_pair(
                pair, gran, use_gate=not args.no_gate,
                max_hold_bars=args.max_hold, cooldown_bars=args.cooldown,
                step=args.step,
            )
        print(f"\n{gran}:")
        report(all_trades, "all", by_pair=args.by_pair)
        if args.by_hour and all_trades:
            for hr in sorted({t["hour"] for t in all_trades}):
                report([t for t in all_trades if t["hour"] == hr], f"  {hr:02d}:00 UTC")
        if args.by_month and all_trades:
            for month in sorted({t["ts"][:7] for t in all_trades}):
                report([t for t in all_trades if t["ts"][:7] == month], f"  {month}")
        if all_trades:
            for label, sub in (
                ("longs", [t for t in all_trades if t["direction"] == 1]),
                ("shorts", [t for t in all_trades if t["direction"] == -1]),
            ):
                report(sub, label)
    print("\n" + "=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
