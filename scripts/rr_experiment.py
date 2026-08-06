"""
Replay every resolved trade at alternative reward:risk ratios.

The measured win rate (39.5% over 1121 trades) sits below the 45% needed to break
even at the live ``_RR = 1.5`` once the 12.5% average cost ratio is paid, but above
the 37.5% needed at RR 2.0. Whether widening the target actually helps cannot be
argued from that arithmetic alone — a farther target is hit less often, and how much
less is an empirical question about how far price actually runs after entry.

This script answers it by re-resolving each trade against real M5 bars, holding the
entry and stop fixed and moving only the target. Resolution semantics are copied from
``Storage.evaluate_tracked_signals``: stop checked before target within a bar,
12-hour timeout to the last close, and R computed from pips net of the round-trip
spread. RR 1.5 therefore reproduces the recorded outcomes and acts as the control.

Usage:
    python scripts/rr_experiment.py                 # all resolved trades
    python scripts/rr_experiment.py --refresh       # ignore the bar cache
    python scripts/rr_experiment.py --rr 1.5 2.0 3.0
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forex.config import get_settings
from forex.oanda import OandaClient
from forex.pairs import pip_value

MAX_HOLD_HOURS = 12.0
DEFAULT_RRS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
CACHE_DIR = Path("data/bar_cache")


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    """OANDA stamps carry 9 fractional digits, which fromisoformat rejects."""
    if not value:
        return None
    raw = value.strip().replace("Z", "+00:00")
    if "." in raw:
        head, _, tail = raw.partition(".")
        digits = "".join(c for c in tail if c.isdigit())[:6]
        offset = tail[len(tail) - 6:] if "+" in tail or "-" in tail else "+00:00"
        raw = f"{head}.{digits:0<6}{offset}"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_trades(db: Path) -> List[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        """
        SELECT t.id, t.pair, t.signal, t.direction, t.entry_price, t.stop_price,
               t.stop_pips, t.spread_pips, t.entry_ts, t.cost_ratio,
               o.outcome AS recorded_outcome, o.r_multiple AS recorded_r
        FROM forex_trade_outcomes o
        JOIN forex_signal_tracking t ON t.id = o.tracking_id
        WHERE t.entry_price > 0 AND t.stop_price > 0 AND t.stop_pips > 0
        ORDER BY t.entry_ts
        """
    )]
    conn.close()
    return rows


def fetch_bars(client: OandaClient, pair: str, start: datetime, end: datetime,
               refresh: bool) -> List[dict]:
    """M5 bars for one pair over the whole experiment window, cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{pair}_M5.json"
    if cache.exists() and not refresh:
        meta = json.loads(cache.read_text())
        if meta["start"] <= start.isoformat() and meta["end"] >= end.isoformat():
            return meta["bars"]
    bars = [
        {"timestamp": b.timestamp, "high": b.high, "low": b.low, "close": b.close}
        for b in client.get_candles_range(
            pair, start.isoformat().replace("+00:00", "Z"),
            end.isoformat().replace("+00:00", "Z"), "M5",
        )
    ]
    cache.write_text(json.dumps(
        {"start": start.isoformat(), "end": end.isoformat(), "bars": bars}
    ))
    return bars


def replay(trade: dict, bars: List[dict], rr: float) -> Optional[dict]:
    """
    Re-resolve one trade with the target moved to ``rr`` x risk.

    Returns None when the bar window does not cover the trade, so a gap in history
    drops the trade from every RR equally instead of biasing one of them.
    """
    entry_dt = parse_ts(trade["entry_ts"])
    if entry_dt is None:
        return None
    direction = trade["direction"] or 1
    entry = trade["entry_price"]
    stop = trade["stop_price"]
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    target = entry + direction * rr * risk
    deadline = entry_dt + timedelta(hours=MAX_HOLD_HOURS)

    window = [
        b for b in bars
        if (dt := parse_ts(b["timestamp"])) is not None and entry_dt < dt <= deadline
    ]
    if not window:
        return None

    pip = pip_value(trade["pair"])
    exit_price: Optional[float] = None
    reason = None
    mfe = 0.0
    for b in window:
        hi, lo = b["high"], b["low"]
        favorable = (hi - entry) if direction == 1 else (entry - lo)
        mfe = max(mfe, favorable)
        if direction == 1:
            if lo <= stop:                       # stop first = conservative
                exit_price, reason = stop, "STOP"
                break
            if hi >= target:
                exit_price, reason = target, "TARGET"
                break
        else:
            if hi >= stop:
                exit_price, reason = stop, "STOP"
                break
            if lo <= target:
                exit_price, reason = target, "TARGET"
                break
    if exit_price is None:
        exit_price, reason = window[-1]["close"], "TIMEOUT"

    gross_pips = (exit_price - entry) * direction / pip
    net_pips = gross_pips - (trade["spread_pips"] or 0.0)
    r = net_pips / trade["stop_pips"]
    return {
        "r": r,
        "net_pips": net_pips,
        "win": net_pips > 0,
        "reason": reason,
        "mfe_r": mfe / risk,
    }


def summarize(results: List[dict]) -> dict:
    n = len(results)
    if not n:
        return {"n": 0}
    wins = sum(1 for x in results if x["win"])
    total_r = sum(x["r"] for x in results)
    return {
        "n": n,
        "wr": 100.0 * wins / n,
        "exp_r": total_r / n,
        "total_r": total_r,
        "net_pips": sum(x["net_pips"] for x in results),
        "targets": sum(1 for x in results if x["reason"] == "TARGET"),
        "stops": sum(1 for x in results if x["reason"] == "STOP"),
        "timeouts": sum(1 for x in results if x["reason"] == "TIMEOUT"),
    }


def print_table(title: str, rows: List[tuple]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'RR':>5} {'n':>5} {'WR%':>6} {'exp R':>8} {'total R':>9} "
          f"{'net pips':>9} {'tgt':>5} {'stop':>5} {'t/o':>5}")
    for label, s in rows:
        if not s.get("n"):
            print(f"{label:>5} {'no data':>5}")
            continue
        print(f"{label:>5} {s['n']:>5} {s['wr']:>6.1f} {s['exp_r']:>8.3f} "
              f"{s['total_r']:>9.1f} {s['net_pips']:>9.0f} {s['targets']:>5} "
              f"{s['stops']:>5} {s['timeouts']:>5}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, nargs="+", default=DEFAULT_RRS)
    ap.add_argument("--refresh", action="store_true", help="bypass the bar cache")
    args = ap.parse_args()

    settings = get_settings()
    if not settings.oanda_api_key:
        print("OANDA_API_KEY is not set; cannot fetch bars.")
        return 1

    trades = load_trades(settings.db_path)
    print(f"Resolved trades with usable levels: {len(trades)}")
    if not trades:
        return 1

    stamps = [d for d in (parse_ts(t["entry_ts"]) for t in trades) if d]
    start, end = min(stamps), max(stamps) + timedelta(hours=MAX_HOLD_HOURS + 1)
    pairs = sorted({t["pair"] for t in trades})
    print(f"Window {start:%Y-%m-%d} to {end:%Y-%m-%d} across {len(pairs)} pairs")

    client = OandaClient(settings)
    bars_by_pair: Dict[str, List[dict]] = {}
    for pair in pairs:
        bars_by_pair[pair] = fetch_bars(client, pair, start, end, args.refresh)
        print(f"  {pair}: {len(bars_by_pair[pair])} M5 bars")

    # Trades resolvable at every RR — the common set keeps the comparison honest.
    by_rr: Dict[float, Dict[int, dict]] = {}
    for rr in args.rr:
        by_rr[rr] = {}
        for t in trades:
            res = replay(t, bars_by_pair.get(t["pair"], []), rr)
            if res is not None:
                by_rr[rr][t["id"]] = res
    common = set.intersection(*(set(v) for v in by_rr.values())) if by_rr else set()
    print(f"Replayed on {len(common)} trades covered at every RR "
          f"({len(trades) - len(common)} dropped for missing bars)")

    print_table("Overall, by reward:risk", [
        (f"{rr:g}", summarize([by_rr[rr][i] for i in common])) for rr in args.rr
    ])

    by_id = {t["id"]: t for t in trades}
    for signal in sorted({t["signal"] for t in trades if t["signal"]}):
        ids = [i for i in common if by_id[i]["signal"] == signal]
        if len(ids) < 20:
            continue
        print_table(f"{signal}  (n={len(ids)})", [
            (f"{rr:g}", summarize([by_rr[rr][i] for i in ids])) for rr in args.rr
        ])

    # How far price actually runs: the ceiling on any target choice.
    control = args.rr[len(args.rr) // 2]
    mfes = sorted(by_rr[control][i]["mfe_r"] for i in common)
    print("\n=== Max favorable excursion (R), 12h window ===")
    for q in (0.25, 0.5, 0.6, 0.7, 0.75, 0.9):
        print(f"  p{q * 100:>4.0f}: {mfes[int(q * (len(mfes) - 1))]:.2f}R")
    for level in (1.0, 1.5, 2.0, 2.5, 3.0):
        hit = 100.0 * sum(1 for m in mfes if m >= level) / len(mfes)
        print(f"  reached {level:.1f}R before the stop or timeout: {hit:.1f}%")

    # Control check: RR 1.5 replay should track the recorded outcomes.
    if 1.5 in by_rr:
        ids = [i for i in common if by_id[i]["recorded_r"] is not None]
        agree = sum(
            1 for i in ids
            if (by_rr[1.5][i]["r"] > 0) == (by_id[i]["recorded_outcome"] == "WIN")
        )
        print(f"\nControl: RR 1.5 replay agrees with recorded outcome on "
              f"{agree}/{len(ids)} ({100.0 * agree / max(len(ids), 1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
