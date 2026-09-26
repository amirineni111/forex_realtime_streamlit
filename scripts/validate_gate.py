"""
Re-measure the entry-quality gate against everything that has actually resolved.

The thresholds in ``forex.quality`` were fitted on trades resolved between
2026-06-10 and 2026-09-25. Thresholds chosen on a sample and then evaluated on that
same sample flatter themselves, so the only number worth trusting is what the gate
does on trades it has never seen. This script reports both, plus a per-month
breakdown with the rule held fixed, which is the closest thing to out-of-sample
evidence available without waiting.

Run it whenever a few hundred more trades have resolved. What to look for:

  * ``gate expectancy`` positive and meaningfully above ``all trades`` — the gate is
    still selecting.
  * the per-month column not trending toward zero — if it is, the edge is decaying
    and the thresholds need re-deriving rather than re-tightening.
  * ``kept %`` not collapsing — a gate that keeps 2% of setups is overfitted to
    whatever happened to work.

Usage:
    python scripts/validate_gate.py
    python scripts/validate_gate.py --since 2026-08-01
    python scripts/validate_gate.py --db data/forex_data.sqlite3
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forex import quality
from forex.signals import (
    _MAX_STOP_ATR_MULT, _MIN_STOP_PIPS, _RR, _STOP_ATR_MULT, breakeven_win_rate,
)
from forex.timeutil import parse_ts


def load_rows(db: Path, since: Optional[str]) -> List[dict]:
    """Resolved trades that carry the logged feature vector the gate needs."""
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT t.pair, t.entry_ts, t.features_json, t.cost_ratio, t.spread_pips,
               t.atr14, t.total_score, t.regime, t.session,
               o.outcome, o.r_multiple
        FROM forex_trade_outcomes o
        JOIN forex_signal_tracking t ON t.id = o.tracking_id
        WHERE o.outcome IN ('WIN', 'LOSS')
          AND t.features_json IS NOT NULL
    """
    params: list = []
    if since:
        sql += " AND t.entry_ts >= ?"
        params.append(since)
    sql += " ORDER BY t.entry_ts"
    rows = [dict(r) for r in conn.execute(sql, params)]
    conn.close()
    return rows


def current_cost_ratio(row: dict) -> Optional[float]:
    """
    Cost ratio this setup *would* carry under today's stop sizing.

    The stored ``cost_ratio`` cannot be used: it was computed when the stop was
    defined as ``max(..., 8 × spread)``, which pinned it at exactly 0.125 on 772 of
    846 trades. Replaying the gate against those stored values would reject almost
    everything for a cost the current sizing would never have charged. Recomputing
    from spread and ATR is what makes the comparison apples-to-apples.
    """
    spread, atr = row.get("spread_pips"), row.get("atr14")
    if not spread or not atr or atr <= 0:
        return None
    pip = 0.01 if "JPY" in (row.get("pair") or "") else 0.0001
    atr_pips = atr / pip
    vol_stop = max(_STOP_ATR_MULT * atr_pips, _MIN_STOP_PIPS)
    cost_stop = spread / quality.MAX_COST_RATIO
    ceiling = max(_MAX_STOP_ATR_MULT * atr_pips, _MIN_STOP_PIPS)
    stop_pips = min(max(vol_stop, cost_stop), ceiling)
    return spread / stop_pips if stop_pips > 0 else None


def block_axis(block: str) -> str:
    """Bucket a block message by which rule produced it, not by its numbers."""
    lowered = block.lower()
    for needle, label in (
        ("rsi", "RSI extended in trade direction"),
        ("session range", "price extended in session range"),
        ("macd", "MACD histogram extended"),
        ("ema gap", "EMA gap extended"),
        ("cost", "cost too high for the risk"),
        ("excluded hour", "excluded hour"),
        ("excluded (spread", "excluded pair"),
    ):
        if needle in lowered:
            return label
    return block


def stats(rows: List[dict]) -> dict:
    if not rows:
        return {"n": 0, "win_rate": None, "expectancy": None, "total_r": 0.0}
    wins = sum(1 for r in rows if r["outcome"] == "WIN")
    rs = [r["r_multiple"] for r in rows if r["r_multiple"] is not None]
    return {
        "n": len(rows),
        "win_rate": wins / len(rows),
        "expectancy": (sum(rs) / len(rs)) if rs else None,
        "total_r": sum(rs),
    }


def fmt(s: dict) -> str:
    if not s["n"]:
        return "     no trades"
    return (f"n={s['n']:5d}  WR={s['win_rate']:6.1%}  "
            f"E={s['expectancy']:+.4f}R  total={s['total_r']:+8.1f}R")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/forex_data.sqlite3", type=Path)
    ap.add_argument("--since", help="only trades entered on/after this ISO date")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}", file=sys.stderr)
        return 1

    rows = load_rows(args.db, args.since)
    if not rows:
        print("No resolved trades with logged features yet. "
              "The gate cannot be validated until the scanner has armed and "
              "resolved signals carrying features_json.")
        return 1

    kept: List[dict] = []
    rejected: List[dict] = []
    block_counts: dict = {}

    for row in rows:
        try:
            feats = json.loads(row["features_json"])
        except (TypeError, ValueError):
            continue
        dt = parse_ts(row["entry_ts"])
        row["_cost_ratio"] = current_cost_ratio(row)
        verdict = quality.evaluate(
            feats,
            pair=row["pair"],
            hour_utc=dt.hour if dt else None,
            cost_ratio=row["_cost_ratio"],
        )
        (kept if verdict.passed else rejected).append(row)
        for block in verdict.blocks:
            key = block_axis(block)
            block_counts[key] = block_counts.get(key, 0) + 1

    all_s, kept_s, rej_s = stats(rows), stats(kept), stats(rejected)

    print("=" * 78)
    print(f"ENTRY-QUALITY GATE VALIDATION   ({args.db})")
    if args.since:
        print(f"since {args.since}")
    print("=" * 78)
    print(f"  all trades   {fmt(all_s)}")
    print(f"  GATE PASSED  {fmt(kept_s)}")
    print(f"  gate blocked {fmt(rej_s)}")
    if all_s["n"]:
        print(f"\n  kept {kept_s['n'] / all_s['n']:.1%} of setups")

    # The bar the gate has to clear, given what the trades actually cost.
    costs = [r["_cost_ratio"] for r in kept if r.get("_cost_ratio") is not None]
    if costs:
        mean_cost = sum(costs) / len(costs)
        needed = breakeven_win_rate(_RR, mean_cost)
        print(f"  mean cost ratio on kept trades: {mean_cost:.1%}")
        print(f"  breakeven win rate at RR {_RR}: {needed:.1%}")
        if kept_s["win_rate"] is not None:
            margin = kept_s["win_rate"] - needed
            verdict = "ABOVE breakeven" if margin > 0 else "BELOW breakeven"
            print(f"  margin: {margin:+.1%} -> {verdict}")

    print("\n  why setups were blocked (a setup can trip several):")
    for key, count in sorted(block_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {count:5d}  {key}")

    # Per-month, rule fixed. This is the decay check.
    print("\n  per month (gate held fixed — watch for decay):")
    print(f"    {'month':<9}{'all':>28}   {'gated':>28}")
    months = sorted({(parse_ts(r['entry_ts']) or None) and
                     parse_ts(r["entry_ts"]).strftime("%Y-%m") for r in rows if r["entry_ts"]})
    for month in [m for m in months if m]:
        m_all = [r for r in rows if (parse_ts(r["entry_ts"]) or None)
                 and parse_ts(r["entry_ts"]).strftime("%Y-%m") == month]
        m_kept = [r for r in kept if (parse_ts(r["entry_ts"]) or None)
                  and parse_ts(r["entry_ts"]).strftime("%Y-%m") == month]
        a, k = stats(m_all), stats(m_kept)
        a_txt = f"n={a['n']:4d} WR={a['win_rate']:5.1%} E={a['expectancy']:+.3f}R" if a["n"] else "-"
        k_txt = f"n={k['n']:4d} WR={k['win_rate']:5.1%} E={k['expectancy']:+.3f}R" if k["n"] else "-"
        print(f"    {month:<9}{a_txt:>28}   {k_txt:>28}")

    print("\n  current thresholds:")
    print(f"    MAX_RSI_DIR        {quality.MAX_RSI_DIR}")
    print(f"    MAX_RANGE_POS_DIR  {quality.MAX_RANGE_POS_DIR}")
    print(f"    MAX_MACD_HIST_ATR  {quality.MAX_MACD_HIST_ATR}")
    print(f"    MAX_EMA_GAP_ATR    {quality.MAX_EMA_GAP_ATR}")
    print(f"    MAX_COST_RATIO     {quality.MAX_COST_RATIO}")
    print(f"    EXCLUDED_PAIRS     {sorted(quality.EXCLUDED_PAIRS)}")
    print(f"    EXCLUDED_HOURS     {sorted(quality.EXCLUDED_HOURS)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
