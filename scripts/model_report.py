"""
Judge a model on live, resolved trades rather than on its own training split.

A model's training metrics are a promise; this is the receipt. It reads the
probabilities the scanner actually logged at arm time, joins them to how those
trades really finished, and asks the only question that matters: would gating on
this probability have made more money than not gating at all?

Read the ``model_mode`` column before trusting any of it:

  shadow  the model scored every directional setup the rules proposed, so winners
          and losers are both represented and the numbers mean what they say.
  active  the model was gating, so only trades it *allowed* have outcomes. Its
          win rate here is measured on the sample it selected and says nothing
          about the trades it blocked. Never promote on the strength of these.

Usage:
    python scripts/model_report.py
    python scripts/model_report.py --mode active     # inspect the censored rows
    python scripts/model_report.py --min-n 50
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forex.config import get_settings
from forex.model import calibration_bins, gated_performance, roc_auc
from forex.signals import _PROB_MARGIN, _RR, breakeven_win_rate

# Enough resolved trades for a threshold comparison to mean anything. Below this the
# expectancy difference is swamped by the spread of individual R outcomes.
MIN_SAMPLE = 60


def load_scored(db: Path, mode: str) -> List[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    where = "t.model_mode = ?" if mode != "any" else "1=1"
    rows = [dict(r) for r in conn.execute(
        f"""
        SELECT t.id, t.pair, t.signal, t.model_prob, t.required_prob, t.model_mode,
               t.cost_ratio, o.outcome, o.r_multiple
        FROM forex_trade_outcomes o
        JOIN forex_signal_tracking t ON t.id = o.tracking_id
        WHERE t.model_prob IS NOT NULL AND o.r_multiple IS NOT NULL AND {where}
        ORDER BY t.id
        """,
        () if mode == "any" else (mode,),
    )]
    conn.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="shadow", choices=["shadow", "active", "any"])
    ap.add_argument("--min-n", type=int, default=MIN_SAMPLE)
    args = ap.parse_args()

    settings = get_settings()
    rows = load_scored(settings.db_path, args.mode)
    print(f"Resolved trades scored in '{args.mode}' mode: {len(rows)}")
    if not rows:
        print(
            "\nNothing to report yet. Put a model in shadow with:\n"
            "    python -c \"from pathlib import Path; from forex.storage import Storage; "
            "Storage(Path('data/forex_data.sqlite3')).shadow_model(5)\"\n"
            "then let the scanner run — rows appear as tracked trades resolve."
        )
        return 0

    if args.mode == "active":
        print("WARNING: these rows were gated by the model that scored them. Only "
              "trades it allowed appear, so the sample is censored.")

    p = [r["model_prob"] for r in rows]
    y = [1.0 if r["outcome"] == "WIN" else 0.0 for r in rows]
    r_mult = [r["r_multiple"] for r in rows]

    auc = roc_auc(y, p)
    base_wr = 100.0 * sum(y) / len(y)
    base_exp = sum(r_mult) / len(r_mult)
    print(f"\nBaseline (take everything): n={len(rows)} WR={base_wr:.1f}% "
          f"expectancy={base_exp:+.4f}R")
    print(f"Discrimination: AUC={auc:.4f}" if auc is not None else "AUC: n/a")
    print("  AUC 0.50 = the probability is noise; below 0.50 it is backwards.")

    print("\n=== Calibration (does p mean what it says?) ===")
    print(f"{'bin':>4} {'n':>5} {'predicted':>10} {'actual':>8}")
    for b in calibration_bins(y, p, bins=5):
        print(f"{b['bin']:>4} {b['n']:>5} {b['pred']:>10.3f} {b['actual']:>8.3f}")

    # The decision test: gating only pays if it beats taking everything.
    live_req = round(breakeven_win_rate(_RR, 0.0) + _PROB_MARGIN, 4)
    thresholds = sorted({0.35, 0.40, 0.45, live_req, 0.50, 0.55, 0.60})
    print("\n=== Would gating have helped? ===")
    print(f"{'thresh':>7} {'taken':>6} {'kept%':>6} {'WR%':>6} {'exp R':>8} "
          f"{'vs base':>9}")
    best: Optional[dict] = None
    for t in thresholds:
        g = gated_performance(p, r_mult, t)
        if g["expectancy_r"] is None:
            print(f"{t:>7.3f} {0:>6} {'0.0':>6} {'-':>6} {'no trades':>8}")
            continue
        taken = [i for i, pv in enumerate(p) if pv >= t]
        wr = 100.0 * sum(y[i] for i in taken) / len(taken)
        lift = g["expectancy_r"] - base_exp
        flag = "  <- live bar" if abs(t - live_req) < 1e-9 else ""
        print(f"{t:>7.3f} {g['trades_taken']:>6} {100 * g['selectivity']:>6.1f} "
              f"{wr:>6.1f} {g['expectancy_r']:>+8.4f} {lift:>+9.4f}{flag}")
        if best is None or g["expectancy_r"] > best["expectancy_r"]:
            best = {**g, "lift": lift}

    print("\n=== Verdict ===")
    if len(rows) < args.min_n:
        print(f"NOT ENOUGH DATA — {len(rows)} resolved trades, {args.min_n} wanted. "
              f"Keep the model in shadow and re-run.")
    elif best is None or best["lift"] <= 0:
        print("DO NOT PROMOTE — no threshold beat taking every rules signal. "
              "Gating on this model would cost money and block trades.")
    elif best["expectancy_r"] <= 0:
        print(f"DO NOT PROMOTE — best threshold {best['threshold']:.3f} improves on "
              f"the baseline by {best['lift']:+.4f}R but is still {best['expectancy_r']:+.4f}R. "
              f"Filtering a losing strategy more finely does not make it a winning one.")
    else:
        print(f"CANDIDATE — threshold {best['threshold']:.3f} gives "
              f"{best['expectancy_r']:+.4f}R vs {base_exp:+.4f}R baseline "
              f"({best['lift']:+.4f}R lift) on {best['trades_taken']} trades.\n"
              f"Confirm it holds on a further ~{args.min_n} trades before promoting — "
              f"one threshold picked from this table is fitted to this table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
