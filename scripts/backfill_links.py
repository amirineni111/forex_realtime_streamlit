"""
One-off repair of the historical outcome records.

What this CAN do
----------------
Populate ``forex_trade_outcomes.tracking_id`` by matching each legacy outcome back
to its ``forex_signal_tracking`` row on (pair, signal, entry_price), and copy the
legacy ``exit_pips`` into the new ``gross_pips`` column. Only unambiguous matches
(exactly one candidate) are linked; anything ambiguous is left alone and reported.

What this deliberately does NOT do
----------------------------------
Reconstruct model features for the pre-existing trades. Those rows were written
before feature logging existed, and the ``forex_signal_tracking`` rows of that era
hold only prices and ATR — no RSI, ADX, MACD, MTF state, structure or session. The
``forex_snapshots`` table that did hold them is pruned to the last 20 scans, so the
inputs are genuinely gone.

They are therefore left untrainable rather than imputed. Filling ~25 features from
pair-level averages would produce a model fitted to invented inputs and an
out-of-sample AUC that means nothing. The learning loop starts from the trades
recorded after this change; the legacy rows stay useful only as the descriptive
baseline (39.56% win rate, -0.0134R gross).

    python scripts/backfill_links.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forex.config import get_settings
from forex.features import FEATURE_VERSION
from forex.storage import Storage


LINK_SQL = """
UPDATE forex_trade_outcomes AS o
SET tracking_id = (
    SELECT t.id FROM forex_signal_tracking t
    WHERE t.pair = o.pair AND t.signal = o.signal AND t.entry_price = o.entry_price
)
WHERE o.tracking_id IS NULL
  AND (
    SELECT COUNT(*) FROM forex_signal_tracking t
    WHERE t.pair = o.pair AND t.signal = o.signal AND t.entry_price = o.entry_price
  ) = 1
"""

GROSS_SQL = """
UPDATE forex_trade_outcomes
SET gross_pips = exit_pips
WHERE gross_pips IS NULL AND exit_pips IS NOT NULL
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Link legacy outcomes to tracking rows")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    settings = get_settings()
    storage = Storage(settings.db_path)          # ensures the schema is migrated
    print(f"DB: {settings.db_path.resolve()}")

    with storage._connect() as conn:
        unlinked = conn.execute(
            "SELECT COUNT(*) FROM forex_trade_outcomes WHERE tracking_id IS NULL"
        ).fetchone()[0]
        linkable = conn.execute(
            "SELECT COUNT(*) FROM forex_trade_outcomes o WHERE o.tracking_id IS NULL AND ("
            "  SELECT COUNT(*) FROM forex_signal_tracking t"
            "  WHERE t.pair=o.pair AND t.signal=o.signal AND t.entry_price=o.entry_price"
            ") = 1"
        ).fetchone()[0]

    print(f"Outcomes missing tracking_id : {unlinked}")
    print(f"Unambiguously linkable       : {linkable}")
    print(f"Ambiguous / unmatched        : {unlinked - linkable} (left untouched)")

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply.")
        return 0

    with storage._connect() as conn:
        linked = conn.execute(LINK_SQL).rowcount
        gross = conn.execute(GROSS_SQL).rowcount
    print(f"\nLinked {linked} outcomes to their tracking rows.")
    print(f"Copied exit_pips -> gross_pips on {gross} rows.")

    trainable = len(storage.load_training_rows(feature_version=FEATURE_VERSION))
    print(f"\nTrainable rows (linked AND carrying a stored feature vector): {trainable}")
    if trainable == 0:
        print(
            "Expected: the legacy trades predate feature logging and their inputs are\n"
            "not recoverable, so none of them are trainable. New signals armed from now\n"
            "on will store features and become training data as they resolve."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
