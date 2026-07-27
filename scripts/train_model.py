"""
Retrain the forex direction model and report honest out-of-sample metrics.

Always run from the project root:

    python scripts/train_model.py                 # evaluate + train, do not activate
    python scripts/train_model.py --activate      # promote the model the scanner serves
    python scripts/train_model.py --min-auc 0.55  # refuse to activate a weak model

The gate is deliberate: a model is only worth serving if it beats chance out of
sample AND turns a profit at the decision threshold. Both are printed, and
``--activate`` refuses when either fails unless ``--force`` is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train_model.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forex.config import get_settings
from forex.features import FEATURE_VERSION
from forex.signals import breakeven_win_rate, _RR
from forex.storage import Storage
from forex.training import evaluate_and_fit, gate_summary, save_candidate


def _fmt(value, spec=".4f", dash="n/a"):
    return dash if value is None else format(value, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the forex direction model")
    ap.add_argument("--activate", action="store_true",
                    help="promote this model to the one the scanner serves")
    ap.add_argument("--force", action="store_true",
                    help="activate even if the quality gate fails")
    ap.add_argument("--l2", type=float, default=1.0, help="L2 penalty strength")
    ap.add_argument("--folds", type=int, default=5, help="walk-forward folds")
    ap.add_argument("--min-auc", type=float, default=0.53,
                    help="minimum out-of-sample AUC required to activate")
    ap.add_argument("--notes", default="", help="free-text note stored with the model")
    args = ap.parse_args()

    settings = get_settings()
    storage = Storage(settings.db_path)
    print(f"DB: {settings.db_path.resolve()}")

    report = evaluate_and_fit(
        storage, l2=args.l2, folds=args.folds, min_auc=args.min_auc,
    )
    print(f"Labelled rows with stored features (v{FEATURE_VERSION}): {report['n_rows']}")
    if report["error"]:
        print(f"\n{report['error']}\n"
              "Run the scanner and let signals resolve before retraining. Nothing was written.")
        return 1

    m = report["metrics"]
    print(f"Feature matrix: {m['n_train']} rows")
    print(f"Base win rate:  {report['base_rate']:.4f}   (breakeven at RR={_RR} is "
          f"{breakeven_win_rate(_RR, 0.0):.4f} before cost)")

    print("\n-- Walk-forward out-of-sample evaluation --")
    for f in report["folds"]:
        print(f"  fold {f['fold']}: train={f['train_n']:>5} test={f['test_n']:>5} "
              f"auc={_fmt(f['auc'])} actual_wr={f['actual_win_rate']:.4f}")

    print(f"\n  pooled OOS AUC        : {_fmt(m['auc'])}   (0.50 = no skill)")
    print(f"  fold AUC std dev      : {_fmt(report['fold_auc_std'])}   (high = regime-fitted)")
    print(f"  Brier score           : {_fmt(m['brier'], '.5f')}")
    print(f"  top-decile precision  : {_fmt(m['top_decile_prec'])}")
    print(f"  base rate             : {m['base_rate']:.4f}")

    print("\n  calibration (predicted vs actual):")
    for b in report["calibration"]:
        print(f"    bin {b['bin']}  n={b['n']:>4}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    print("\n-- Performance at the decision threshold --")
    for g in report["gated"]:
        print(f"  p>={g['threshold']:.3f}: took {g['trades_taken']:>4}/{g['trades_available']} "
              f"({g['selectivity']:.0%})  expectancy={_fmt(g.get('expectancy_r'))}R  "
              f"total={_fmt(g.get('total_r'), '.1f')}R")
    print(f"  ungated baseline expectancy: {report['baseline_expectancy_r']:.4f}R")

    print("\n-- Largest standardised coefficients (what carries the edge) --")
    for name, coef in report["coefficients"][:12]:
        print(f"  {name:<28} {coef:+.4f}")

    print(f"\nQuality gate: {gate_summary(report)}")

    model_id = save_candidate(storage, report, notes=args.notes)
    activate = args.activate and (report["passes"] or args.force)
    if args.activate and not report["passes"] and not args.force:
        print("Refusing to activate a model that fails the gate. Re-run with --force "
              "to override, or keep collecting data.")
    if activate and model_id is not None:
        storage.activate_model(model_id)

    print(f"\nSaved model id={model_id}  active={activate}")
    if not activate:
        print("The scanner will keep running rules-only until a model is activated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
