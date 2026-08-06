"""
Tests for the shared retrain pipeline and the promote/rollback controls that the
Model tab drives. The dashboard buttons are thin wrappers over these functions, so
covering them here covers the UI behaviour that matters.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from forex.features import FEATURE_VERSION, build_features
from forex.model import MIN_TRAIN_SAMPLES, ForexModel
from forex.storage import Storage
from forex.training import evaluate_and_fit, gate_summary, save_candidate


T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _snapshot(rng, i, h1, h4, runway):
    return {
        "close": 1.10, "atr14": 0.0010, "adx14": float(rng.uniform(10, 40)),
        "rsi14": float(rng.uniform(30, 70)), "ema9": 1.1005, "ema20": 1.0996,
        "macd_histogram": 0.0002, "bb_width_pct": 0.5, "spread_pips": 1.0,
        "stop_pips": 12.0, "total_score": float(rng.uniform(45, 90)),
        "momentum_score": 30.0, "reversion_score": 0.0, "session_score": 10.0,
        "mtf_score": 30.0, "sr_score": 15.0,
        "nearest_support": 1.10 - runway * 0.001,
        "nearest_resistance": 1.10 + runway * 0.001,
        "h1_direction": {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}[h1],
        "h4_direction": {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}[h4],
        "base_strength": 60.0, "quote_strength": 40.0,
        "session_high": 1.102, "session_low": 1.098, "day_change_pct": 0.2,
        "current_session": "London_NY_Overlap",
        "as_of": (T0 + timedelta(minutes=30 * i)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
    }


def _seed(store: Storage, n: int, with_signal: bool = True, seed: int = 42):
    """
    Insert n resolved trades carrying features; optionally with a learnable edge.

    Rows are written directly in one transaction rather than through
    ``record_tracked_signal``: that path opens a connection per call and enforces the
    re-arm cooldown, which makes seeding hundreds of rows both slow and lossy. The
    arming path has its own coverage in test_forex_pipeline.py.
    """
    rng = np.random.default_rng(seed)
    tracking_rows, outcome_rows = [], []
    for i in range(n):
        h1, h4 = int(rng.choice([1, -1, 0])), int(rng.choice([1, -1, 0]))
        runway = float(rng.uniform(0, 5))
        direction = 1 if rng.uniform() < 0.5 else -1
        snap = _snapshot(rng, i, h1, h4, runway)
        f = build_features(snap, direction)
        if with_signal:
            logit = (0.9 * f["h1_agrees"] + 0.7 * f["h4_agrees"]
                     + 0.35 * f["dist_to_target_level_atr"] - 1.2)
            p = 1 / (1 + np.exp(-logit))
        else:
            p = 0.4
        win = bool(rng.uniform() < p)
        created = (T0 + timedelta(minutes=30 * i)).isoformat()
        tracking_rows.append((
            i + 1, f"SYN{i:04d}_USD",
            "BUY_CANDIDATE" if direction > 0 else "SHORT_CANDIDATE", direction,
            1.10, 1.0988, 1.1018, 12.0, 18.0, 0.001, snap["as_of"], "closed", created,
            json.dumps(f), FEATURE_VERSION, 0.45, 0.083, 1.0,
            snap["total_score"], snap["adx14"], "TREND", "London_NY_Overlap",
        ))
        outcome_rows.append((
            i + 1, f"SYN{i:04d}_USD", "BUY_CANDIDATE", 1.10, 1.1018, 17.0, 18.0, 1.0,
            17.0, 1.417 if win else -1.083, "WIN" if win else "LOSS", 45,
            "TARGET" if win else "STOP", created,
        ))

    with store._connect() as c:
        c.executemany(
            "INSERT INTO forex_signal_tracking (id,pair,signal,direction,entry_price,"
            "stop_price,target_price,stop_pips,target_pips,atr14,entry_ts,status,created_at,"
            "features_json,feature_version,required_prob,cost_ratio,spread_pips,"
            "total_score,adx14,regime,session) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tracking_rows,
        )
        c.executemany(
            "INSERT INTO forex_trade_outcomes (watchlist_id,tracking_id,pair,signal,"
            "entry_price,exit_price,exit_pips,gross_pips,cost_pips,net_pips,r_multiple,"
            "outcome,hold_minutes,exit_reason,created_at) VALUES (0,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            outcome_rows,
        )


@pytest.fixture
def store(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "train.sqlite3")


class TestEvaluateAndFit:
    def test_refuses_below_the_minimum_sample_size(self, store):
        _seed(store, 30)
        report = evaluate_and_fit(store)
        assert report["error"] is not None
        assert report["model"] is None
        assert str(MIN_TRAIN_SAMPLES) in report["error"]

    def test_learns_and_passes_the_gate_on_a_real_edge(self, store):
        _seed(store, 500, with_signal=True)
        report = evaluate_and_fit(store)
        assert report["error"] is None
        assert report["metrics"]["auc"] > 0.60
        assert report["passes"] is True
        assert gate_summary(report).startswith("PASS")

    def test_fails_the_gate_on_pure_noise(self, store):
        """The gate exists to stop a no-skill model from vetoing live setups."""
        _seed(store, 500, with_signal=False, seed=11)
        report = evaluate_and_fit(store)
        assert report["error"] is None
        assert report["passes"] is False
        assert gate_summary(report).startswith("FAIL")

    def test_report_carries_everything_the_ui_renders(self, store):
        _seed(store, 400)
        report = evaluate_and_fit(store)
        for key in ("metrics", "gated", "calibration", "folds", "coefficients",
                    "baseline_expectancy_r", "fold_auc_std", "passes"):
            assert key in report, f"missing {key}"
        assert len(report["gated"]) >= 3
        assert all("threshold" in g for g in report["gated"])
        assert report["metrics"]["feature_version"] == FEATURE_VERSION

    def test_min_auc_threshold_is_respected(self, store):
        _seed(store, 500, with_signal=True)
        assert evaluate_and_fit(store, min_auc=0.53)["passes"] is True
        # An unreachable bar must fail even a genuinely skilful model.
        assert evaluate_and_fit(store, min_auc=0.99)["passes"] is False


class TestCandidateAndPromotion:
    def test_candidate_is_saved_inactive(self, store):
        """Retrain must never silently change what the scanner serves."""
        _seed(store, 400)
        report = evaluate_and_fit(store)
        mid = save_candidate(store, report, notes="unit test")
        assert mid is not None
        assert store.load_active_model_json() is None
        saved = [m for m in store.load_models() if m["id"] == mid][0]
        assert saved["is_active"] == 0
        assert saved["notes"] == "unit test"

    def test_promote_activates_exactly_one(self, store):
        _seed(store, 400)
        first = save_candidate(store, evaluate_and_fit(store))
        second = save_candidate(store, evaluate_and_fit(store, l2=5.0))
        assert store.activate_model(second) is True
        active = [m for m in store.load_models() if m["is_active"]]
        assert [m["id"] for m in active] == [second]

    def test_rollback_to_an_earlier_model(self, store):
        _seed(store, 400)
        first = save_candidate(store, evaluate_and_fit(store))
        second = save_candidate(store, evaluate_and_fit(store, l2=5.0))
        store.activate_model(second)
        assert store.activate_model(first) is True
        active = [m for m in store.load_models() if m["is_active"]]
        assert [m["id"] for m in active] == [first]

    def test_activating_a_missing_id_is_rejected(self, store):
        assert store.activate_model(99999) is False

    def test_deactivate_all_reverts_to_rules_only(self, store):
        _seed(store, 400)
        mid = save_candidate(store, evaluate_and_fit(store))
        store.activate_model(mid)
        assert store.load_active_model_json() is not None
        store.deactivate_all_models()
        assert store.load_active_model_json() is None
        # The model itself survives, so it can be rolled forward again.
        assert any(m["id"] == mid for m in store.load_models())

    def test_shadow_model_is_served_but_not_active(self, store):
        _seed(store, 400)
        mid = save_candidate(store, evaluate_and_fit(store))
        assert store.shadow_model(mid) is True
        # Shadow is the offline lane: the scanner can score with it, but nothing
        # gates on it, so load_active_model_json must stay empty.
        assert store.load_active_model_json() is None
        assert store.load_shadow_model_json() is not None

    def test_a_model_cannot_gate_and_shadow_at_once(self, store):
        _seed(store, 400)
        mid = save_candidate(store, evaluate_and_fit(store))
        store.shadow_model(mid)
        store.activate_model(mid)
        row = [m for m in store.load_models() if m["id"] == mid][0]
        assert (row["is_active"], row["is_shadow"]) == (1, 0)
        # And back the other way, so promotion is reversible into shadow.
        store.shadow_model(mid)
        row = [m for m in store.load_models() if m["id"] == mid][0]
        assert (row["is_active"], row["is_shadow"]) == (0, 1)

    def test_only_one_model_shadows_at_a_time(self, store):
        _seed(store, 400)
        first = save_candidate(store, evaluate_and_fit(store))
        second = save_candidate(store, evaluate_and_fit(store, l2=5.0))
        store.shadow_model(first)
        store.shadow_model(second)
        assert [m["id"] for m in store.load_models() if m["is_shadow"]] == [second]

    def test_shadowing_a_missing_id_is_rejected(self, store):
        assert store.shadow_model(99999) is False

    def test_clear_shadow_leaves_the_model_stored(self, store):
        _seed(store, 400)
        mid = save_candidate(store, evaluate_and_fit(store))
        store.shadow_model(mid)
        store.clear_shadow_model()
        assert store.load_shadow_model_json() is None
        assert any(m["id"] == mid for m in store.load_models())

    def test_promoted_model_is_servable(self, store):
        _seed(store, 400)
        report = evaluate_and_fit(store)
        mid = save_candidate(store, report)
        store.activate_model(mid)
        served = ForexModel.from_json(store.load_active_model_json())
        feats = build_features(_snapshot(np.random.default_rng(0), 0, 1, 1, 3.0), 1)
        assert 0.0 <= served.predict_proba(feats) <= 1.0
        assert served.feature_version == FEATURE_VERSION

    def test_save_candidate_returns_none_on_a_failed_report(self, store):
        _seed(store, 20)
        report = evaluate_and_fit(store)
        assert save_candidate(store, report) is None
        assert store.load_models() == []
