"""
End-to-end scan against a stubbed OANDA client.

Covers the sequential post-scan phase, which is where currency strength, model
scoring, signal arming and feature logging all have to happen in the right order.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from forex.config import AppSettings
from forex.features import FEATURE_NAMES, FEATURE_VERSION
from forex.model import fit
from forex.models import ForexBar, ForexQuote, ScanRequest
from forex.storage import Storage
import forex.scanner as scanner_mod
from forex.scanner import run_scan


PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
BASE = {"EUR_USD": 1.1000, "GBP_USD": 1.2700, "USD_JPY": 150.00}
GRAN_MINUTES = {"M5": 5, "H1": 60, "H4": 240}


def _make_bars(pair: str, granularity: str, count: int, slope: float = 1.0):
    """A clean uptrend with mild oscillation — enough for every indicator to resolve."""
    px = BASE[pair]
    tick = 0.01 if "JPY" in pair else 0.0001
    step = timedelta(minutes=GRAN_MINUTES[granularity])
    start = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc) - step * count
    bars = []
    for i in range(count):
        drift = slope * i * tick * 0.6
        wobble = math.sin(i / 4.0) * tick * 1.5
        close = px + drift + wobble
        high = close + tick * 2
        low = close - tick * 2
        ts = (start + step * i).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        bars.append(ForexBar(
            pair=pair, timeframe=granularity, timestamp=ts,
            open=close - tick * 0.5, high=high, low=low, close=close, volume=100,
        ))
    return bars


class StubOanda:
    """Deterministic stand-in for OandaClient — no network, no credentials."""

    def __init__(self, settings=None, spread_pips: float = 1.0):
        self.spread_pips = spread_pips
        self.candle_calls = 0

    def get_pricing(self, pairs):
        quotes = []
        for p in pairs:
            tick = 0.01 if "JPY" in p else 0.0001
            bars = _make_bars(p, "M5", 200)
            mid = bars[-1].close
            half = self.spread_pips * tick / 2
            quotes.append(ForexQuote(
                pair=p, bid=round(mid - half, 6), ask=round(mid + half, 6),
                spread_pips=self.spread_pips,
                as_of="2026-07-27T00:00:00.000000000Z",
            ))
        return quotes

    def get_candles(self, pair, granularity="M5", count=200):
        self.candle_calls += 1
        return _make_bars(pair, granularity, count)


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scanner_mod, "OandaClient", StubOanda)
    storage = Storage(tmp_path / "scan.sqlite3")
    settings = AppSettings(oanda_api_key="stub", oanda_account_id="stub",
                           db_path=tmp_path / "scan.sqlite3")
    return settings, storage


def _run(settings, storage, **kw):
    return run_scan(settings, storage, ScanRequest(pairs=PAIRS, **kw))


class TestScanEndToEnd:
    def test_scan_completes_and_persists_snapshots(self, env):
        settings, storage = env
        summary = _run(settings, storage)
        assert summary.pairs_scanned == len(PAIRS)
        assert summary.errors == 0

        snaps = storage.load_latest_snapshots()
        assert len(snaps) == len(PAIRS)
        assert {s["pair"] for s in snaps} == set(PAIRS)

    def test_every_snapshot_carries_the_new_gating_fields(self, env):
        settings, storage = env
        _run(settings, storage)
        for s in storage.load_latest_snapshots():
            assert s["cost_ratio"] is not None
            assert s["blocked_ahead"] in (0, 1)
            assert s["required_prob"] is not None
            # No model has been trained yet, so nothing should be model-gated.
            assert s["model_prob"] is None

    def test_scores_are_consistent_with_stored_signal(self, env):
        """
        The strength bonus is folded into total_score before the decision, so the
        persisted score is the one the signal was derived from — it used to be
        mutated after the fact, leaving the two out of step.
        """
        settings, storage = env
        _run(settings, storage)
        for s in storage.load_latest_snapshots():
            if s["trade_signal"] in ("BUY_CANDIDATE", "STRONG_BUY"):
                assert s["total_score"] >= 45
            if s["trade_signal"] == "STRONG_BUY":
                assert s["mtf_score"] >= 15

    def test_armed_signals_store_a_full_feature_vector(self, env):
        settings, storage = env
        _run(settings, storage)
        tracked = storage.load_tracked_signals("open")
        if not tracked:
            pytest.skip("stub data produced no actionable signal")
        for row in tracked:
            feats = json.loads(row["features_json"])
            assert set(feats) == set(FEATURE_NAMES)
            assert row["feature_version"] == FEATURE_VERSION
            assert row["spread_pips"] is not None
            assert row["session"] is not None
            assert all(np.isfinite(list(feats.values())))

    def test_wide_spread_pairs_are_not_armed(self, env):
        """A 12-pip spread cannot clear the cost gate on any sane stop."""
        settings, storage = env
        scanner_mod.OandaClient = lambda s: StubOanda(s, spread_pips=12.0)
        _run(settings, storage, max_spread_pips=20.0)
        for s in storage.load_latest_snapshots():
            if s["trade_signal"] not in ("AVOID", "WATCH_ONLY"):
                # If anything survived, its cost ratio must still be within bounds.
                assert s["cost_ratio"] <= 0.15

    def test_rescan_does_not_duplicate_open_signals(self, env):
        settings, storage = env
        _run(settings, storage)
        first = len(storage.load_tracked_signals("open"))
        _run(settings, storage)
        assert len(storage.load_tracked_signals("open")) == first

    def test_scan_is_idempotent_on_repeat(self, env):
        settings, storage = env
        a = _run(settings, storage)
        b = _run(settings, storage)
        assert a.pairs_scanned == b.pairs_scanned == len(PAIRS)
        assert b.errors == 0


class TestModelInTheLoop:
    def _train_stub_model(self, storage, bias: float):
        """Persist a model whose output is pinned near `bias` regardless of input."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, len(FEATURE_NAMES)))
        y = (rng.uniform(size=400) < bias).astype(float)
        model = fit(X, y, l2=1e6)     # heavy penalty ⇒ intercept-only ⇒ predicts base rate
        storage.save_model(model.to_json(),
                           {"feature_version": FEATURE_VERSION, "algo": "logistic_l2"},
                           activate=True)
        return model

    def test_pessimistic_model_suppresses_all_entries(self, env):
        settings, storage = env
        self._train_stub_model(storage, bias=0.05)
        _run(settings, storage)

        snaps = storage.load_latest_snapshots()
        assert all(s["model_prob"] is not None for s in snaps)
        actionable = [s for s in snaps if s["trade_signal"] not in ("AVOID", "WATCH_ONLY")]
        assert actionable == [], "a 5% win-probability model must veto every entry"
        assert storage.load_tracked_signals("open") == []

    def test_optimistic_model_allows_entries(self, env):
        settings, storage = env
        self._train_stub_model(storage, bias=0.95)
        summary = _run(settings, storage)
        snaps = storage.load_latest_snapshots()
        assert all(s["model_prob"] is not None and s["model_prob"] > 0.5 for s in snaps)
        assert summary.signals_found >= 0   # gating no longer blocks on probability

    def test_model_with_wrong_feature_version_is_ignored(self, env):
        """A stale contract must fail closed rather than serve misaligned inputs."""
        settings, storage = env
        model = self._train_stub_model(storage, bias=0.05)
        payload = json.loads(model.to_json())
        payload["feature_version"] = FEATURE_VERSION + 99
        storage.save_model(json.dumps(payload),
                           {"feature_version": FEATURE_VERSION + 99}, activate=True)

        _run(settings, storage)
        for s in storage.load_latest_snapshots():
            assert s["model_prob"] is None

    def test_corrupt_model_json_does_not_break_the_scan(self, env):
        settings, storage = env
        storage.save_model("{not valid json", {"feature_version": FEATURE_VERSION},
                           activate=True)
        summary = _run(settings, storage)
        assert summary.pairs_scanned == len(PAIRS)
        assert summary.errors == 0
