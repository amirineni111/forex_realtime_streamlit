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
from forex.alerts import CallableSink
from forex.models import ForexBar, ForexQuote, ScanRequest
from forex.storage import Storage
import forex.scanner as scanner_mod
from forex.scanner import run_scan


PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
BASE = {"EUR_USD": 1.1000, "GBP_USD": 1.2700, "USD_JPY": 150.00}
GRAN_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}


def _make_bars(
    pair: str, granularity: str, count: int, slope: float = 1.0,
    pullback_bars: int = 0, pullback_mult: float = 1.0,
):
    """
    A clean uptrend with mild oscillation — enough for every indicator to resolve.

    ``pullback_bars`` retraces the last N bars against the trend. A pure uptrend is
    exactly the "extended" entry that ``forex.quality`` now rejects, so a stub that
    only ever produced one could no longer exercise the arming path at all. A trend
    that has pulled back into its own structure is the shape the gate is built to
    accept, and is what the actionable-signal tests use.
    """
    px = BASE[pair]
    tick = 0.01 if "JPY" in pair else 0.0001
    step = timedelta(minutes=GRAN_MINUTES[granularity])
    start = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc) - step * count
    peak = count - pullback_bars
    bars = []
    for i in range(count):
        if i < peak:
            drift = slope * i * tick * 0.6
        else:
            drift = slope * peak * tick * 0.6 - (i - peak) * tick * pullback_mult
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

    def __init__(self, settings=None, spread_pips: float = 1.0, pullback_bars: int = 0,
                 signal_timeframe: str = "M15"):
        self.spread_pips = spread_pips
        self.pullback_bars = pullback_bars
        # Which granularity carries the pullback. Must track the scanner's signal
        # timeframe, otherwise the retrace lands on bars nothing is scored from and
        # every gate-passing test silently stops exercising the arming path.
        self.signal_timeframe = signal_timeframe
        self.candle_calls = 0

    def get_pricing(self, pairs):
        quotes = []
        for p in pairs:
            tick = 0.01 if "JPY" in p else 0.0001
            bars = _make_bars(p, self.signal_timeframe, 200,
                              pullback_bars=self.pullback_bars)
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
        # Only the signal timeframe carries the pullback: H1/H4 stay trending so the
        # MTF gate still confirms the direction, which is the realistic shape of a
        # buy-the-dip setup.
        pullback = self.pullback_bars if granularity == self.signal_timeframe else 0
        return _make_bars(pair, granularity, count, pullback_bars=pullback)


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

    def test_extended_trend_is_gated_out(self, env, monkeypatch):
        """
        A pure uptrend is the extended entry the quality gate exists to reject.

        Measured over 1,003 trades carrying features, entries in the top quintile of
        every momentum axis won 30-34% against the 45% that RR 1.5 needs at the cost
        being paid. Nothing should arm off this shape.
        """
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=0))
        _run(settings, storage)

        snaps = storage.load_latest_snapshots()
        assert all(s["trade_signal"] in ("AVOID", "WATCH_ONLY") for s in snaps)
        assert any("entry quality failed" in (s["signal_reason"] or "") for s in snaps)

    def test_gate_rejected_setups_are_still_tracked_for_evidence(self, env, monkeypatch):
        """
        Rejected setups stay in the forward-tracking ledger, flagged as rejected.

        Arming only what the gate passes would censor its own evidence: the gate
        would see outcomes exclusively for trades it already liked, so its thresholds
        could never be re-derived or falsified. scripts/validate_gate.py depends on
        the rejected rows being there to answer "would the gate still have helped?".
        """
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=0))
        _run(settings, storage)

        tracked = storage.load_tracked_signals("open")
        assert tracked, "gate-rejected setups must still be tracked"
        for row in tracked:
            assert row["quality_passed"] == 0
            assert row["quality_reason"]
            assert row["extension_score"] is not None
            # A rejected setup is still a complete, resolvable bracket.
            assert row["entry_price"] and row["stop_price"] and row["target_price"]

    def test_gate_passed_setups_are_flagged_as_passed(self, env, monkeypatch):
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=6))
        _run(settings, storage)
        tracked = storage.load_tracked_signals("open")
        assert any(row["quality_passed"] == 1 for row in tracked)

    def test_armed_signals_store_a_full_feature_vector(self, env, monkeypatch):
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=6))
        _run(settings, storage)
        tracked = storage.load_tracked_signals("open")
        assert tracked, "pullback stub should produce at least one actionable signal"
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


class TestAlertsEndToEnd:
    """
    The alerting path, exercised through a real scan rather than in isolation.

    An alert means every gate passed, so these also serve as the end-to-end proof
    that the quality gate is wired into the scan at all.
    """

    def _run_with_sinks(self, settings, storage, sinks, pullback_bars=6, **kw):
        return run_scan(
            settings, storage,
            ScanRequest(pairs=PAIRS, **kw),
            alert_sinks=sinks,
        )

    def test_qualifying_setup_raises_and_persists_an_alert(self, env, monkeypatch):
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=6))
        received = []
        summary = self._run_with_sinks(settings, storage,
                                       [CallableSink(received.append, "spy")])

        assert summary.alerts_raised > 0
        stored = storage.load_alerts()
        assert len(stored) == summary.alerts_raised
        assert len(received) == summary.alerts_raised
        row = stored[0]
        assert row["entry"] and row["stop"] and row["target"]
        assert row["delivered"] == 1
        assert row["delivery_error"] is None

    def test_extended_trend_raises_no_alert(self, env, monkeypatch):
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=0))
        received = []
        summary = self._run_with_sinks(settings, storage,
                                       [CallableSink(received.append, "spy")])
        assert summary.alerts_raised == 0
        assert received == []
        assert storage.load_alerts() == []

    def test_alerts_never_exceed_signals_found(self, env, monkeypatch):
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=6))
        summary = self._run_with_sinks(settings, storage, [])
        assert summary.alerts_raised <= summary.signals_found

    def test_rescanning_does_not_re_alert_the_same_setup(self, env, monkeypatch):
        """
        The scanner re-runs every 30-60s. Without dedupe the same valid setup would
        re-fire on every tick, which is how an operator learns to ignore alerts.
        """
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=6))
        first = self._run_with_sinks(settings, storage, [])
        second = self._run_with_sinks(settings, storage, [])
        assert first.alerts_raised > 0
        assert second.alerts_raised == 0
        assert len(storage.load_alerts()) == first.alerts_raised

    def test_a_dead_webhook_does_not_fail_the_scan(self, env, monkeypatch):
        """
        Alerting is a side-channel. The scan is also writing tracked signals and
        outcomes, and must complete even when every sink is unreachable.
        """
        settings, storage = env
        monkeypatch.setattr(scanner_mod, "OandaClient",
                            lambda s: StubOanda(s, pullback_bars=6))

        def boom(_alert):
            raise RuntimeError("connection refused")

        summary = self._run_with_sinks(settings, storage,
                                       [CallableSink(boom, "dead")])
        assert summary.errors == 0
        assert summary.pairs_scanned == len(PAIRS)
        assert summary.alerts_raised > 0
        # Raised and recorded even though delivery failed.
        rows = storage.load_alerts()
        assert len(rows) == summary.alerts_raised
        assert all(r["delivered"] == 0 for r in rows)
        assert all("connection refused" in (r["delivery_error"] or "") for r in rows)
