"""
Tests for the entry-quality gate and the alert pipeline.

The thresholds in ``forex.quality`` were derived from 1,898 resolved trades, so the
tests here encode the *direction* of each measured effect rather than the exact
constant — the constants are expected to move as data accumulates, the sign of the
relationship is what must not silently flip.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from forex import quality
from forex.alerts import (
    Alert, CallableSink, URGENCY_HIGH, URGENCY_NORMAL, build_alert,
    classify_urgency, dispatch, format_digest,
)
from forex.storage import Storage


def _features(**over) -> dict:
    """A calm, gate-passing feature vector."""
    base = {
        "rsi_dir": 50.0,
        "range_pos_dir": 0.0,
        "macd_hist_atr": 0.0,
        "ema_gap_atr": 0.1,
        "h4_agrees": 1.0,
    }
    base.update(over)
    return base


# ── Extension gate ─────────────────────────────────────────────────────────

class TestExtensionGate:
    """
    Every momentum feature measured *inverted*: more momentum in the trade's
    direction predicted a lower win rate, monotonically across quintiles. The gate
    must therefore reject extension, not reward it.
    """

    def test_calm_entry_passes(self):
        v = quality.evaluate(_features(), pair="EUR_USD", hour_utc=12, cost_ratio=0.05)
        assert v.passed
        assert v.blocks == []
        assert v.extension_score == 0.0

    @pytest.mark.parametrize("field,value", [
        ("rsi_dir", 75.0),
        ("range_pos_dir", 0.95),
        ("macd_hist_atr", 0.60),
        ("ema_gap_atr", 1.50),
    ])
    def test_each_extension_axis_blocks_on_its_own(self, field, value):
        v = quality.evaluate(_features(**{field: value}), pair="EUR_USD",
                             hour_utc=12, cost_ratio=0.05)
        assert not v.passed
        assert len(v.blocks) == 1

    def test_extension_score_rises_with_extension(self):
        calm = quality.evaluate(_features(), "EUR_USD", 12, 0.05).extension_score
        mild = quality.evaluate(_features(rsi_dir=70.0), "EUR_USD", 12, 0.05).extension_score
        wild = quality.evaluate(
            _features(rsi_dir=90.0, range_pos_dir=1.0, macd_hist_atr=0.8, ema_gap_atr=2.0),
            "EUR_USD", 12, 0.05,
        ).extension_score
        assert calm < mild < wild

    def test_short_side_uses_the_same_thresholds(self):
        """
        Features arrive already oriented to the trade direction, which is what lets
        one threshold set cover both sides. A short into a collapsing market has a
        *high* rsi_dir just as a long into a spike does.
        """
        v = quality.evaluate(_features(rsi_dir=80.0), pair="USD_JPY", hour_utc=12,
                             cost_ratio=0.05)
        assert not v.passed


class TestStructuralExclusions:
    def test_wide_spread_pairs_are_excluded(self):
        v = quality.evaluate(_features(), pair="GBP_JPY", hour_utc=12, cost_ratio=0.05)
        assert not v.passed
        assert "GBP_JPY" in v.reason

    def test_hour_exclusions_are_empty_by_default(self):
        """
        The hour list is deliberately empty. It was populated from M5 outcomes and
        did not survive re-derivation on M15 — hour 10 was M5's worst bucket
        (-0.422R) and M15's best (+0.192R). Anything added back has to come from
        live outcomes on the timeframe actually being traded.
        """
        assert quality.EXCLUDED_HOURS == frozenset()
        for hour in range(24):
            assert quality.evaluate(_features(), "EUR_USD", hour, 0.05).passed

    def test_hour_exclusions_are_honoured_when_populated(self):
        """The mechanism still works — only the default list is empty."""
        original = quality.EXCLUDED_HOURS
        quality.EXCLUDED_HOURS = frozenset({21})
        try:
            assert not quality.evaluate(_features(), "EUR_USD", 21, 0.05).passed
            assert quality.evaluate(_features(), "EUR_USD", 13, 0.05).passed
        finally:
            quality.EXCLUDED_HOURS = original

    def test_cost_above_the_limit_blocks(self):
        v = quality.evaluate(_features(), "EUR_USD", 12, cost_ratio=0.20)
        assert not v.passed
        assert "cost" in v.reason.lower()

    def test_cost_near_the_limit_warns_without_blocking(self):
        v = quality.evaluate(_features(), "EUR_USD", 12,
                             cost_ratio=quality.MAX_COST_RATIO * 0.9)
        assert v.passed
        assert v.warnings

    def test_missing_inputs_degrade_to_neutral_rather_than_raising(self):
        v = quality.evaluate({}, pair="EUR_USD", hour_utc=None, cost_ratio=None)
        assert v.passed
        assert v.extension_score == 0.0

    def test_cost_limit_implies_the_stop_multiple(self):
        assert quality.MIN_STOP_SPREAD_MULT == pytest.approx(1.0 / quality.MAX_COST_RATIO)


# ── Alerts ─────────────────────────────────────────────────────────────────

class _Snap:
    """Minimal stand-in for a scored ForexSnapshot."""

    def __init__(self, **kw):
        defaults = dict(
            pair="EUR_USD", trade_signal="BUY_CANDIDATE", suggested_entry=1.1000,
            suggested_stop=1.0985, suggested_target=1.1022, stop_pips=15.0,
            target_pips=22.5, rr_ratio=1.5, spread_pips=1.2, cost_ratio=0.08,
            total_score=65.0, regime="TREND", current_session="London_NY_Overlap",
            as_of="2026-09-25T13:00:00.000000000Z",
        )
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestBuildAlert:
    def test_gate_failure_produces_no_alert(self):
        verdict = quality.evaluate(_features(rsi_dir=90.0), "EUR_USD", 12, 0.05)
        assert build_alert(_Snap(), verdict) is None

    def test_non_actionable_signal_produces_no_alert(self):
        verdict = quality.evaluate(_features(), "EUR_USD", 12, 0.05)
        assert build_alert(_Snap(trade_signal="WATCH_ONLY"), verdict) is None
        assert build_alert(_Snap(trade_signal="AVOID"), verdict) is None

    def test_missing_levels_produce_no_alert(self):
        verdict = quality.evaluate(_features(), "EUR_USD", 12, 0.05)
        assert build_alert(_Snap(suggested_stop=None), verdict) is None

    def test_passing_setup_produces_an_actionable_alert(self):
        verdict = quality.evaluate(_features(), "EUR_USD", 12, 0.05)
        alert = build_alert(_Snap(), verdict)
        assert alert is not None
        assert alert.direction == 1
        assert alert.side == "LONG"
        assert alert.entry and alert.stop and alert.target
        assert "EUR_USD" in alert.headline()

    def test_short_signal_sets_negative_direction(self):
        verdict = quality.evaluate(_features(), "USD_JPY", 12, 0.05)
        alert = build_alert(_Snap(pair="USD_JPY", trade_signal="STRONG_SHORT"), verdict)
        assert alert.direction == -1
        assert alert.side == "SHORT"

    def test_urgency_needs_both_a_calm_entry_and_a_strong_score(self):
        assert classify_urgency(0.0, 70) == URGENCY_HIGH
        assert classify_urgency(0.0, 45) == URGENCY_NORMAL   # calm but weak
        assert classify_urgency(0.4, 90) == URGENCY_NORMAL   # strong but stretched

    def test_body_states_the_win_rate_the_cost_implies(self):
        verdict = quality.evaluate(_features(), "EUR_USD", 12, 0.05)
        body = build_alert(_Snap(cost_ratio=0.10), verdict).body()
        assert "win rate" in body
        assert "%" in body


class TestDispatch:
    def test_every_alert_reaches_every_sink(self):
        seen_a, seen_b = [], []
        alerts = [
            build_alert(_Snap(), quality.evaluate(_features(), "EUR_USD", 12, 0.05)),
            build_alert(_Snap(pair="GBP_USD"),
                        quality.evaluate(_features(), "GBP_USD", 12, 0.05)),
        ]
        report = dispatch(alerts, [CallableSink(seen_a.append, "a"),
                                   CallableSink(seen_b.append, "b")])
        assert len(seen_a) == len(seen_b) == 2
        assert report["sent"] == 4
        assert report["errors"] == []

    def test_a_failing_sink_does_not_stop_the_others(self):
        """
        Alerting is a side-channel. A dead webhook must not abort a scan that is
        also writing tracked signals and outcomes.
        """
        delivered = []

        def boom(_alert):
            raise RuntimeError("webhook down")

        alert = build_alert(_Snap(), quality.evaluate(_features(), "EUR_USD", 12, 0.05))
        report = dispatch([alert], [CallableSink(boom, "bad"),
                                    CallableSink(delivered.append, "good")])
        assert delivered == [alert]
        assert report["sent"] == 1
        assert len(report["errors"]) == 1
        assert "webhook down" in report["errors"][0]

    def test_digest_summarises_without_raising_on_empty(self):
        assert "No qualifying setups" in format_digest([])
        alert = build_alert(_Snap(), quality.evaluate(_features(), "EUR_USD", 12, 0.05))
        assert "EUR_USD" in format_digest([alert])


# ── Alert persistence / dedupe ─────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "alerts.sqlite3")


def _alert(pair="EUR_USD", direction=1) -> Alert:
    verdict = quality.evaluate(_features(), pair, 12, 0.05)
    return build_alert(_Snap(pair=pair,
                             trade_signal="BUY_CANDIDATE" if direction > 0
                             else "STRONG_SHORT"), verdict)


class TestAlertStorage:
    def test_alert_round_trips(self, store):
        alert_id = store.record_alert(_alert())
        assert alert_id
        rows = store.load_alerts()
        assert len(rows) == 1
        assert rows[0]["pair"] == "EUR_USD"
        assert json.loads(rows[0]["payload_json"])["signal"] == "BUY_CANDIDATE"

    def test_same_setup_inside_the_cooldown_is_suppressed(self, store):
        """
        The scanner re-runs every 30-60s and the same setup stays valid across many
        ticks. Without dedupe it would re-fire on every one, which is exactly how
        someone learns to ignore the alerts.
        """
        assert store.record_alert(_alert()) is not None
        assert store.record_alert(_alert()) is None
        assert len(store.load_alerts()) == 1

    def test_opposite_direction_is_a_different_setup(self, store):
        assert store.record_alert(_alert(direction=1)) is not None
        assert store.record_alert(_alert(direction=-1)) is not None
        assert len(store.load_alerts()) == 2

    def test_a_different_pair_is_not_deduped(self, store):
        assert store.record_alert(_alert(pair="EUR_USD")) is not None
        assert store.record_alert(_alert(pair="GBP_USD")) is not None

    def test_zero_cooldown_allows_re_alerting(self, store):
        assert store.record_alert(_alert(), cooldown_minutes=0) is not None
        assert store.record_alert(_alert(), cooldown_minutes=0) is not None

    def test_delivery_failure_is_recorded_against_the_alert(self, store):
        alert_id = store.record_alert(_alert())
        store.mark_alert_delivered(alert_id, error="webhook 500")
        row = store.load_alerts()[0]
        assert row["delivered"] == 0
        assert row["delivery_error"] == "webhook 500"

    def test_successful_delivery_clears_the_error(self, store):
        alert_id = store.record_alert(_alert())
        store.mark_alert_delivered(alert_id)
        row = store.load_alerts()[0]
        assert row["delivered"] == 1
        assert row["delivery_error"] is None

    def test_acknowledge_stamps_the_row(self, store):
        alert_id = store.record_alert(_alert())
        store.acknowledge_alert(alert_id)
        assert store.load_alerts()[0]["acknowledged_at"] is not None


class TestAlertResolution:
    """An alert closes on its own bracket and freezes the exit price for the feed."""

    @staticmethod
    def _armed(store, minutes_ago=30):
        alert_id = store.record_alert(_alert())
        with store._connect() as conn:
            conn.execute(
                "UPDATE forex_alerts SET created_at=datetime('now', ?) WHERE id=?",
                (f"-{minutes_ago} minutes", alert_id),
            )
        return store.load_alerts()[0]

    @staticmethod
    def _bar(minutes_ago, high, low, close):
        ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        return {"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "high": high, "low": low, "close": close}

    def test_new_alert_starts_open(self, store):
        assert self._armed(store)["status"] == "open"

    def test_target_touch_closes_at_target(self, store):
        row = self._armed(store)
        mid = (row["stop"] + row["target"]) / 2
        bars = [self._bar(40, mid, mid, mid),                 # before the alert
                self._bar(20, row["target"] + 1e-4, mid, mid)]
        assert store.evaluate_alerts("EUR_USD", bars) == 1
        closed = store.load_alerts()[0]
        assert closed["status"] == "closed"
        assert closed["exit_reason"] == "TARGET"
        assert closed["exit_price"] == row["target"]

    def test_stop_wins_when_one_bar_spans_both(self, store):
        row = self._armed(store)
        bars = [self._bar(40, row["entry"], row["entry"], row["entry"]),
                self._bar(20, row["target"] + 1e-4, row["stop"] - 1e-4, row["entry"])]
        store.evaluate_alerts("EUR_USD", bars)
        assert store.load_alerts()[0]["exit_reason"] == "STOP"

    def test_untouched_alert_stays_open(self, store):
        row = self._armed(store)
        bars = [self._bar(40, row["entry"], row["entry"], row["entry"]),
                self._bar(20, row["entry"], row["entry"], row["entry"])]
        assert store.evaluate_alerts("EUR_USD", bars) == 0
        assert store.load_alerts()[0]["status"] == "open"

    def test_bars_before_the_alert_are_ignored(self, store):
        row = self._armed(store)
        bars = [self._bar(40, row["target"] + 1e-4, row["entry"], row["entry"]),
                self._bar(20, row["entry"], row["entry"], row["entry"])]
        assert store.evaluate_alerts("EUR_USD", bars) == 0

    def test_aged_out_alert_without_bar_coverage_expires_without_a_price(self, store):
        self._armed(store, minutes_ago=24 * 60)
        row = store.load_alerts()[0]
        bars = [self._bar(20, row["entry"], row["entry"], row["entry"])]
        assert store.evaluate_alerts("EUR_USD", bars, max_hold_hours=12) == 1
        closed = store.load_alerts()[0]
        assert closed["exit_reason"] == "EXPIRED"
        assert closed["exit_price"] is None
