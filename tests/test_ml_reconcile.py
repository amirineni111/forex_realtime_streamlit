"""
Tests for the daily ML prediction mirror and the New York close reconciliation.

The calendar rules and the SQL-Server-hit budget are the parts worth pinning
down: the dashboard refreshes every 60 seconds against a source that updates
once a weekday, so "does this open a connection?" is a correctness question.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from forex.config import AppSettings
from forex.ml_sync import (
    backfill_reconciliation, candle_trading_date, expected_prediction_date,
    implied_direction, last_completed_session_date, live_progress, reconcile_due,
    reconcile_session, score_prediction, sync_due, sync_predictions,
)
from forex.pairs import to_oanda_pair
from forex.storage import Storage

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "test.sqlite3")


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        sql_server="test-server", sql_database="testdb",
        sql_username="u", sql_password="p",
    )


def _prediction(pair="EUR_USD", prediction_date="2026-07-29", target_date="2026-07-30",
                signal="BUY", prob_buy=0.62, prob_sell=0.38, base_close=1.14668):
    return {
        "prediction_date": prediction_date, "pair": pair, "sql_pair": pair.replace("_", ""),
        "target_date": target_date, "predicted_signal": signal, "signal_confidence": max(prob_buy, prob_sell),
        "prob_buy": prob_buy, "prob_sell": prob_sell, "prob_hold": 0.0,
        "base_close": base_close, "model_name": "daily_automation_model",
        "model_version": "5.2_binary_rates", "source_created_at": "2026-07-29T20:57:57",
    }


# ── Pair normalisation ──────────────────────────────────────────────────────

class TestPairNormalisation:
    """SQL Server stores pairs unseparated; OANDA needs EUR_USD."""

    @pytest.mark.parametrize("raw,expected", [
        ("EURUSD", "EUR_USD"), ("eurusd", "EUR_USD"), ("EUR/USD", "EUR_USD"),
        ("EUR_USD", "EUR_USD"), ("EUR-USD", "EUR_USD"), ("USDJPY", "USD_JPY"),
    ])
    def test_known_forms(self, raw, expected):
        assert to_oanda_pair(raw) == expected

    def test_unrecognised_symbol_is_not_mangled(self):
        """An unexpected symbol must surface as itself, not be split into a wrong pair."""
        assert to_oanda_pair("XAUUSDEXTRA") == "XAUUSDEXTRA"


# ── Calendar ────────────────────────────────────────────────────────────────

class TestExpectedPredictionDate:
    """The mirror's target: the newest weekday whose 20:55 ET run has finished."""

    def test_before_the_run_uses_the_previous_weekday(self):
        # Thursday 10:00 ET — Thursday's run has not happened yet.
        assert expected_prediction_date(_et(2026, 7, 30, 10)) == date(2026, 7, 29)

    def test_after_the_run_uses_today(self):
        assert expected_prediction_date(_et(2026, 7, 30, 22)) == date(2026, 7, 30)

    def test_monday_morning_falls_back_to_friday(self):
        assert expected_prediction_date(_et(2026, 8, 3, 9)) == date(2026, 7, 31)

    def test_weekend_falls_back_to_friday(self):
        assert expected_prediction_date(_et(2026, 8, 1, 12)) == date(2026, 7, 31)
        assert expected_prediction_date(_et(2026, 8, 2, 23)) == date(2026, 7, 31)


class TestLastCompletedSession:
    """Sessions close at 17:00 ET and are labelled by the day they close on."""

    def test_before_the_close(self):
        assert last_completed_session_date(_et(2026, 7, 30, 12)) == date(2026, 7, 29)

    def test_after_the_close(self):
        assert last_completed_session_date(_et(2026, 7, 30, 18)) == date(2026, 7, 30)

    def test_weekend_reports_friday(self):
        assert last_completed_session_date(_et(2026, 8, 1, 12)) == date(2026, 7, 31)

    def test_monday_before_close_reports_friday(self):
        assert last_completed_session_date(_et(2026, 8, 3, 9)) == date(2026, 7, 31)


class TestCandleTradingDate:
    """
    Verified against live data: the candle starting 2026-07-29T21:00Z closes at
    1.15280, which is SQL Server's recorded close for 2026-07-30.
    """

    def test_edt_candle_maps_to_next_day(self):
        assert candle_trading_date("2026-07-29T21:00:00.000000000Z") == date(2026, 7, 30)

    def test_est_candle_maps_to_next_day(self):
        # In standard time the same 17:00 ET boundary sits at 22:00Z.
        assert candle_trading_date("2026-01-14T22:00:00.000000000Z") == date(2026, 1, 15)

    def test_unparseable_timestamp(self):
        assert candle_trading_date("not-a-timestamp") is None


# ── Scoring ─────────────────────────────────────────────────────────────────

class TestImpliedDirection:
    def test_lean_from_probabilities(self):
        assert implied_direction(0.62, 0.38) == "UP"
        assert implied_direction(0.38, 0.62) == "DOWN"
        assert implied_direction(0.5, 0.5) is None
        assert implied_direction(None, 0.5) is None


class TestScorePrediction:
    def test_buy_that_rose_is_a_hit(self):
        row = score_prediction(_prediction(signal="BUY"), 1.14668, 1.15280)
        assert row["actual_direction"] == "UP"
        assert row["signal_outcome"] == "HIT"
        assert row["implied_outcome"] == "HIT"
        assert row["actual_return_pct"] == pytest.approx(0.5337, abs=1e-3)
        assert row["actual_pips"] == pytest.approx(61.2, abs=0.2)

    def test_buy_that_fell_is_a_miss(self):
        row = score_prediction(_prediction(signal="BUY"), 1.14668, 1.14000)
        assert row["signal_outcome"] == "MISS"

    def test_sell_that_fell_is_a_hit(self):
        row = score_prediction(
            _prediction(signal="SELL", prob_buy=0.38, prob_sell=0.62), 1.14668, 1.14000
        )
        assert row["signal_outcome"] == "HIT"
        assert row["implied_outcome"] == "HIT"

    def test_hold_is_an_abstention_but_its_lean_is_still_scored(self):
        """
        ~70% of daily rows are HOLD (the confidence gate abstaining). Scoring only
        BUY/SELL would leave most sessions unmeasured, so the underlying
        prob_buy/prob_sell lean is scored separately.
        """
        row = score_prediction(
            _prediction(signal="HOLD", prob_buy=0.52, prob_sell=0.48), 1.14668, 1.15280
        )
        assert row["signal_outcome"] == "ABSTAIN"
        assert row["implied_direction"] == "UP"
        assert row["implied_outcome"] == "HIT"

    def test_jpy_pips_use_the_right_scale(self):
        row = score_prediction(_prediction(pair="USD_JPY"), 157.000, 157.500)
        assert row["actual_pips"] == pytest.approx(50.0, abs=0.1)

    def test_unchanged_close_is_flat_and_unscoreable_for_lean(self):
        row = score_prediction(_prediction(), 1.14668, 1.14668)
        assert row["actual_direction"] == "FLAT"
        assert row["implied_outcome"] == "N/A"


class TestLiveProgress:
    """
    The dashboard's running read of an open session. It must agree with what
    ``score_prediction`` will write at the close, or the box would preview a
    verdict the Reconcile tab then contradicts.
    """

    def test_price_above_the_base_close_tracks_a_buy(self):
        row = live_progress(_prediction(signal="BUY"), 1.15280)
        assert row["current_direction"] == "UP"
        assert row["signal_status"] == "HIT"
        assert row["lean_status"] == "HIT"
        assert row["move_pips"] == pytest.approx(61.2, abs=0.2)
        assert row["move_pct"] == pytest.approx(0.5337, abs=1e-3)

    def test_price_below_the_base_close_runs_against_a_buy(self):
        assert live_progress(_prediction(signal="BUY"), 1.14000)["signal_status"] == "MISS"

    def test_hold_abstains_live_too_but_keeps_a_lean(self):
        row = live_progress(
            _prediction(signal="HOLD", prob_buy=0.52, prob_sell=0.48), 1.15280
        )
        assert row["signal_status"] == "ABSTAIN"
        assert row["lean_status"] == "HIT"

    def test_matches_the_reconcile_verdict_for_the_same_price(self):
        prediction = _prediction(signal="SELL", prob_buy=0.38, prob_sell=0.62)
        settled = score_prediction(prediction, prediction["base_close"], 1.14000)
        running = live_progress(prediction, 1.14000)
        assert running["signal_status"] == settled["signal_outcome"]
        assert running["current_direction"] == settled["actual_direction"]
        assert running["move_pips"] == settled["actual_pips"]

    @pytest.mark.parametrize("price", [None, float("nan"), "n/a"])
    def test_missing_price_yields_no_verdict(self, price):
        row = live_progress(_prediction(), price)
        assert row["current_direction"] is None
        assert row["signal_status"] is None

    def test_missing_base_close_yields_no_verdict(self):
        """A pair with no source close cannot be compared to anything."""
        row = live_progress(_prediction(base_close=None), 1.15280)
        assert row["signal_status"] is None


# ── Sync budget ─────────────────────────────────────────────────────────────

class TestSyncDue:
    """
    The whole point of the mirror: at a 1-minute refresh, an up-to-date mirror
    must produce zero SQL Server connections.
    """

    def test_not_due_when_mirror_has_the_latest_run(self, storage, settings):
        storage.save_ml_predictions([_prediction(prediction_date="2026-07-30")])
        due, reason = sync_due(storage, settings, now=_et(2026, 7, 30, 23))
        assert due is False
        assert "up to date" in reason

    def test_due_when_mirror_is_behind(self, storage, settings):
        storage.save_ml_predictions([_prediction(prediction_date="2026-07-29")])
        due, _ = sync_due(storage, settings, now=_et(2026, 7, 30, 23))
        assert due is True

    def test_due_when_mirror_is_empty(self, storage, settings):
        due, _ = sync_due(storage, settings, now=_et(2026, 7, 30, 23))
        assert due is True

    def test_not_due_when_unconfigured(self, storage):
        due, reason = sync_due(storage, AppSettings(), now=_et(2026, 7, 30, 23))
        assert due is False
        assert "not configured" in reason

    def test_failed_attempt_still_throttles(self, storage, settings):
        """
        Failures are logged as attempts. Without that, an unreachable SQL Server
        would be dialled on every single 60-second refresh.
        """
        storage.log_ml_sync("predictions", "error", 0, None, "connection refused")
        due, reason = sync_due(storage, settings, now=datetime.now(timezone.utc))
        assert due is False
        assert "throttled" in reason

    def test_throttle_expires(self, storage, settings):
        storage.log_ml_sync("predictions", "error", 0, None, "connection refused")
        later = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ml_sync_min_interval_minutes + 1
        )
        due, _ = sync_due(storage, settings, now=later)
        assert due is True


class TestSyncPredictions:
    def test_skips_without_touching_the_source(self, storage, settings, monkeypatch):
        storage.save_ml_predictions([_prediction(prediction_date="2026-07-31")])

        def _boom(*args, **kwargs):
            raise AssertionError("SQL Server must not be contacted when up to date")

        monkeypatch.setattr("forex.ml_sync.fetch_predictions", _boom)
        result = sync_predictions(storage, settings, now=_et(2026, 7, 31, 23))
        assert result["status"] == "skipped"

    def test_source_failure_is_reported_not_raised(self, storage, settings, monkeypatch):
        from forex.sqlserver import MLSourceError

        def _fail(*args, **kwargs):
            raise MLSourceError("server asleep")

        monkeypatch.setattr("forex.ml_sync.fetch_predictions", _fail)
        result = sync_predictions(storage, settings, force=True)
        assert result["status"] == "error"
        assert storage.load_ml_last_sync(kind="predictions")["status"] == "error"

    def test_rows_land_in_the_mirror(self, storage, settings, monkeypatch):
        monkeypatch.setattr(
            "forex.ml_sync.fetch_predictions",
            lambda *a, **k: [_prediction(), _prediction(pair="GBP_USD")],
        )
        result = sync_predictions(storage, settings, force=True)
        assert result["status"] == "ok"
        assert result["rows"] == 2
        assert storage.load_ml_max_prediction_date() == "2026-07-29"
        assert len(storage.load_ml_latest_predictions()) == 2

    def test_rerun_upstream_overwrites_rather_than_duplicating(self, storage, settings):
        storage.save_ml_predictions([_prediction(signal="BUY")])
        storage.save_ml_predictions([_prediction(signal="SELL")])
        rows = storage.load_ml_latest_predictions()
        assert len(rows) == 1
        assert rows[0]["predicted_signal"] == "SELL"


# ── Reconciliation against OANDA ────────────────────────────────────────────

class _Bar:
    def __init__(self, timestamp, close):
        self.timestamp, self.close = timestamp, close


class _StubClient:
    """Daily closes taken from the live EUR_USD series."""

    def __init__(self, closes=None, fail=False):
        self.calls = []
        self.fail = fail
        self.closes = closes or [
            ("2026-07-27T21:00:00.000000000Z", 1.13864),  # trading date 07-28
            ("2026-07-28T21:00:00.000000000Z", 1.14668),  # trading date 07-29
            ("2026-07-29T21:00:00.000000000Z", 1.15280),  # trading date 07-30
        ]

    def get_candles(self, pair, granularity="M5", count=200):
        self.calls.append(pair)
        if self.fail:
            raise RuntimeError("oanda down")
        return [_Bar(ts, close) for ts, close in self.closes]


class TestReconcileSession:
    def test_scores_against_the_previous_sessions_close(self, storage):
        storage.save_ml_predictions([_prediction()])
        client = _StubClient()
        result = reconcile_session(
            storage, client, target_date=date(2026, 7, 30)
        )
        assert result["status"] == "ok"
        assert result["scored"] == 1

        row = storage.load_ml_reconciliation(target_date="2026-07-30")[0]
        assert row["base_close"] == pytest.approx(1.14668)
        assert row["actual_close"] == pytest.approx(1.15280)
        assert row["signal_outcome"] == "HIT"

    def test_already_scored_pairs_cost_zero_oanda_calls(self, storage):
        storage.save_ml_predictions([_prediction()])
        first = _StubClient()
        reconcile_session(storage, first, target_date=date(2026, 7, 30))
        assert first.calls == ["EUR_USD"]

        second = _StubClient()
        result = reconcile_session(storage, second, target_date=date(2026, 7, 30))
        assert second.calls == []
        assert result["status"] == "up_to_date"

    def test_force_rescores(self, storage):
        storage.save_ml_predictions([_prediction()])
        reconcile_session(storage, _StubClient(), target_date=date(2026, 7, 30))
        client = _StubClient()
        result = reconcile_session(
            storage, client, target_date=date(2026, 7, 30), force=True
        )
        assert client.calls == ["EUR_USD"]
        assert result["scored"] == 1

    def test_missing_candle_leaves_the_session_pending(self, storage):
        storage.save_ml_predictions([_prediction(target_date="2026-07-31")])
        result = reconcile_session(storage, _StubClient(), target_date=date(2026, 7, 31))
        assert result["status"] == "pending"
        assert result["pending"] == 1
        assert storage.load_ml_reconciliation(target_date="2026-07-31") == []

    def test_gap_before_the_session_uses_the_last_available_close(self, storage):
        """
        A Monday session's previous close is the preceding Friday's, not
        Sunday's — the base is the last *available* candle, never target-1.
        """
        storage.save_ml_predictions([
            _prediction(prediction_date="2026-07-31", target_date="2026-08-03")
        ])
        client = _StubClient(closes=[
            ("2026-07-30T21:00:00.000000000Z", 1.15306),  # trading date 07-31 (Fri)
            ("2026-08-02T21:00:00.000000000Z", 1.16000),  # trading date 08-03 (Mon)
        ])
        reconcile_session(storage, client, target_date=date(2026, 8, 3))
        row = storage.load_ml_reconciliation(target_date="2026-08-03")[0]
        assert row["base_close"] == pytest.approx(1.15306)

    def test_oanda_failure_does_not_write_partial_results(self, storage):
        storage.save_ml_predictions([_prediction()])
        result = reconcile_session(
            storage, _StubClient(fail=True), target_date=date(2026, 7, 30)
        )
        assert result["status"] == "error"
        assert storage.load_ml_reconciliation(target_date="2026-07-30") == []

    def test_no_predictions_for_the_session(self, storage):
        result = reconcile_session(storage, _StubClient(), target_date=date(2026, 7, 30))
        assert result["status"] == "skipped"

    def test_older_sessions_widen_the_candle_window(self, storage):
        """
        OANDA returns the most recent ``count`` candles. A fixed count would make
        re-evaluating a session from weeks ago quietly find nothing, so the
        window has to scale with how far back the session is.
        """
        seen = {}

        class _Recording(_StubClient):
            def get_candles(inner, pair, granularity="M5", count=200):
                seen[pair] = count
                return [_Bar(ts, c) for ts, c in inner.closes]

        storage.save_ml_predictions([_prediction(target_date="2026-06-01")])
        reconcile_session(
            storage, _Recording(), target_date=date(2026, 6, 1), now=_et(2026, 7, 31, 18)
        )
        # ~60 calendar days back is ~43 sessions; the window must cover them.
        assert seen["EUR_USD"] > 40

    def test_recent_session_keeps_a_small_window(self, storage):
        seen = {}

        class _Recording(_StubClient):
            def get_candles(inner, pair, granularity="M5", count=200):
                seen[pair] = count
                return [_Bar(ts, c) for ts, c in inner.closes]

        storage.save_ml_predictions([_prediction()])
        reconcile_session(
            storage, _Recording(), target_date=date(2026, 7, 30), now=_et(2026, 7, 30, 18)
        )
        assert seen["EUR_USD"] == 12


class TestStaleInstrument:
    """
    OANDA lists USD_INR but its newest daily candle is from October 2022, while
    the daily model still predicts it every weekday. Treated as merely "pending"
    it would consume one request per retry window forever.
    """

    _DELISTED = [("2022-10-20T21:00:00.000000000Z", 83.284)]

    def test_recorded_as_unscoreable_rather_than_pending(self, storage):
        storage.save_ml_predictions([_prediction(pair="USD_INR")])
        result = reconcile_session(
            storage, _StubClient(closes=self._DELISTED), target_date=date(2026, 7, 30)
        )
        assert result["unscoreable"] == 1
        assert result["pending"] == 0

        row = storage.load_ml_reconciliation(target_date="2026-07-30")[0]
        assert row["signal_outcome"] == "NO_DATA"
        assert row["actual_close"] is None

    def test_stops_the_retry_loop(self, storage):
        storage.save_ml_predictions([_prediction(pair="USD_INR")])
        reconcile_session(
            storage, _StubClient(closes=self._DELISTED), target_date=date(2026, 7, 30)
        )
        due, reason = reconcile_due(storage, date(2026, 7, 30))
        assert due is False
        assert "fully reconciled" in reason

    def test_excluded_from_hit_rates(self, storage):
        storage.save_ml_predictions([
            _prediction(pair="EUR_USD", signal="BUY"),
            _prediction(pair="USD_INR", signal="BUY"),
        ])

        class _PerPair(_StubClient):
            def get_candles(inner, pair, granularity="M5", count=200):
                inner.calls.append(pair)
                series = TestStaleInstrument._DELISTED if pair == "USD_INR" else inner.closes
                return [_Bar(ts, close) for ts, close in series]

        reconcile_session(storage, _PerPair(), target_date=date(2026, 7, 30))
        summary = storage.load_ml_reconciliation_by_date()[0]
        assert summary["pairs"] == 1        # USD_INR held out
        assert summary["signal_calls"] == 1
        assert summary["lean_calls"] == 1

    def test_a_merely_late_candle_still_counts_as_pending(self, storage):
        """A session that closed hours ago must not be written off as delisted."""
        storage.save_ml_predictions([_prediction(target_date="2026-07-31")])
        result = reconcile_session(
            storage, _StubClient(), target_date=date(2026, 7, 31)
        )
        assert result["pending"] == 1
        assert result["unscoreable"] == 0


class TestReconcileDue:
    def test_not_due_once_every_pair_is_scored(self, storage):
        storage.save_ml_predictions([_prediction()])
        reconcile_session(storage, _StubClient(), target_date=date(2026, 7, 30))
        due, reason = reconcile_due(storage, date(2026, 7, 30))
        assert due is False
        assert "fully reconciled" in reason

    def test_pending_session_is_throttled_between_attempts(self, storage):
        storage.save_ml_predictions([_prediction(target_date="2026-07-31")])
        reconcile_session(storage, _StubClient(), target_date=date(2026, 7, 31))
        due, reason = reconcile_due(storage, date(2026, 7, 31))
        assert due is False
        assert "throttled" in reason

    def test_due_after_the_retry_window(self, storage):
        from forex.ml_sync import RECONCILE_RETRY_MINUTES

        storage.save_ml_predictions([_prediction(target_date="2026-07-31")])
        reconcile_session(storage, _StubClient(), target_date=date(2026, 7, 31))
        later = datetime.now(timezone.utc) + timedelta(minutes=RECONCILE_RETRY_MINUTES + 1)
        due, _ = reconcile_due(storage, date(2026, 7, 31), now=later)
        assert due is True


class TestBackfill:
    """
    The reason backfill exists as its own function: looping reconcile_session over
    a 60-day window would issue one OANDA request per (session, pair).
    """

    def _seed(self, storage):
        storage.save_ml_predictions([
            _prediction(pair="EUR_USD", prediction_date="2026-07-27", target_date="2026-07-28"),
            _prediction(pair="EUR_USD", prediction_date="2026-07-28", target_date="2026-07-29"),
            _prediction(pair="EUR_USD", prediction_date="2026-07-29", target_date="2026-07-30"),
            _prediction(pair="GBP_USD", prediction_date="2026-07-29", target_date="2026-07-30"),
        ])

    def test_one_request_per_pair_covers_every_session(self, storage):
        self._seed(storage)
        client = _StubClient()
        result = backfill_reconciliation(
            storage, client, now=_et(2026, 7, 30, 18)
        )
        # Three EUR_USD sessions + one GBP_USD session, two requests total.
        assert sorted(client.calls) == ["EUR_USD", "GBP_USD"]
        assert result["status"] == "ok"
        assert len(storage.load_ml_reconciled_dates()) >= 2

    def test_sessions_without_a_prior_close_are_left_pending(self, storage):
        self._seed(storage)
        # The 07-28 session's base would be 07-27, which this series lacks.
        result = backfill_reconciliation(storage, _StubClient(), now=_et(2026, 7, 30, 18))
        assert result["pending"] == 1
        assert "2026-07-28" not in storage.load_ml_reconciled_dates()

    def test_second_run_is_a_no_op(self, storage):
        self._seed(storage)
        backfill_reconciliation(storage, _StubClient(), now=_et(2026, 7, 30, 18))
        client = _StubClient()
        result = backfill_reconciliation(storage, client, now=_et(2026, 7, 30, 18))
        assert result["status"] in ("up_to_date", "pending")
        assert "EUR_USD" not in client.calls or result["scored"] == 0

    def test_open_sessions_are_never_scored(self, storage):
        """Only sessions that have passed 17:00 ET are eligible."""
        self._seed(storage)
        backfill_reconciliation(storage, _StubClient(), now=_et(2026, 7, 30, 12))
        assert "2026-07-30" not in storage.load_ml_reconciled_dates()


class TestGridHeight:
    """
    Grid sizing: a default st.dataframe scrolls past ~10 rows, which is fewer
    than every built-in pair universe.
    """

    def test_fits_the_data_without_scrolling(self):
        from app import GRID_HEADER_PX, GRID_ROW_PX, _grid_height

        assert _grid_height(14, max_rows=20) == GRID_HEADER_PX + 14 * GRID_ROW_PX

    def test_caps_at_the_configured_rows(self):
        from app import GRID_HEADER_PX, GRID_ROW_PX, _grid_height

        assert _grid_height(80, max_rows=20) == GRID_HEADER_PX + 20 * GRID_ROW_PX

    def test_never_leaves_empty_grid_below_short_tables(self):
        """A 3-row table must not render 20 rows of blank space."""
        from app import GRID_HEADER_PX, GRID_ROW_PX, _grid_height

        assert _grid_height(3, max_rows=20) == GRID_HEADER_PX + 3 * GRID_ROW_PX

    def test_empty_table_still_has_a_usable_height(self):
        from app import _grid_height

        assert _grid_height(0, max_rows=20) > 0

    def test_default_covers_the_largest_built_in_universe(self):
        from app import DEFAULT_TABLE_ROWS
        from forex.pairs import UNIVERSE_MAP

        assert DEFAULT_TABLE_ROWS >= max(len(p) for p in UNIVERSE_MAP.values())


# ── Aggregation ─────────────────────────────────────────────────────────────

class TestReconciliationSummaries:
    def test_hit_rates_separate_actioned_signals_from_leans(self, storage):
        storage.save_ml_predictions([
            _prediction(pair="EUR_USD", signal="BUY", prob_buy=0.62, prob_sell=0.38),
            _prediction(pair="GBP_USD", signal="HOLD", prob_buy=0.52, prob_sell=0.48),
        ])
        # EUR_USD rises (BUY hit, lean hit); GBP_USD is scored on lean only.
        reconcile_session(storage, _StubClient(), target_date=date(2026, 7, 30))

        summary = storage.load_ml_reconciliation_by_date()[0]
        assert summary["pairs"] == 2
        assert summary["signal_calls"] == 1   # only the BUY was an actual call
        assert summary["signal_hits"] == 1
        assert summary["lean_calls"] == 2     # both rows carry a directional lean
        assert summary["lean_hits"] == 2
