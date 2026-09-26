"""
Regression tests for the signal-quality fixes and the learning loop.

Several of these encode bugs that were measured in production data rather than
hypothesised, so the docstrings carry the evidence that motivated them.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from forex import signals
from forex.features import FEATURE_NAMES, FEATURE_VERSION, build_features, to_vector
from forex.model import (
    ForexModel, build_matrix, fit, gated_performance, roc_auc, top_decile_precision,
    walk_forward,
)
from forex.signals import (
    _MAX_COST_RATIO, _mtf_confluence, _sr_proximity, breakeven_win_rate, score_pair,
)
from forex.storage import Storage


# ── MTF confluence ─────────────────────────────────────────────────────────

class TestMTFConfluence:
    """
    The M5 direction used to be passed in as one of three confluence votes, so a
    setup whose higher timeframes were both NEUTRAL scored FULL/+30 on its own
    say-so. 103 of 143 historical FULL rows had a NEUTRAL or absent higher
    timeframe. Those free points are most of what let mediocre setups clear the
    70-point STRONG threshold.
    """

    def test_both_higher_timeframes_neutral_is_not_confirmation(self):
        score, label = _mtf_confluence("LONG", "NEUTRAL", "NEUTRAL")
        assert score == 0.0
        assert label == "UNCONFIRMED"

    def test_missing_higher_timeframes_is_not_confirmation(self):
        assert _mtf_confluence("LONG", None, None) == (0.0, "UNCONFIRMED")

    def test_single_agreeing_timeframe_is_partial_not_full(self):
        assert _mtf_confluence("LONG", "LONG", "NEUTRAL") == (15.0, "PARTIAL")
        assert _mtf_confluence("SHORT", "NEUTRAL", "SHORT") == (15.0, "PARTIAL")

    def test_full_requires_both_timeframes_agreeing(self):
        assert _mtf_confluence("LONG", "LONG", "LONG") == (30.0, "FULL")
        assert _mtf_confluence("SHORT", "SHORT", "SHORT") == (30.0, "FULL")

    def test_conflict_and_opposition_score_zero(self):
        assert _mtf_confluence("LONG", "LONG", "SHORT") == (0.0, "CONFLICT")
        assert _mtf_confluence("LONG", "SHORT", "SHORT") == (0.0, "OPPOSED")

    def test_neutral_dominant_scores_nothing(self):
        assert _mtf_confluence("NEUTRAL", "LONG", "LONG") == (0.0, "NONE")


# ── Support / resistance ───────────────────────────────────────────────────

class TestSRDirectionAwareness:
    """
    The at-a-level bonus used to fire regardless of trade direction, so a long
    pinned under resistance scored the same +25 as a long bouncing off support.
    10 of 11 historical STRONG_BUY rows were at_key_level with mean sr_score 24.1
    — the bonus was manufacturing the top tier.
    """

    LEVELS = [
        {"price": 1.1000, "type": "S", "touches": 3, "strength": 3.0},
        {"price": 1.1060, "type": "R", "touches": 3, "strength": 3.0},
    ]
    ATR = 0.0010

    def test_long_off_support_with_clear_runway_is_rewarded(self):
        score, _, at_level, blocked, _, _ = _sr_proximity(1.1002, self.ATR, self.LEVELS, "LONG")
        assert score == 25.0
        assert at_level is True
        assert blocked is False

    def test_long_pinned_under_resistance_is_penalised_not_rewarded(self):
        levels = [
            {"price": 1.1000, "type": "S", "touches": 3, "strength": 3.0},
            {"price": 1.1008, "type": "R", "touches": 3, "strength": 3.0},
        ]
        score, reason, _, blocked, _, _ = _sr_proximity(1.1002, self.ATR, levels, "LONG")
        assert blocked is True
        assert score < 25.0
        assert "BLOCKED" in reason

    def test_short_into_support_is_blocked(self):
        levels = [
            {"price": 1.1055, "type": "S", "touches": 3, "strength": 3.0},
            {"price": 1.1060, "type": "R", "touches": 3, "strength": 3.0},
        ]
        _, _, _, blocked, _, _ = _sr_proximity(1.1058, self.ATR, levels, "SHORT")
        assert blocked is True

    def test_short_off_resistance_with_runway_is_rewarded(self):
        score, _, at_level, blocked, _, _ = _sr_proximity(1.1058, self.ATR, self.LEVELS, "SHORT")
        assert score == 25.0
        assert at_level is True
        assert blocked is False

    def test_neutral_direction_scores_nothing(self):
        score, _, at_level, blocked, _, _ = _sr_proximity(1.1030, self.ATR, self.LEVELS, "NEUTRAL")
        assert score == 0.0 and at_level is False and blocked is False


# ── Cost arithmetic ────────────────────────────────────────────────────────

class TestCostMath:
    def test_zero_cost_breakeven_is_exactly_forty_percent(self):
        """
        The measured 39.56% win rate over 910 trades was not underperformance —
        it is precisely the breakeven rate of a 1.5 RR bracket, i.e. the
        signature of random entry.
        """
        assert breakeven_win_rate(1.5, 0.0) == pytest.approx(0.40, abs=1e-9)

    def test_cost_raises_the_bar(self):
        # At the historically measured 0.237 cost ratio you need ~49.5% to break even.
        assert breakeven_win_rate(1.5, 0.237) == pytest.approx(0.4948, abs=1e-3)

    def test_breakeven_is_monotonic_in_cost(self):
        rates = [breakeven_win_rate(1.5, c) for c in (0.0, 0.05, 0.10, 0.15, 0.25)]
        assert rates == sorted(rates)


def _indicators(**over):
    base = {
        "close": 1.1000, "rsi14": 55.0, "ema9": 1.1005, "ema20": 1.0995,
        "macd": 0.0004, "macd_histogram": 0.0002, "atr14": 0.0010, "adx14": 30.0,
        "bb_upper": 1.1030, "bb_middle": 1.1000, "bb_lower": 1.0970,
        "bb_width_pct": 0.55, "session_high": 1.1020, "session_low": 1.0980,
        "day_change_pct": 0.15,
    }
    base.update(over)
    return base


class TestScorePairGating:
    def test_wide_spread_relative_to_stop_is_vetoed(self):
        """
        The cost veto has to be able to actually fire.

        It used to be unreachable: the stop was defined as ``max(..., 8 x spread)``,
        which bounds cost_ratio at 1/8 = 0.125 by construction, so the ``> 0.15``
        veto was dead code and every wide-spread setup was silently handed a stop
        wide enough to make its cost look acceptable. Measured effect: the spread
        floor bound on 772 of 846 trades, pinning cost at exactly 12.5% of risk
        against a gross edge of -0.05R.

        Widening is now capped at _MAX_STOP_WIDEN_RATIO x the timeframe's bracket
        scale, so a spread the pair's *current volatility* cannot amortise leaves
        cost_ratio above the limit and the setup is rejected rather than re-bracketed.
        A 10-pip spread against a 2-pip ATR is exactly that case.
        """
        out = score_pair(
            pair="USD_HKD", bid=1.0995, ask=1.1005, spread_pips=10.0,
            indicators=_indicators(atr14=0.0002), session="London",
            max_spread_pips=12.0, h1_direction="LONG", h4_direction="LONG",
        )
        assert out["cost_ratio"] is not None
        assert out["cost_ratio"] > _MAX_COST_RATIO
        assert out["trade_signal"] == "AVOID"
        assert "cost" in out["signal_reason"].lower()

    def test_stop_widens_toward_the_cost_limit_when_volatility_alone_is_too_tight(self):
        """
        A spread that the volatility stop would not amortise widens the stop to the
        point where cost lands exactly on the limit — provided that point is inside
        the ceiling.
        """
        out = score_pair(
            pair="EUR_USD", bid=1.0999, ask=1.1001, spread_pips=2.0,
            indicators=_indicators(atr14=0.0002), session="London_NY_Overlap",
            h1_direction="LONG", h4_direction="LONG",
        )
        # 2 pip spread at a 10% cost limit wants a 20-pip stop; the M15 bracket
        # scale on a 2-pip ATR is only 10 pips, so cost is what set the size.
        assert out["prov_stop_pips"] == pytest.approx(20.0, rel=1e-3)
        assert out["cost_ratio"] == pytest.approx(_MAX_COST_RATIO, rel=1e-3)

    def test_stop_never_widens_past_the_volatility_ceiling(self):
        """
        Widening is bounded: a stop far outside the pair's current volatility is no
        longer the same trade, so cost is amortised only up to the ceiling and the
        setup is vetoed beyond it.
        """
        atr = 0.0002
        out = score_pair(
            pair="EUR_USD", bid=1.0995, ask=1.1005, spread_pips=5.0,
            indicators=_indicators(atr14=atr), session="London_NY_Overlap",
            max_spread_pips=8.0, h1_direction="LONG", h4_direction="LONG",
        )
        pip = 0.0001
        ceiling_pips = (signals._MAX_STOP_WIDEN_RATIO
                        * signals.stop_atr_mult() * atr / pip)
        assert out["prov_stop_pips"] <= ceiling_pips + 1e-6
        assert out["cost_ratio"] > _MAX_COST_RATIO
        assert out["trade_signal"] == "AVOID"

    def test_bracket_scale_is_per_timeframe(self):
        """
        ATR is a property of the bar size, so one multiple cannot serve every
        timeframe. M5 ATR on a major is 2-4 pips; M15 is ~3x that. Backtesting over
        9 months put the M15 optimum at 5xATR, where a ~1.6 pip spread is ~6% of
        risk instead of the 12.5% the old M5 bracket was pinned at.
        """
        assert signals.stop_atr_mult("M15") > signals.stop_atr_mult("M5")
        assert signals.stop_atr_mult() == signals.stop_atr_mult("M15")
        assert signals.stop_atr_mult("nonsense") == signals._STOP_ATR_MULT

        atr = 0.0010
        kw = dict(pair="EUR_USD", bid=1.0999, ask=1.1001, spread_pips=1.0,
                  indicators=_indicators(atr14=atr), session="London_NY_Overlap",
                  h1_direction="LONG", h4_direction="LONG")
        m5 = score_pair(timeframe="M5", **kw)
        m15 = score_pair(timeframe="M15", **kw)
        assert m15["prov_stop_pips"] > m5["prov_stop_pips"]

    def test_cost_ratio_is_reported_for_actionable_setups(self):
        out = score_pair(
            pair="EUR_USD", bid=1.0999, ask=1.1001, spread_pips=1.0,
            indicators=_indicators(), session="London_NY_Overlap",
            h1_direction="LONG", h4_direction="LONG",
        )
        assert out["cost_ratio"] == pytest.approx(1.0 / out["prov_stop_pips"], rel=1e-3)

    def test_model_probability_below_requirement_downgrades_to_watch_only(self):
        kwargs = dict(
            pair="EUR_USD", bid=1.0999, ask=1.1001, spread_pips=1.0,
            indicators=_indicators(), session="London_NY_Overlap",
            h1_direction="LONG", h4_direction="LONG",
        )
        actionable = score_pair(**kwargs)
        assert actionable["trade_signal"] in ("STRONG_BUY", "BUY_CANDIDATE")

        vetoed = score_pair(**kwargs, model_prob=0.20)
        assert vetoed["trade_signal"] == "WATCH_ONLY"
        assert "P(win)" in vetoed["signal_reason"]

    def test_high_model_probability_leaves_signal_actionable(self):
        out = score_pair(
            pair="EUR_USD", bid=1.0999, ask=1.1001, spread_pips=1.0,
            indicators=_indicators(), session="London_NY_Overlap",
            h1_direction="LONG", h4_direction="LONG", model_prob=0.80,
        )
        assert out["trade_signal"] in ("STRONG_BUY", "BUY_CANDIDATE")

    # A plain short needs a score over 45 without the MTF confluence that would
    # promote it to STRONG_SHORT, which the suppression deliberately spares.
    _PLAIN_SHORT = dict(
        pair="EUR_USD", bid=1.0969, ask=1.0971, spread_pips=1.0,
        session="London_NY_Overlap", h1_direction="NEUTRAL", h4_direction="NEUTRAL",
    )

    @staticmethod
    def _bearish():
        return _indicators(
            close=1.0970, rsi14=38.0, ema9=1.0965, ema20=1.0985,
            macd=-0.0004, macd_histogram=-0.0002, day_change_pct=-0.15,
        )

    def test_plain_short_setup_is_suppressed_to_watch_only(self, monkeypatch):
        out = score_pair(indicators=self._bearish(), **self._PLAIN_SHORT)
        assert out["total_score"] >= 45          # would otherwise be actionable
        assert out["trade_signal"] == "WATCH_ONLY"
        assert "suppressed" in out["signal_reason"]

        # Same setup with the switch off is a SHORT_CANDIDATE, so the downgrade is
        # attributable to the suppression rather than to the fixture being weak.
        monkeypatch.setattr(signals, "_SUPPRESS_SHORT_CANDIDATE", False)
        assert score_pair(
            indicators=self._bearish(), **self._PLAIN_SHORT
        )["trade_signal"] == "SHORT_CANDIDATE"

    def test_strong_short_survives_the_suppression(self):
        # STRONG_SHORT carries the MTF gate and was not measured as losing, so the
        # switch must not take it down with the plain shorts.
        out = score_pair(
            pair="EUR_USD", bid=1.0969, ask=1.0971, spread_pips=1.0,
            indicators=self._bearish(), session="London_NY_Overlap",
            h1_direction="SHORT", h4_direction="SHORT",
        )
        assert out["trade_signal"] == "STRONG_SHORT"
        assert out["suggested_stop"] is not None

    def test_suppressed_short_carries_no_trade_levels(self):
        out = score_pair(indicators=self._bearish(), **self._PLAIN_SHORT)
        # WATCH_ONLY rows are not tracked, so emitting a stop/target would put an
        # untracked bracket in front of the user.
        assert out["trade_signal"] == "WATCH_ONLY"
        assert out["suggested_stop"] is None
        assert out["suggested_target"] is None

    def test_strength_bonus_is_inside_total_score(self):
        without = score_pair(
            pair="EUR_USD", bid=1.0999, ask=1.1001, spread_pips=1.0,
            indicators=_indicators(), session="London", h1_direction="LONG",
        )
        with_bonus = score_pair(
            pair="EUR_USD", bid=1.0999, ask=1.1001, spread_pips=1.0,
            indicators=_indicators(), session="London", h1_direction="LONG",
            strength_bonus=10.0,
        )
        assert with_bonus["total_score"] == pytest.approx(without["total_score"] + 10.0)


# ── Features ───────────────────────────────────────────────────────────────

class TestFeatures:
    SNAP = {
        "close": 1.1000, "atr14": 0.0010, "adx14": 28.0, "rsi14": 62.0,
        "ema9": 1.1006, "ema20": 1.0996, "macd_histogram": 0.0003,
        "bb_width_pct": 0.5, "spread_pips": 1.0, "stop_pips": 12.0,
        "total_score": 70.0, "momentum_score": 40.0, "reversion_score": 0.0,
        "session_score": 10.0, "mtf_score": 30.0, "sr_score": 15.0,
        "nearest_support": 1.0980, "nearest_resistance": 1.1040,
        "h1_direction": "LONG", "h4_direction": "LONG",
        "base_strength": 80.0, "quote_strength": 20.0,
        "session_high": 1.1020, "session_low": 1.0980,
        "day_change_pct": 0.3, "current_session": "London_NY_Overlap",
        "as_of": "2026-07-27T13:30:00.000000000Z",
    }

    def test_vector_matches_contract_length(self):
        v = to_vector(build_features(self.SNAP, 1))
        assert len(v) == len(FEATURE_NAMES)
        assert all(np.isfinite(v))

    def test_higher_timeframe_agreement_flips_with_direction(self):
        long_f = build_features(self.SNAP, 1)
        short_f = build_features(self.SNAP, -1)
        assert long_f["h1_agrees"] == 1.0 and long_f["h4_agrees"] == 1.0
        assert short_f["h1_agrees"] == -1.0 and short_f["h4_agrees"] == -1.0

    def test_structure_is_direction_relative(self):
        """For a long the level ahead is resistance; for a short it is support."""
        long_f = build_features(self.SNAP, 1)
        short_f = build_features(self.SNAP, -1)
        # long: resistance 40 pips up = 4 ATR ahead, support 20 pips down = 2 ATR behind
        assert long_f["dist_to_target_level_atr"] == pytest.approx(4.0, abs=0.05)
        assert long_f["dist_to_protective_level_atr"] == pytest.approx(2.0, abs=0.05)
        # short mirrors them
        assert short_f["dist_to_target_level_atr"] == pytest.approx(2.0, abs=0.05)
        assert short_f["dist_to_protective_level_atr"] == pytest.approx(4.0, abs=0.05)

    def test_rsi_is_oriented_to_the_trade(self):
        assert build_features(self.SNAP, 1)["rsi_dir"] == 62.0
        assert build_features(self.SNAP, -1)["rsi_dir"] == 38.0

    def test_atr_is_scale_free(self):
        """
        Raw ATR is ~100x larger on JPY pairs in price units; feeding it in unscaled
        would let the model use it as a pair identifier.
        """
        jpy = dict(self.SNAP, close=150.00, atr14=0.15)
        eur = dict(self.SNAP, close=1.1000, atr14=0.0011)
        assert build_features(jpy, 1)["atr_pct"] == pytest.approx(
            build_features(eur, 1)["atr_pct"], abs=0.02
        )

    def test_missing_inputs_produce_finite_neutral_values(self):
        v = to_vector(build_features({"close": None, "atr14": None}, 1))
        assert len(v) == len(FEATURE_NAMES)
        assert all(np.isfinite(v))

    def test_bad_timestamp_does_not_raise(self):
        f = build_features(dict(self.SNAP, as_of="not-a-timestamp"), 1)
        assert f["hour_sin"] == 0.0 and f["hour_cos"] == 0.0


# ── Model ──────────────────────────────────────────────────────────────────

class TestModel:
    @staticmethod
    def _synthetic(n=800, signal=True, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, len(FEATURE_NAMES)))
        if signal:
            w = np.zeros(len(FEATURE_NAMES))
            w[[1, 5, 9]] = [1.1, -0.7, 0.5]
            p = 1 / (1 + np.exp(-(X @ w - 0.3)))
        else:
            p = np.full(n, 0.4)
        y = (rng.uniform(size=n) < p).astype(float)
        r = np.where(y == 1, 1.5, -1.0)
        return X, y, r

    def test_learns_a_real_signal(self):
        X, y, r = self._synthetic(signal=True)
        wf = walk_forward(X, y, r, folds=4)
        assert wf["auc"] > 0.65

    def test_reports_no_skill_on_noise(self):
        """The critical guard: a model that 'finds' signal in noise is worse than none."""
        X, y, r = self._synthetic(signal=False, seed=7)
        wf = walk_forward(X, y, r, folds=4)
        assert 0.40 < wf["auc"] < 0.60

    def test_walk_forward_predictions_are_out_of_sample(self):
        X, y, r = self._synthetic(signal=True)
        wf = walk_forward(X, y, r, folds=4)
        # Every OOS prediction comes from a model trained only on earlier rows.
        assert len(wf["oos_p"]) == len(wf["oos_y"]) == len(wf["oos_r"])
        assert len(wf["oos_p"]) < len(y)   # the initial training block is never scored

    def test_json_roundtrip_is_exact(self):
        X, y, _ = self._synthetic()
        m = fit(X, y)
        m2 = ForexModel.from_json(m.to_json())
        assert m2.predict_proba_vector(X[0]) == pytest.approx(m.predict_proba_vector(X[0]))
        assert m2.feature_version == FEATURE_VERSION

    def test_probabilities_stay_in_range(self):
        X, y, _ = self._synthetic()
        m = fit(X, y)
        probs = [m.predict_proba_vector(row) for row in X[:200]]
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_predict_proba_accepts_a_feature_dict(self):
        X, y, _ = self._synthetic()
        m = fit(X, y)
        feats = build_features(TestFeatures.SNAP, 1)
        assert 0.0 <= m.predict_proba(feats) <= 1.0

    def test_auc_handles_ties_and_single_class(self):
        assert roc_auc([1, 1, 1], [0.5, 0.5, 0.5]) is None
        assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)

    def test_gating_improves_expectancy_when_the_model_has_skill(self):
        X, y, r = self._synthetic(signal=True)
        wf = walk_forward(X, y, r, folds=4)
        g = gated_performance(wf["oos_p"], wf["oos_r"], 0.55)
        assert g["expectancy_r"] > g["baseline_expectancy_r"]

    def test_build_matrix_labels_wins_as_one(self):
        rows = [
            {"features": build_features(TestFeatures.SNAP, 1), "outcome": "WIN", "r_multiple": 1.5},
            {"features": build_features(TestFeatures.SNAP, -1), "outcome": "LOSS", "r_multiple": -1.0},
        ]
        X, y, r = build_matrix(rows)
        assert X.shape == (2, len(FEATURE_NAMES))
        assert y.tolist() == [1.0, 0.0]
        assert r.tolist() == [1.5, -1.0]


# ── Storage / evaluation loop ──────────────────────────────────────────────

def _bars(start: datetime, count: int, high, low, close=None):
    out = []
    for i in range(count):
        ts = (start + timedelta(minutes=5 * (i + 1))).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        out.append({
            "timestamp": ts, "open": 1.1000,
            "high": high(i), "low": low(i),
            "close": close(i) if close else 1.1000,
        })
    return out


@pytest.fixture
def store(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "test.sqlite3")


class TestTrackingAndEvaluation:
    T0 = datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc)
    TS0 = "2026-07-27T08:00:00.000000000Z"

    def _arm(self, store, **over):
        kwargs = dict(
            pair="EUR_USD", signal="BUY_CANDIDATE", direction=1,
            entry=1.1000, stop=1.0988, target=1.1018,
            stop_pips=12.0, target_pips=18.0, atr14=0.0008, entry_ts=self.TS0,
            features=build_features(TestFeatures.SNAP, 1),
            feature_version=FEATURE_VERSION,
            model_prob=0.55, required_prob=0.45, cost_ratio=0.083,
            spread_pips=1.0, total_score=72.0, adx14=28.0,
            regime="TREND", session="London_NY_Overlap",
        )
        kwargs.update(over)
        return store.record_tracked_signal(**kwargs)

    def test_features_are_persisted_at_arm_time(self, store):
        tid = self._arm(store)
        assert tid is not None
        row = store.load_tracked_signals("open")[0]
        feats = json.loads(row["features_json"])
        assert set(feats) == set(FEATURE_NAMES)
        assert row["feature_version"] == FEATURE_VERSION
        assert row["model_prob"] == 0.55

    def test_model_mode_is_persisted_at_arm_time(self, store):
        self._arm(store, model_mode="shadow")
        row = store.load_tracked_signals("open")[0]
        # Without this, shadow rows and gated rows pool together in the report and
        # the censored sample silently flatters the model.
        assert row["model_mode"] == "shadow"

    def test_target_hit_records_a_linked_net_of_cost_win(self, store):
        tid = self._arm(store)
        bars = _bars(self.T0, 6, high=lambda i: 1.1020 if i == 2 else 1.1005,
                     low=lambda i: 1.0995)
        assert store.evaluate_tracked_signals("EUR_USD", bars) == 1

        out = store.load_trade_outcomes()[0]
        assert out["outcome"] == "WIN"
        assert out["tracking_id"] == tid
        assert out["exit_reason"] == "TARGET"
        assert out["gross_pips"] == pytest.approx(18.0, abs=0.2)
        assert out["cost_pips"] == pytest.approx(1.0)
        # R is computed from net pips, so the spread shows up in the result.
        assert out["net_pips"] == pytest.approx(17.0, abs=0.2)
        assert out["r_multiple"] == pytest.approx(17.0 / 12.0, abs=0.01)

    def test_hold_minutes_is_real_trade_duration(self, store):
        """
        hold_minutes used to be (now - created_at), i.e. how long until a scan
        happened to evaluate the row — which is why wins and losses both averaged
        ~506 minutes in the recorded data. It must measure entry bar -> exit bar.
        """
        self._arm(store)
        bars = _bars(self.T0, 6, high=lambda i: 1.1020 if i == 2 else 1.1005,
                     low=lambda i: 1.0995)
        store.evaluate_tracked_signals("EUR_USD", bars)
        out = store.load_trade_outcomes()[0]
        # entry 08:00, target touched on the 3rd forward bar = 08:15
        assert out["hold_minutes"] == 15

    def test_stop_is_checked_before_target_within_a_bar(self, store):
        """Conservative resolution: a bar spanning both levels counts as a loss."""
        self._arm(store)
        bars = _bars(self.T0, 3, high=lambda i: 1.1020, low=lambda i: 1.0980)
        store.evaluate_tracked_signals("EUR_USD", bars)
        out = store.load_trade_outcomes()[0]
        assert out["outcome"] == "LOSS"
        assert out["exit_reason"] == "STOP"

    def test_short_stop_and_target_resolve_on_the_right_sides(self, store):
        self._arm(store, signal="SHORT_CANDIDATE", direction=-1,
                  stop=1.1012, target=1.0982)
        bars = _bars(self.T0, 4, high=lambda i: 1.1005,
                     low=lambda i: 1.0980 if i == 1 else 1.0995)
        store.evaluate_tracked_signals("EUR_USD", bars)
        out = store.load_trade_outcomes()[0]
        assert out["outcome"] == "WIN" and out["exit_reason"] == "TARGET"

    def test_unresolved_signal_stays_open(self, store):
        self._arm(store)
        bars = _bars(self.T0, 4, high=lambda i: 1.1005, low=lambda i: 1.0995)
        assert store.evaluate_tracked_signals("EUR_USD", bars) == 0
        assert len(store.load_tracked_signals("open")) == 1

    def test_cooldown_blocks_immediate_rearm(self, store):
        assert self._arm(store) is not None
        assert self._arm(store) is None      # same pair+direction still open

    def test_resolved_trade_becomes_training_data(self, store):
        """The whole point of the loop: an outcome joins back to its own features."""
        self._arm(store)
        bars = _bars(self.T0, 6, high=lambda i: 1.1020 if i == 2 else 1.1005,
                     low=lambda i: 1.0995)
        store.evaluate_tracked_signals("EUR_USD", bars)

        rows = store.load_training_rows(feature_version=FEATURE_VERSION)
        assert len(rows) == 1
        assert rows[0]["outcome"] == "WIN"
        assert set(rows[0]["features"]) == set(FEATURE_NAMES)

        X, y, r = build_matrix(rows)
        assert X.shape == (1, len(FEATURE_NAMES))
        assert y[0] == 1.0

    def test_rows_without_features_are_excluded_from_training(self, store):
        """Legacy trades must not be imputed into the training set."""
        self._arm(store, features=None, feature_version=None)
        bars = _bars(self.T0, 6, high=lambda i: 1.1020 if i == 2 else 1.1005,
                     low=lambda i: 1.0995)
        store.evaluate_tracked_signals("EUR_USD", bars)
        assert store.load_training_rows(feature_version=FEATURE_VERSION) == []


class TestModelStore:
    def test_save_and_activate_round_trips(self, store):
        X = np.random.default_rng(0).normal(size=(300, len(FEATURE_NAMES)))
        y = (np.random.default_rng(1).uniform(size=300) < 0.4).astype(float)
        m = fit(X, y)
        metrics = {"algo": "logistic_l2", "feature_version": FEATURE_VERSION,
                   "n_train": 300, "n_test": 100, "auc": 0.55, "brier": 0.24,
                   "top_decile_prec": 0.5, "base_rate": 0.4}
        mid = store.save_model(m.to_json(), metrics, activate=True)
        assert mid > 0
        loaded = ForexModel.from_json(store.load_active_model_json())
        assert loaded.predict_proba_vector(X[0]) == pytest.approx(m.predict_proba_vector(X[0]))

    def test_only_one_model_is_active(self, store):
        X = np.random.default_rng(0).normal(size=(300, len(FEATURE_NAMES)))
        y = (np.random.default_rng(1).uniform(size=300) < 0.4).astype(float)
        metrics = {"algo": "logistic_l2", "feature_version": FEATURE_VERSION}
        store.save_model(fit(X, y).to_json(), metrics, activate=True)
        store.save_model(fit(X, y).to_json(), metrics, activate=True)
        active = [m for m in store.load_models() if m["is_active"]]
        assert len(active) == 1

    def test_inactive_model_is_not_served(self, store):
        X = np.random.default_rng(0).normal(size=(300, len(FEATURE_NAMES)))
        y = (np.random.default_rng(1).uniform(size=300) < 0.4).astype(float)
        store.save_model(fit(X, y).to_json(), {"feature_version": FEATURE_VERSION},
                         activate=False)
        assert store.load_active_model_json() is None
