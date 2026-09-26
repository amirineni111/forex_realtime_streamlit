"""
Entry-quality gate: the filter that decides whether a proposed setup is worth
paying the spread for.

Why this module exists
----------------------
Measured over 1,898 resolved trades (2026-06-10 -> 2026-09-25) the scorer had no
edge: 39.6% win rate against the 40.0% that RR 1.5 needs at zero cost, and against
the 45.0% it needs at the 12.5% cost ratio actually being paid. Expectancy was
-0.082R and the equity curve was monotonically down.

Crucially the failure was *not* in the bracket. Replaying every trade against real
M5 bars at ATR multiples from 1.5x to 5x and reward:risk from 1.0 to 3.0 leaves
**gross** R negative everywhere (-0.10R to -0.18R). No stop or target geometry
rescues an entry that is wrong; widening the stop only shrinks the cost term.

What the data does say is that the entries are systematically on the wrong side of
short-horizon mean reversion. Every momentum-flavoured feature is *inverted* --
higher momentum in the trade's direction predicts a *lower* win rate, monotonically
across quintiles (n=1,003 trades carrying logged features):

    rsi_dir          18-49 -> 48.8% WR / +0.054R  ...  70-91 -> 33.3% WR / -0.320R
    range_pos_dir    bottom -> 43.3% / -0.031R    ...  top   -> 29.9% / -0.377R
    macd_hist_atr    bottom -> 45.8% / +0.005R    ...  top   -> 33.8% / -0.244R
    ema_gap_atr      bottom -> 45.8% / -0.009R    ...  top   -> 33.8% / -0.264R

That is one coherent statement, not four coincidences: **buying extension on M5
loses**. By the time EMA9/EMA20/MACD all agree, the move being detected is the move
that is about to retrace, and the bracket gets tagged on the pullback.

The gate below therefore rejects *extended* entries rather than ranking them. On the
1,003 trades with logged features it keeps 23% and turns -0.149R into +0.057R, and --
the part that matters -- it is positive in each month independently, with the rule
fixed in advance rather than refitted per month:

    2026-07   36 trades   50.0% WR   +0.122R
    2026-08  102 trades   50.0% WR   +0.067R
    2026-09   89 trades   46.1% WR   +0.019R

The decay across those three months is real and is why scripts/validate_gate.py
exists: re-run it as trades accumulate and expect to re-tune. Treat +0.057R as the
optimistic end -- the thresholds were chosen on this same history, so some of that
is fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# -- Extension thresholds ---------------------------------------------------
# Each is the quintile boundary above which measured expectancy turned clearly
# negative, rounded to a round number. They are deliberately *loose* -- the point is
# to reject the extended tail, not to chase the optimum on 1,003 samples.

MAX_RSI_DIR = 60.0        # RSI oriented to the trade; >60 = chasing
MAX_RANGE_POS_DIR = 0.70  # position in session range, +1 = extended in our favour
MAX_MACD_HIST_ATR = 0.20  # MACD histogram in ATR units, signed to the trade
MAX_EMA_GAP_ATR = 0.75    # EMA9-EMA20 gap in ATR units, signed to the trade

# -- Structural exclusions --------------------------------------------------
# Pairs whose spread is too wide for the bracket this system can carry. Measured:
#   GBP_JPY  n=74  25.7% WR  -0.391R   (3.4 pip median spread)
#   USD_HKD  n=34  29.4% WR  -0.229R   (3.5)
#   USD_SGD  n=50  32.0% WR  -0.170R   (3.1)
# These are excluded by name *and* caught generically by the cost veto; the explicit
# list is here so the reason is legible rather than emergent.
EXCLUDED_PAIRS = frozenset({"GBP_JPY", "USD_HKD", "USD_SGD"})

# Hour-of-day exclusions, deliberately empty.
#
# An earlier version excluded {10, 15, 20, 21, 22, 23} on the strength of M5 outcome
# data, where those hours measured -0.28R to -0.66R. Re-deriving them on M15 bars
# (scripts/backtest.py --by-hour, 9 months, 7 majors, exclusions lifted) showed the
# list does not transfer at all — the two worst offenders invert:
#
#              M5 (live outcomes)      M15 (backtest)
#   hour 10        -0.422R               +0.192R   <- M5's worst, M15's best
#   hour 22        (excluded)            +0.091R
#   hour 15        -0.277R               -0.064R
#   hour 19        (not excluded)        -0.152R   <- M15's worst
#
# and removing the filter entirely moved total expectancy by 0.003R (-0.0005 to
# -0.0031R over 1,500-1,800 trades). So the hours were an artifact of the M5 bracket
# being tagged by intrabar noise, not a property of the sessions themselves.
#
# Re-fitting the list to M15 was rejected rather than done: every hourly bucket has
# |t| < 1.7, and picking the worst few out of 24 comparisons is the textbook way to
# manufacture a filter that backtests well and then does nothing. The cost veto and
# the extension gate already price in what the illiquid hours actually cost — wider
# spreads raise cost_ratio, which is a measured input rather than a clock reading.
#
# Populate this only from live outcomes on the current timeframe, via
# scripts/validate_gate.py, and only when a bucket survives a sample worth trusting.
EXCLUDED_HOURS: frozenset = frozenset()

# -- Cost discipline --------------------------------------------------------
# Round-trip spread as a fraction of the risk taken. At RR 1.5 the breakeven win
# rate is (1 + cost_ratio) / (1 + RR), so cost translates directly into the win
# rate the entry must beat:
#     cost 0.125 -> 45.0% required      cost 0.060 -> 42.4% required
#     cost 0.083 -> 43.3% required      cost 0.000 -> 40.0% required
# The system's best measured win rate on gated trades is ~48%, so the cost ratio
# has to stay near or below 0.10 for the margin to be real rather than nominal.
MAX_COST_RATIO = 0.10

# Stop distance as a multiple of the spread. This is the same constraint as
# MAX_COST_RATIO seen from the other side (1 / 0.10 = 10) and is what a caller
# sizing a bracket should use.
MIN_STOP_SPREAD_MULT = 1.0 / MAX_COST_RATIO


@dataclass
class QualityVerdict:
    """Why a setup was accepted or rejected. ``blocks`` non-empty => not tradeable."""

    passed: bool
    blocks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extension_score: float = 0.0   # 0 = calm entry, 1+ = extended on every axis

    @property
    def reason(self) -> str:
        if self.blocks:
            return "; ".join(self.blocks)
        return "; ".join(self.warnings) if self.warnings else "clean entry"


def extension_score(features: dict) -> float:
    """
    How far the entry is chasing an already-made move, on a 0-1+ scale.

    Each of the four measured-inverted features contributes the fraction by which
    it exceeds its threshold, normalised by that threshold's own scale. 0 means
    every axis is calm; 1.0 means the average axis sits at roughly twice its limit.
    Used for ranking and for alert severity -- the hard accept/reject is
    ``evaluate``'s job.
    """
    def over(value: Optional[float], limit: float, scale: float) -> float:
        if value is None:
            return 0.0
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, (v - limit) / scale)

    parts = (
        over(features.get("rsi_dir"), MAX_RSI_DIR, 30.0),
        over(features.get("range_pos_dir"), MAX_RANGE_POS_DIR, 0.30),
        over(features.get("macd_hist_atr"), MAX_MACD_HIST_ATR, 0.30),
        over(features.get("ema_gap_atr"), MAX_EMA_GAP_ATR, 0.75),
    )
    return round(sum(parts) / len(parts), 4)


def evaluate(
    features: dict,
    pair: str,
    hour_utc: Optional[int] = None,
    cost_ratio: Optional[float] = None,
) -> QualityVerdict:
    """
    Accept or reject a proposed entry.

    ``features`` is a dict from ``forex.features.build_features`` -- i.e. already
    oriented to the trade direction, which is what lets one set of thresholds cover
    both longs and shorts. Anything missing is treated as neutral rather than
    fatal, so a partially populated setup degrades to "unknown" instead of raising.
    """
    blocks: List[str] = []
    warnings: List[str] = []
    features = features or {}

    if pair in EXCLUDED_PAIRS:
        blocks.append(f"{pair} excluded (spread too wide for this bracket)")

    if hour_utc is not None and int(hour_utc) in EXCLUDED_HOURS:
        blocks.append(f"{int(hour_utc):02d}:00 UTC is an excluded hour (measured negative)")

    if cost_ratio is not None and cost_ratio > MAX_COST_RATIO:
        blocks.append(
            f"cost {cost_ratio:.1%} of risk exceeds {MAX_COST_RATIO:.0%} "
            f"(needs {(1 + cost_ratio) / 2.5:.1%} win rate to break even)"
        )

    # Extension checks -- the four inverted axes, each blocking on its own.
    rsi_dir = features.get("rsi_dir")
    if rsi_dir is not None and rsi_dir > MAX_RSI_DIR:
        blocks.append(
            f"RSI {rsi_dir:.0f} already extended in trade direction (>{MAX_RSI_DIR:.0f})"
        )

    range_pos = features.get("range_pos_dir")
    if range_pos is not None and range_pos > MAX_RANGE_POS_DIR:
        blocks.append(
            f"price at {range_pos:+.2f} of session range in trade direction "
            f"(>{MAX_RANGE_POS_DIR:.2f}) -- buying the top of the range"
        )

    macd_hist = features.get("macd_hist_atr")
    if macd_hist is not None and macd_hist > MAX_MACD_HIST_ATR:
        blocks.append(
            f"MACD histogram {macd_hist:+.2f}xATR already extended (>{MAX_MACD_HIST_ATR:.2f})"
        )

    ema_gap = features.get("ema_gap_atr")
    if ema_gap is not None and ema_gap > MAX_EMA_GAP_ATR:
        blocks.append(
            f"EMA gap {ema_gap:+.2f}xATR already extended (>{MAX_EMA_GAP_ATR:.2f})"
        )

    # Non-blocking context.
    if features.get("h4_agrees") == -1.0:
        warnings.append("H4 trend opposes the trade")
    if cost_ratio is not None and MAX_COST_RATIO * 0.75 < cost_ratio <= MAX_COST_RATIO:
        warnings.append(
            f"cost {cost_ratio:.1%} of risk is close to the {MAX_COST_RATIO:.0%} limit"
        )

    return QualityVerdict(
        passed=not blocks,
        blocks=blocks,
        warnings=warnings,
        extension_score=extension_score(features),
    )
