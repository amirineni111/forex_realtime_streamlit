from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class ForexBar(BaseModel):
    pair: str
    timeframe: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


class ForexQuote(BaseModel):
    pair: str
    bid: float
    ask: float
    spread_pips: float
    as_of: str


class ForexSnapshot(BaseModel):
    # `model_prob` is a legitimate field name here; opt out of pydantic's
    # "model_" protected-namespace warning rather than renaming it.
    model_config = ConfigDict(protected_namespaces=())

    pair: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread_pips: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    day_change_pct: Optional[float] = None
    rsi14: Optional[float] = None
    ema9: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr14: Optional[float] = None
    adx14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width_pct: Optional[float] = None
    current_session: Optional[str] = None
    session_high: Optional[float] = None
    session_low: Optional[float] = None
    momentum_score: float = 0.0
    reversion_score: float = 0.0
    session_score: float = 0.0
    regime: Optional[str] = None
    total_score: float = 0.0
    trade_signal: str = "AVOID"
    signal_reason: str = ""
    risk_notes: str = ""
    as_of: str = ""
    # Suggested trade levels (ATR-based)
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    stop_pips: Optional[float] = None
    target_pips: Optional[float] = None
    rr_ratio: Optional[float] = None
    # Multi-timeframe confluence
    h1_direction: Optional[str] = None
    h4_direction: Optional[str] = None
    mtf_score: float = 0.0
    mtf_confluence: Optional[str] = None
    # Support/Resistance
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    sr_score: float = 0.0
    at_key_level: bool = False
    sr_levels_json: Optional[str] = None
    # Currency strength
    base_strength: Optional[float] = None
    quote_strength: Optional[float] = None
    strength_assessment: Optional[str] = None
    # Structure / cost / model gating
    blocked_ahead: bool = False
    cost_ratio: Optional[float] = None
    model_prob: Optional[float] = None
    required_prob: Optional[float] = None


class ScanRequest(BaseModel):
    pairs: List[str]
    max_spread_pips: float = 2.0
    signal_mode: str = "All"
    # Bars the signal is computed on. M5 was the original choice and is measurably
    # the wrong one: M5 ATR on a major is 2-4 pips, so a stop wide enough to amortise
    # a ~1.6 pip spread sits 5-10x outside the signal's own horizon. Backtested over
    # 9 months (scripts/backtest.py), M5 setups are vetoed on cost 100% of the time
    # once the cost gate is honest, while M15 carries a positive gross edge.
    signal_timeframe: str = "M15"
    # Hours a tracked bracket may stay open before it is closed at market. Scales with
    # the timeframe: an M15 bracket at 5xATR is ~27 pips out and needs room to resolve.
    max_hold_hours: float = 24.0


class ScanSummary(BaseModel):
    pairs_scanned: int = 0
    errors: int = 0
    signals_found: int = 0
    # Setups that cleared the entry-quality gate and were pushed as alerts. Always
    # ≤ signals_found, and typically far below it — the gate keeps about a fifth of
    # what the rules propose, which is the point of having it.
    alerts_raised: int = 0
