from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


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
    total_score: float = 0.0
    trade_signal: str = "AVOID"
    signal_reason: str = ""
    risk_notes: str = ""
    as_of: str = ""


class ScanRequest(BaseModel):
    pairs: List[str]
    max_spread_pips: float = 2.0
    signal_mode: str = "All"


class ScanSummary(BaseModel):
    pairs_scanned: int = 0
    errors: int = 0
    signals_found: int = 0
