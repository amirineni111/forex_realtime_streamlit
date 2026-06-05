from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import ForexSnapshot, ForexQuote, ScanSummary

SQLITE_TIMEOUT = 30.0
SQLITE_BUSY_MS = 30000
MAX_SCAN_RUNS = 20


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=SQLITE_TIMEOUT)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_MS}")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS forex_scan_runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at  TEXT    DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT,
                    pairs_scanned INTEGER,
                    summary_json TEXT
                );

                CREATE TABLE IF NOT EXISTS forex_scan_logs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id    INTEGER,
                    pair       TEXT,
                    signal     TEXT,
                    error      TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS forex_snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id         INTEGER NOT NULL,
                    pair            TEXT,
                    bid             REAL,
                    ask             REAL,
                    mid             REAL,
                    spread_pips     REAL,
                    open            REAL,
                    high            REAL,
                    low             REAL,
                    close           REAL,
                    day_change_pct  REAL,
                    rsi14           REAL,
                    ema9            REAL,
                    ema20           REAL,
                    ema50           REAL,
                    macd            REAL,
                    macd_signal     REAL,
                    macd_histogram  REAL,
                    atr14           REAL,
                    bb_upper        REAL,
                    bb_middle       REAL,
                    bb_lower        REAL,
                    bb_width_pct    REAL,
                    current_session TEXT,
                    session_high    REAL,
                    session_low     REAL,
                    momentum_score  REAL,
                    reversion_score REAL,
                    session_score   REAL,
                    total_score     REAL,
                    trade_signal    TEXT,
                    signal_reason   TEXT,
                    risk_notes      TEXT,
                    as_of           TEXT
                );

                CREATE TABLE IF NOT EXISTS forex_quotes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair        TEXT,
                    bid         REAL,
                    ask         REAL,
                    spread_pips REAL,
                    as_of       TEXT
                );

                CREATE TABLE IF NOT EXISTS forex_watchlist (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair         TEXT NOT NULL,
                    signal       TEXT,
                    entry_price  REAL,
                    target_price REAL,
                    stop_price   REAL,
                    stop_pips    REAL,
                    target_pips  REAL,
                    notes        TEXT,
                    status       TEXT DEFAULT 'watching',
                    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                    closed_at    TEXT
                );
            """)

    # ── Scan run lifecycle ──────────────────────────────────────────────────

    def start_scan(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO forex_scan_runs (started_at) VALUES (CURRENT_TIMESTAMP)"
            )
            return cur.lastrowid

    def finish_scan(self, scan_id: int, summary: ScanSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE forex_scan_runs SET finished_at=CURRENT_TIMESTAMP, "
                "pairs_scanned=?, summary_json=? WHERE id=?",
                (summary.pairs_scanned, summary.model_dump_json(), scan_id),
            )
            # Prune old runs
            conn.execute(
                "DELETE FROM forex_snapshots WHERE scan_id NOT IN "
                f"(SELECT id FROM forex_scan_runs ORDER BY id DESC LIMIT {MAX_SCAN_RUNS})"
            )
            conn.execute(
                "DELETE FROM forex_scan_logs WHERE scan_id NOT IN "
                f"(SELECT id FROM forex_scan_runs ORDER BY id DESC LIMIT {MAX_SCAN_RUNS})"
            )
            conn.execute(
                f"DELETE FROM forex_scan_runs WHERE id NOT IN "
                f"(SELECT id FROM forex_scan_runs ORDER BY id DESC LIMIT {MAX_SCAN_RUNS})"
            )

    def log_pair(self, scan_id: int, pair: str, signal: Optional[str], error: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO forex_scan_logs (scan_id, pair, signal, error) VALUES (?,?,?,?)",
                (scan_id, pair, signal, error),
            )

    # ── Snapshots ───────────────────────────────────────────────────────────

    def save_snapshots(self, scan_id: int, snapshots: List[ForexSnapshot]) -> None:
        rows = []
        for s in snapshots:
            rows.append((
                scan_id, s.pair, s.bid, s.ask, s.mid, s.spread_pips,
                s.open, s.high, s.low, s.close, s.day_change_pct,
                s.rsi14, s.ema9, s.ema20, s.ema50,
                s.macd, s.macd_signal, s.macd_histogram,
                s.atr14, s.bb_upper, s.bb_middle, s.bb_lower, s.bb_width_pct,
                s.current_session, s.session_high, s.session_low,
                s.momentum_score, s.reversion_score, s.session_score, s.total_score,
                s.trade_signal, s.signal_reason, s.risk_notes, s.as_of,
            ))
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO forex_snapshots "
                "(scan_id,pair,bid,ask,mid,spread_pips,open,high,low,close,day_change_pct,"
                "rsi14,ema9,ema20,ema50,macd,macd_signal,macd_histogram,"
                "atr14,bb_upper,bb_middle,bb_lower,bb_width_pct,"
                "current_session,session_high,session_low,"
                "momentum_score,reversion_score,session_score,total_score,"
                "trade_signal,signal_reason,risk_notes,as_of) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_latest_snapshots(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forex_snapshots WHERE scan_id = "
                "(SELECT MAX(id) FROM forex_scan_runs WHERE finished_at IS NOT NULL) "
                "ORDER BY total_score DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def load_scan_logs(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forex_scan_logs WHERE scan_id = "
                "(SELECT MAX(id) FROM forex_scan_runs WHERE finished_at IS NOT NULL) "
                "ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def load_latest_scan_run(self) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM forex_scan_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ── Live quotes ─────────────────────────────────────────────────────────

    def save_quotes(self, quotes: List[ForexQuote]) -> None:
        rows = [(q.pair, q.bid, q.ask, q.spread_pips, q.as_of) for q in quotes]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO forex_quotes (pair,bid,ask,spread_pips,as_of) VALUES (?,?,?,?,?)",
                rows,
            )
            # Keep only last 500 rows
            conn.execute(
                "DELETE FROM forex_quotes WHERE id NOT IN "
                "(SELECT id FROM forex_quotes ORDER BY id DESC LIMIT 500)"
            )

    def load_latest_quotes(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT q.* FROM forex_quotes q "
                "INNER JOIN ("
                "  SELECT pair, MAX(id) AS max_id FROM forex_quotes GROUP BY pair"
                ") latest ON q.pair=latest.pair AND q.id=latest.max_id "
                "ORDER BY q.pair"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Watchlist ────────────────────────────────────────────────────────────

    def add_watchlist(
        self, pair: str, signal: str, entry: float,
        target: float, stop: float, stop_pips: float,
        target_pips: float, notes: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO forex_watchlist "
                "(pair,signal,entry_price,target_price,stop_price,stop_pips,target_pips,notes) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (pair, signal, entry, target, stop, stop_pips, target_pips, notes),
            )

    def close_watchlist(self, row_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE forex_watchlist SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE id=?",
                (row_id,),
            )

    def load_watchlist(self, status: str = "watching") -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forex_watchlist WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        return [dict(r) for r in rows]
