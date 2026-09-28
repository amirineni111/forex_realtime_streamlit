from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import ForexSnapshot, ForexQuote, ScanSummary
from .timeutil import parse_ts

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

                CREATE TABLE IF NOT EXISTS forex_trade_outcomes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id INTEGER NOT NULL,
                    pair         TEXT,
                    signal       TEXT,
                    entry_price  REAL,
                    exit_price   REAL,
                    exit_pips    REAL,
                    r_multiple   REAL,
                    outcome      TEXT,
                    hold_minutes INTEGER,
                    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS forex_signal_tracking (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair         TEXT NOT NULL,
                    signal       TEXT,
                    direction    INTEGER,
                    entry_price  REAL,
                    stop_price   REAL,
                    target_price REAL,
                    stop_pips    REAL,
                    target_pips  REAL,
                    atr14        REAL,
                    entry_ts     TEXT,
                    status       TEXT DEFAULT 'open',
                    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS forex_performance_stats (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    computed_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                    dimension       TEXT,
                    dimension_value TEXT,
                    trades          INTEGER,
                    wins            INTEGER,
                    win_rate        REAL,
                    avg_r           REAL,
                    expectancy      REAL
                );

                -- Trained direction models. Coefficients are stored inline so the
                -- scanner can serve predictions without a model file on disk, and so
                -- every historical model stays auditable against its own metrics.
                CREATE TABLE IF NOT EXISTS forex_models (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
                    feature_version  INTEGER,
                    algo             TEXT,
                    n_train          INTEGER,
                    n_test           INTEGER,
                    auc              REAL,
                    brier            REAL,
                    top_decile_prec  REAL,
                    base_rate        REAL,
                    model_json       TEXT,
                    metrics_json     TEXT,
                    is_active        INTEGER DEFAULT 0,
                    notes            TEXT
                );

                -- Local mirror of the daily end-of-day ML predictions produced by
                -- the sqlserver_copilot_forex repo. Mirrored rather than queried
                -- live because the source changes once per weekday while this
                -- dashboard refreshes every minute.
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    prediction_date   TEXT NOT NULL,  -- close the prediction was made FROM
                    pair              TEXT NOT NULL,  -- OANDA form, e.g. EUR_USD
                    sql_pair          TEXT,           -- source form, e.g. EURUSD
                    target_date       TEXT,           -- session the prediction is FOR
                    predicted_signal  TEXT,           -- BUY / SELL / HOLD(=abstain)
                    signal_confidence REAL,
                    prob_buy          REAL,
                    prob_sell         REAL,
                    prob_hold         REAL,
                    base_close        REAL,           -- source close at prediction_date
                    model_name        TEXT,
                    model_version     TEXT,
                    source_created_at TEXT,
                    synced_at         TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (prediction_date, pair)
                );

                -- Every attempt to reach SQL Server / OANDA for ML data, including
                -- failures. Throttling reads this: without logging failures, an
                -- unreachable server would be dialled on every 60s refresh.
                -- Setups that cleared the entry-quality gate and were pushed as
                -- near-real-time alerts. Kept separately from forex_signal_tracking
                -- because an alert is a *notification* event: it has its own dedupe
                -- window and its own delivery status, and one tracked signal can
                -- legitimately produce no alert (gate rejected it) while an alert
                -- always corresponds to something the gate passed.
                CREATE TABLE IF NOT EXISTS forex_alerts (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    pair            TEXT NOT NULL,
                    direction       INTEGER,
                    signal          TEXT,
                    urgency         TEXT,
                    entry           REAL,
                    stop            REAL,
                    target          REAL,
                    stop_pips       REAL,
                    target_pips     REAL,
                    rr_ratio        REAL,
                    spread_pips     REAL,
                    cost_ratio      REAL,
                    total_score     REAL,
                    regime          TEXT,
                    session         TEXT,
                    extension_score REAL,
                    reason          TEXT,
                    payload_json    TEXT,
                    delivered       INTEGER DEFAULT 0,
                    delivery_error  TEXT,
                    acknowledged_at TEXT
                );

                CREATE TABLE IF NOT EXISTS ml_sync_log (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind                TEXT,
                    attempted_at        TEXT DEFAULT CURRENT_TIMESTAMP,
                    status              TEXT,
                    rows_synced         INTEGER,
                    max_prediction_date TEXT,
                    message             TEXT
                );

                -- One scored prediction per (session, pair), evaluated at the
                -- 17:00 ET New York close against OANDA daily candles.
                CREATE TABLE IF NOT EXISTS ml_reconciliation (
                    target_date       TEXT NOT NULL,
                    pair              TEXT NOT NULL,
                    prediction_date   TEXT,
                    predicted_signal  TEXT,
                    implied_direction TEXT,
                    signal_confidence REAL,
                    prob_buy          REAL,
                    prob_sell         REAL,
                    sql_base_close    REAL,
                    base_close        REAL,
                    actual_close      REAL,
                    actual_return_pct REAL,
                    actual_pips       REAL,
                    actual_direction  TEXT,
                    signal_outcome    TEXT,   -- HIT / MISS / ABSTAIN
                    implied_outcome   TEXT,   -- HIT / MISS / N/A
                    evaluated_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (target_date, pair)
                );
            """)

        # Additive column migrations, safe to re-run on every startup.
        migrations = {
            "forex_snapshots": [
                ("h1_direction", "TEXT"),
                ("h4_direction", "TEXT"),
                ("mtf_score", "REAL DEFAULT 0"),
                ("mtf_confluence", "TEXT"),
                ("nearest_support", "REAL"),
                ("nearest_resistance", "REAL"),
                ("sr_score", "REAL DEFAULT 0"),
                ("at_key_level", "INTEGER DEFAULT 0"),
                ("sr_levels_json", "TEXT"),
                ("base_strength", "REAL"),
                ("quote_strength", "REAL"),
                ("strength_assessment", "TEXT"),
                ("adx14", "REAL"),
                ("regime", "TEXT"),
                ("suggested_entry", "REAL"),
                ("suggested_stop", "REAL"),
                ("suggested_target", "REAL"),
                ("stop_pips", "REAL"),
                ("target_pips", "REAL"),
                ("rr_ratio", "REAL"),
                ("blocked_ahead", "INTEGER DEFAULT 0"),
                ("cost_ratio", "REAL"),
                ("model_prob", "REAL"),
                ("required_prob", "REAL"),
            ],
            # The feature snapshot at arm time. Without this the outcome rows are
            # unlearnable — 910 labelled trades were previously recorded with every
            # input discarded, which is what blocked any model work.
            "forex_signal_tracking": [
                ("features_json", "TEXT"),
                ("feature_version", "INTEGER"),
                ("model_prob", "REAL"),
                ("required_prob", "REAL"),
                ("cost_ratio", "REAL"),
                ("spread_pips", "REAL"),
                ("total_score", "REAL"),
                ("adx14", "REAL"),
                ("regime", "TEXT"),
                ("session", "TEXT"),
                # Which mode produced model_prob. Rows scored by a gating model are a
                # censored sample (only trades it allowed exist), so they cannot be
                # pooled with shadow rows when judging the model.
                ("model_mode", "TEXT"),
                # The entry-quality verdict at arm time. Setups the gate rejected are
                # still tracked — arming only what passes would censor the ledger the
                # same way an active model censors its own evidence, leaving the gate's
                # thresholds impossible to re-derive or falsify later.
                ("quality_passed", "INTEGER"),
                ("quality_reason", "TEXT"),
                ("extension_score", "REAL"),
            ],
            # Shadow mode: a model scores every setup and logs its probability but
            # never vetoes. An active model only ever gets outcomes for the trades it
            # allowed, so its own gate censors the evidence needed to judge it.
            "forex_models": [
                ("is_shadow", "INTEGER DEFAULT 0"),
            ],
            # An alert's own lifecycle, resolved against its own bracket. Not borrowed
            # from forex_signal_tracking: that ledger dedupes on open pair+direction, so
            # the tracked row can carry older levels than the alert that was pushed.
            "forex_alerts": [
                ("status", "TEXT DEFAULT 'open'"),
                ("exit_price", "REAL"),
                ("exit_reason", "TEXT"),
                ("exit_ts", "TEXT"),
            ],
            # tracking_id closes the loop: an outcome can now be joined back to the
            # exact feature vector that produced it.
            "forex_trade_outcomes": [
                ("tracking_id", "INTEGER"),
                ("gross_pips", "REAL"),
                ("cost_pips", "REAL"),
                ("net_pips", "REAL"),
                ("exit_ts", "TEXT"),
                ("exit_reason", "TEXT"),
            ],
        }
        for table, cols in migrations.items():
            for col, typedef in cols:
                try:
                    with self._connect() as conn:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass  # column already exists

        # Indexes go last — they reference columns added by the migrations above.
        with self._connect() as conn:
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_tracking_status
                    ON forex_signal_tracking(status, pair);
                CREATE INDEX IF NOT EXISTS idx_outcomes_tracking
                    ON forex_trade_outcomes(tracking_id);
                CREATE INDEX IF NOT EXISTS idx_models_active
                    ON forex_models(is_active, created_at);
                CREATE INDEX IF NOT EXISTS idx_ml_predictions_target
                    ON ml_predictions(target_date);
                CREATE INDEX IF NOT EXISTS idx_ml_sync_log_kind
                    ON ml_sync_log(kind, attempted_at);
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
                # MTF confluence
                s.h1_direction, s.h4_direction, s.mtf_score, s.mtf_confluence,
                # Support/Resistance
                s.nearest_support, s.nearest_resistance, s.sr_score,
                int(s.at_key_level), s.sr_levels_json,
                # Currency strength
                s.base_strength, s.quote_strength, s.strength_assessment,
                # Regime + suggested trade levels
                s.adx14, s.regime,
                s.suggested_entry, s.suggested_stop, s.suggested_target,
                s.stop_pips, s.target_pips, s.rr_ratio,
                # Structure / cost / model gating
                int(s.blocked_ahead), s.cost_ratio, s.model_prob, s.required_prob,
            ))
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO forex_snapshots "
                "(scan_id,pair,bid,ask,mid,spread_pips,open,high,low,close,day_change_pct,"
                "rsi14,ema9,ema20,ema50,macd,macd_signal,macd_histogram,"
                "atr14,bb_upper,bb_middle,bb_lower,bb_width_pct,"
                "current_session,session_high,session_low,"
                "momentum_score,reversion_score,session_score,total_score,"
                "trade_signal,signal_reason,risk_notes,as_of,"
                "h1_direction,h4_direction,mtf_score,mtf_confluence,"
                "nearest_support,nearest_resistance,sr_score,at_key_level,sr_levels_json,"
                "base_strength,quote_strength,strength_assessment,"
                "adx14,regime,suggested_entry,suggested_stop,suggested_target,"
                "stop_pips,target_pips,rr_ratio,"
                "blocked_ahead,cost_ratio,model_prob,required_prob) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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

    def close_watchlist_with_outcome(self, row_id: int, exit_price: float) -> None:
        """Close a watchlist entry, compute P&L, and record the trade outcome."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM forex_watchlist WHERE id=?", (row_id,)
            ).fetchone()
        if not row:
            return
        row = dict(row)

        pair = row.get("pair", "")
        signal = row.get("signal") or ""
        entry_price = row.get("entry_price") or 0.0
        stop_pips = row.get("stop_pips") or 0.0
        created_at = row.get("created_at") or ""

        pip_value = 0.01 if "JPY" in pair else 0.0001
        direction = -1 if signal in ("STRONG_SHORT", "SHORT_CANDIDATE") else 1
        exit_pips = round((exit_price - entry_price) * direction / pip_value, 1) if entry_price else 0.0
        r_multiple = round(exit_pips / stop_pips, 2) if stop_pips and stop_pips > 0 else None
        outcome = "WIN" if exit_pips > 0 else ("LOSS" if exit_pips < 0 else "BREAKEVEN")

        hold_minutes: Optional[int] = None
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            hold_minutes = int((datetime.now(timezone.utc) - created).total_seconds() / 60)
        except Exception:
            pass

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO forex_trade_outcomes "
                "(watchlist_id,pair,signal,entry_price,exit_price,exit_pips,r_multiple,outcome,hold_minutes) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (row_id, pair, signal, entry_price, exit_price, exit_pips, r_multiple, outcome, hold_minutes),
            )
            conn.execute(
                "UPDATE forex_watchlist SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE id=?",
                (row_id,),
            )
        self.compute_and_save_performance()

    def compute_and_save_performance(self) -> None:
        """Recompute and save aggregated performance stats from all trade outcomes."""
        with self._connect() as conn:
            all_rows = [dict(r) for r in conn.execute("SELECT * FROM forex_trade_outcomes").fetchall()]
        if not all_rows:
            return

        def _stats(rows: list) -> Optional[dict]:
            if not rows:
                return None
            wins = sum(1 for r in rows if r["outcome"] == "WIN")
            n = len(rows)
            win_rate = round(wins / n, 3)
            # avg_r = mean R of winners; expectancy = mean R across ALL trades (per-trade edge)
            win_rs = [r["r_multiple"] for r in rows if r["outcome"] == "WIN" and r["r_multiple"] is not None]
            all_rs = [r["r_multiple"] for r in rows if r["r_multiple"] is not None]
            avg_r = round(sum(win_rs) / len(win_rs), 2) if win_rs else 0.0
            expectancy = round(sum(all_rs) / len(all_rs), 3) if all_rs else 0.0
            return {"trades": n, "wins": wins, "win_rate": win_rate, "avg_r": avg_r, "expectancy": expectancy}

        insert_rows = []
        for dimension, key in [("pair", "pair"), ("signal", "signal")]:
            for value in set(r[key] for r in all_rows if r.get(key)):
                subset = [r for r in all_rows if r.get(key) == value]
                s = _stats(subset)
                if s:
                    insert_rows.append((dimension, value, s["trades"], s["wins"], s["win_rate"], s["avg_r"], s["expectancy"]))
        s = _stats(all_rows)
        if s:
            insert_rows.append(("overall", "all", s["trades"], s["wins"], s["win_rate"], s["avg_r"], s["expectancy"]))

        with self._connect() as conn:
            conn.execute("DELETE FROM forex_performance_stats")
            conn.executemany(
                "INSERT INTO forex_performance_stats "
                "(dimension,dimension_value,trades,wins,win_rate,avg_r,expectancy) VALUES (?,?,?,?,?,?,?)",
                insert_rows,
            )

    def load_trade_outcomes(self, limit: int = 200) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forex_trade_outcomes ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def load_performance_by_dimension(self, dimension: str) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forex_performance_stats WHERE dimension=? ORDER BY win_rate DESC",
                (dimension,),
            ).fetchall()
        return [dict(r) for r in rows]

    def load_watchlist(self, status: str = "watching") -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forex_watchlist WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Automatic signal tracking / calibration ─────────────────────────────

    REARM_COOLDOWN_MINUTES = 45

    def record_tracked_signal(
        self, pair: str, signal: str, direction: int,
        entry: float, stop: float, target: float,
        stop_pips: float, target_pips: float, atr14: float, entry_ts: str,
        features: Optional[dict] = None,
        feature_version: Optional[int] = None,
        model_prob: Optional[float] = None,
        required_prob: Optional[float] = None,
        cost_ratio: Optional[float] = None,
        spread_pips: Optional[float] = None,
        total_score: Optional[float] = None,
        adx14: Optional[float] = None,
        regime: Optional[str] = None,
        session: Optional[str] = None,
        model_mode: Optional[str] = None,
        quality_passed: Optional[bool] = None,
        quality_reason: Optional[str] = None,
        extension_score: Optional[float] = None,
    ) -> Optional[int]:
        """
        Record an actionable signal for hands-off forward evaluation. Skips if an
        open signal already exists for this pair+direction (avoids re-arming every scan),
        or if one was armed within the cooldown window — without this, every scan after
        a stop-out immediately re-enters the same chop and racks up correlated losses.

        ``features`` is the model feature vector captured **at arm time**. Storing it
        here (rather than recomputing later from a snapshot that has since been pruned)
        is what makes the outcome learnable. Returns the new tracking id, or None if
        the signal was suppressed by the dedupe/cooldown rule.
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM forex_signal_tracking "
                "WHERE pair=? AND direction=? AND ("
                "  status='open' OR created_at >= datetime('now', ?)"
                ") LIMIT 1",
                (pair, direction, f"-{self.REARM_COOLDOWN_MINUTES} minutes"),
            ).fetchone()
            if existing:
                return None
            cur = conn.execute(
                "INSERT INTO forex_signal_tracking "
                "(pair,signal,direction,entry_price,stop_price,target_price,"
                "stop_pips,target_pips,atr14,entry_ts,"
                "features_json,feature_version,model_prob,required_prob,cost_ratio,"
                "spread_pips,total_score,adx14,regime,session,model_mode,"
                "quality_passed,quality_reason,extension_score) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (pair, signal, direction, entry, stop, target,
                 stop_pips, target_pips, atr14, entry_ts,
                 json.dumps(features) if features else None, feature_version,
                 model_prob, required_prob, cost_ratio,
                 spread_pips, total_score, adx14, regime, session, model_mode,
                 None if quality_passed is None else int(quality_passed),
                 quality_reason, extension_score),
            )
            return cur.lastrowid

    # ── Alerts ──────────────────────────────────────────────────────────────

    ALERT_COOLDOWN_MINUTES = 45

    def record_alert(self, alert, cooldown_minutes: Optional[int] = None) -> Optional[int]:
        """
        Persist an alert unless the same pair+direction was already alerted inside the
        cooldown window.

        Deduping here rather than at the sink is deliberate: the scanner re-runs every
        30-60 seconds and would otherwise re-fire the same setup on every tick for as
        long as it stays valid, which is precisely the behaviour that trains someone to
        ignore the alerts. Returns the new row id, or None when suppressed.
        """
        window = self.ALERT_COOLDOWN_MINUTES if cooldown_minutes is None else cooldown_minutes
        with self._connect() as conn:
            # A zero/negative window means "no dedupe" and has to be handled here
            # rather than in SQL: datetime('now', '-0 minutes') is simply now, and
            # CURRENT_TIMESTAMP has one-second resolution, so the comparison would
            # still match a row written in the same second.
            existing = None if window <= 0 else conn.execute(
                "SELECT 1 FROM forex_alerts WHERE pair=? AND direction=? "
                "AND created_at >= datetime('now', ?) LIMIT 1",
                (alert.pair, alert.direction, f"-{int(window)} minutes"),
            ).fetchone()
            if existing:
                return None
            cur = conn.execute(
                "INSERT INTO forex_alerts "
                "(pair,direction,signal,urgency,entry,stop,target,stop_pips,target_pips,"
                " rr_ratio,spread_pips,cost_ratio,total_score,regime,session,"
                " extension_score,reason,payload_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (alert.pair, alert.direction, alert.signal, alert.urgency,
                 alert.entry, alert.stop, alert.target, alert.stop_pips,
                 alert.target_pips, alert.rr_ratio, alert.spread_pips,
                 alert.cost_ratio, alert.total_score, alert.regime, alert.session,
                 alert.extension_score, alert.reason, json.dumps(alert.to_dict())),
            )
            return cur.lastrowid

    def mark_alert_delivered(self, alert_id: int, error: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE forex_alerts SET delivered=?, delivery_error=? WHERE id=?",
                (0 if error else 1, error, alert_id),
            )

    def load_alerts(self, limit: int = 50, since_minutes: Optional[int] = None) -> list:
        """Most recent alerts first. ``since_minutes`` restricts to a recent window."""
        sql = "SELECT * FROM forex_alerts"
        params: list = []
        if since_minutes is not None:
            sql += " WHERE created_at >= datetime('now', ?)"
            params.append(f"-{int(since_minutes)} minutes")
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def evaluate_alerts(
        self, pair: str, bars: List[dict], max_hold_hours: float = 12.0,
    ) -> int:
        """
        Close open alerts for ``pair`` whose stop or target was touched by a bar that
        opened after the alert was raised, or that have outlived ``max_hold_hours``
        (closed at the last bar's close). Same stop-first convention as
        ``evaluate_tracked_signals``. Returns the number of alerts closed.
        """
        with self._connect() as conn:
            open_rows = [dict(r) for r in conn.execute(
                "SELECT id, created_at, direction, stop, target FROM forex_alerts "
                "WHERE pair=? AND COALESCE(status,'open')='open'", (pair,)
            ).fetchall()]
        if not open_rows:
            return 0

        now = datetime.now(timezone.utc)
        parsed = [(self._parse_dt(b.get("timestamp")), b) for b in bars]
        closed = 0
        for row in open_rows:
            created = self._parse_dt(row.get("created_at"))
            stop, target = row.get("stop"), row.get("target")
            if created is None or stop is None or target is None:
                continue
            forward = [b for ts, b in parsed if ts is not None and ts > created]
            long_side = (row.get("direction") or 1) == 1
            aged_out = (now - created).total_seconds() > max_hold_hours * 3600

            exit_price = exit_reason = exit_ts = None
            if len(forward) == len(parsed):
                # The bars start after the alert fired, so an earlier touch would be
                # invisible — resolving on what's left could book the wrong exit.
                if aged_out:
                    with self._connect() as conn:
                        conn.execute(
                            "UPDATE forex_alerts SET status='closed', exit_reason='EXPIRED' "
                            "WHERE id=?", (row["id"],),
                        )
                    closed += 1
                continue

            for b in forward:
                hit_stop = b["low"] <= stop if long_side else b["high"] >= stop
                hit_target = b["high"] >= target if long_side else b["low"] <= target
                if hit_stop:
                    exit_price, exit_reason = stop, "STOP"
                elif hit_target:
                    exit_price, exit_reason = target, "TARGET"
                else:
                    continue
                exit_ts = b.get("timestamp")
                break

            if exit_reason is None:
                if forward and aged_out:
                    exit_price, exit_reason = forward[-1]["close"], "TIMEOUT"
                    exit_ts = forward[-1].get("timestamp")
                else:
                    continue

            with self._connect() as conn:
                conn.execute(
                    "UPDATE forex_alerts SET status='closed', exit_price=?, "
                    "exit_reason=?, exit_ts=? WHERE id=?",
                    (exit_price, exit_reason, exit_ts, row["id"]),
                )
            closed += 1
        return closed

    def acknowledge_alert(self, alert_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE forex_alerts SET acknowledged_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), alert_id),
            )

    def evaluate_tracked_signals(
        self, pair: str, bars: List[dict], max_hold_hours: float = 12.0,
    ) -> int:
        """
        Resolve open tracked signals for ``pair`` against forward M5 bars: a stop or
        target touch closes the trade; otherwise it times out at the last close after
        ``max_hold_hours``. Resolved trades feed forex_trade_outcomes (the same table the
        Performance tab reads), so win-rate-by-signal calibrates itself over time.
        Returns the number of signals resolved.

        Accounting is net of transaction cost. Candles are mid prices, so a bracket
        that touches its mid target still costs the full spread to get in and out —
        ``r_multiple`` is therefore computed from ``net_pips``. Reporting gross R was
        overstating every result by roughly a quarter of an R at the measured spreads.
        """
        with self._connect() as conn:
            open_rows = [dict(r) for r in conn.execute(
                "SELECT * FROM forex_signal_tracking WHERE pair=? AND status='open'", (pair,)
            ).fetchall()]
        if not open_rows:
            return 0

        now = datetime.now(timezone.utc)
        pip_value = 0.01 if "JPY" in pair else 0.0001
        resolved = 0

        for row in open_rows:
            entry_ts = row.get("entry_ts") or ""
            direction = row.get("direction") or 1
            entry_price = row.get("entry_price") or 0.0
            stop = row.get("stop_price") or 0.0
            target = row.get("target_price") or 0.0
            stop_pips = row.get("stop_pips") or 0.0

            # Forward bars only (OANDA timestamps share a format → string order is valid)
            forward = [b for b in bars if b.get("timestamp", "") > entry_ts]

            exit_price: Optional[float] = None
            outcome: Optional[str] = None
            exit_ts: Optional[str] = None
            exit_reason: Optional[str] = None
            for b in forward:
                hi, lo = b["high"], b["low"]
                if direction == 1:
                    if lo <= stop:        # stop checked first = conservative
                        exit_price, outcome, exit_reason = stop, "LOSS", "STOP"
                        exit_ts = b.get("timestamp")
                        break
                    if hi >= target:
                        exit_price, outcome, exit_reason = target, "WIN", "TARGET"
                        exit_ts = b.get("timestamp")
                        break
                else:
                    if hi >= stop:
                        exit_price, outcome, exit_reason = stop, "LOSS", "STOP"
                        exit_ts = b.get("timestamp")
                        break
                    if lo <= target:
                        exit_price, outcome, exit_reason = target, "WIN", "TARGET"
                        exit_ts = b.get("timestamp")
                        break

            created = self._parse_dt(row.get("created_at"))
            if outcome is None:
                # Timeout: close at last available close once held longer than max_hold
                aged_out = created is not None and (now - created).total_seconds() > max_hold_hours * 3600
                if aged_out and forward:
                    exit_price = forward[-1]["close"]
                    exit_ts = forward[-1].get("timestamp")
                    exit_reason = "TIMEOUT"
                else:
                    continue  # still live

            gross_pips = round((exit_price - entry_price) * direction / pip_value, 1) if entry_price else 0.0
            # Full round-trip spread: entered at ask (long) / bid (short) and exited on
            # the opposite side, while the bar data that triggered the exit is mid.
            cost_pips = round(row.get("spread_pips") or 0.0, 1)
            net_pips = round(gross_pips - cost_pips, 1)
            if exit_reason == "TIMEOUT":
                outcome = "WIN" if net_pips > 0 else ("LOSS" if net_pips < 0 else "BREAKEVEN")
            r_multiple = round(net_pips / stop_pips, 2) if stop_pips and stop_pips > 0 else None

            # Real trade duration: entry bar → resolving bar. This used to be measured
            # as (now − created_at), i.e. how long until a scan happened to evaluate the
            # row, which is why wins and losses both averaged ~506 minutes.
            entry_dt = self._parse_dt(entry_ts) or created
            exit_dt = self._parse_dt(exit_ts)
            if entry_dt and exit_dt:
                hold_minutes = max(0, int((exit_dt - entry_dt).total_seconds() / 60))
            else:
                hold_minutes = None

            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO forex_trade_outcomes "
                    "(watchlist_id,tracking_id,pair,signal,entry_price,exit_price,exit_pips,"
                    "gross_pips,cost_pips,net_pips,r_multiple,outcome,hold_minutes,"
                    "exit_ts,exit_reason) VALUES (0,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["id"], pair, row.get("signal"), entry_price, exit_price, net_pips,
                     gross_pips, cost_pips, net_pips, r_multiple, outcome, hold_minutes,
                     exit_ts, exit_reason),
                )
                conn.execute(
                    "UPDATE forex_signal_tracking SET status='closed' WHERE id=?",
                    (row["id"],),
                )
            resolved += 1

        if resolved:
            self.compute_and_save_performance()
        return resolved

    _parse_dt = staticmethod(parse_ts)

    def load_tracked_signals(self, status: str = "open", limit: int = 200) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forex_signal_tracking WHERE status=? "
                "ORDER BY created_at DESC LIMIT ?", (status, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Model store / training data ─────────────────────────────────────────

    def load_training_rows(self, feature_version: Optional[int] = None) -> list:
        """
        Resolved trades joined back to the feature vector captured at arm time.

        Only rows with a stored ``features_json`` are usable — trades recorded before
        feature logging existed are unlearnable and are excluded here rather than
        silently imputed, which would teach the model from fabricated inputs.
        Ordered oldest-first so a walk-forward split is just an index cut.
        """
        sql = (
            "SELECT t.id AS tracking_id, t.features_json, t.feature_version, t.pair, "
            "       t.direction, t.signal, t.created_at, t.model_prob, "
            "       o.outcome, o.r_multiple, o.net_pips, o.exit_reason "
            "FROM forex_signal_tracking t "
            "JOIN forex_trade_outcomes o ON o.tracking_id = t.id "
            "WHERE t.features_json IS NOT NULL AND o.outcome IS NOT NULL "
        )
        params: list = []
        if feature_version is not None:
            sql += "AND t.feature_version = ? "
            params.append(feature_version)
        sql += "ORDER BY t.created_at ASC, t.id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        out = []
        for r in rows:
            row = dict(r)
            try:
                row["features"] = json.loads(row.pop("features_json") or "{}")
            except (ValueError, TypeError):
                continue
            if not row["features"]:
                continue
            out.append(row)
        return out

    def save_model(self, model_json: str, metrics: dict, activate: bool = True,
                   notes: str = "") -> int:
        """Persist a trained model and optionally make it the one the scanner serves."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO forex_models "
                "(feature_version,algo,n_train,n_test,auc,brier,top_decile_prec,"
                " base_rate,model_json,metrics_json,is_active,notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,0,?)",
                (
                    metrics.get("feature_version"), metrics.get("algo"),
                    metrics.get("n_train"), metrics.get("n_test"),
                    metrics.get("auc"), metrics.get("brier"),
                    metrics.get("top_decile_prec"), metrics.get("base_rate"),
                    model_json, json.dumps(metrics), notes,
                ),
            )
            model_id = cur.lastrowid
            if activate:
                conn.execute("UPDATE forex_models SET is_active=0")
                conn.execute("UPDATE forex_models SET is_active=1 WHERE id=?", (model_id,))
        return model_id

    def activate_model(self, model_id: int) -> bool:
        """
        Promote one stored model to active, deactivating every other.

        Also used for rollback — promoting an older model id is a valid recovery
        path when a newly promoted one turns out to behave badly live.
        Returns False if the id does not exist.
        """
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM forex_models WHERE id=?", (model_id,)
            ).fetchone()
            if not exists:
                return False
            conn.execute("UPDATE forex_models SET is_active=0")
            # A model cannot shadow and gate at once: shadow numbers are only an
            # honest preview while the model has no influence on what gets traded.
            conn.execute(
                "UPDATE forex_models SET is_active=1, is_shadow=0 WHERE id=?", (model_id,)
            )
        return True

    def deactivate_all_models(self) -> None:
        """Fall back to rules-only scanning without deleting any model."""
        with self._connect() as conn:
            conn.execute("UPDATE forex_models SET is_active=0")

    def shadow_model(self, model_id: int) -> bool:
        """
        Run one model in shadow: scored and logged on every setup, never vetoing.

        This is how a candidate earns promotion. An active model only ever sees
        outcomes for trades it let through, so its live win rate is measured on a
        censored sample and cannot say what the trades it blocked would have done.
        A shadow model is scored on every directional setup the rules propose, so
        its probabilities land on winners and losers alike and stay comparable.
        Returns False if the id does not exist.
        """
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM forex_models WHERE id=?", (model_id,)
            ).fetchone()
            if not exists:
                return False
            conn.execute("UPDATE forex_models SET is_shadow=0")
            conn.execute(
                "UPDATE forex_models SET is_shadow=1, is_active=0 WHERE id=?", (model_id,)
            )
        return True

    def clear_shadow_model(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE forex_models SET is_shadow=0")

    def load_active_model_json(self) -> Optional[str]:
        return self._load_model_json("is_active")

    def load_shadow_model_json(self) -> Optional[str]:
        return self._load_model_json("is_shadow")

    def _load_model_json(self, flag: str) -> Optional[str]:
        if flag not in ("is_active", "is_shadow"):      # guards the interpolation
            raise ValueError(f"unknown model flag: {flag}")
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT model_json FROM forex_models WHERE {flag}=1 "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row["model_json"] if row else None

    def load_models(self, limit: int = 20) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,created_at,algo,feature_version,n_train,n_test,auc,brier,"
                "top_decile_prec,base_rate,is_active,is_shadow,notes "
                "FROM forex_models ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Daily ML predictions (SQL Server mirror) ────────────────────────────

    _ML_PREDICTION_COLS = (
        "prediction_date", "pair", "sql_pair", "target_date", "predicted_signal",
        "signal_confidence", "prob_buy", "prob_sell", "prob_hold", "base_close",
        "model_name", "model_version", "source_created_at",
    )

    def save_ml_predictions(self, rows: List[dict]) -> int:
        """
        Upsert mirrored prediction rows, keyed on (prediction_date, pair).

        REPLACE rather than IGNORE so a re-run of the daily job upstream
        corrects the mirror instead of leaving a stale signal on screen.
        """
        if not rows:
            return 0
        placeholders = ",".join("?" * len(self._ML_PREDICTION_COLS))
        payload = [
            tuple(r.get(col) for col in self._ML_PREDICTION_COLS) for r in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO ml_predictions "
                f"({','.join(self._ML_PREDICTION_COLS)}, synced_at) "
                f"VALUES ({placeholders}, CURRENT_TIMESTAMP)",
                payload,
            )
        return len(payload)

    def load_ml_max_prediction_date(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(prediction_date) AS d FROM ml_predictions"
            ).fetchone()
        return row["d"] if row and row["d"] else None

    def load_ml_latest_predictions(self) -> list:
        """The newest mirrored run — what the home-page box displays."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ml_predictions WHERE prediction_date = "
                "(SELECT MAX(prediction_date) FROM ml_predictions) ORDER BY pair"
            ).fetchall()
        return [dict(r) for r in rows]

    def load_ml_predictions_for_target(self, target_date: str) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ml_predictions WHERE target_date = ? ORDER BY pair",
                (target_date,),
            ).fetchall()
        return [dict(r) for r in rows]

    def load_ml_target_dates(self, limit: int = 60) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT target_date FROM ml_predictions "
                "WHERE target_date IS NOT NULL ORDER BY target_date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["target_date"] for r in rows]

    def log_ml_sync(
        self, kind: str, status: str, rows_synced: int,
        max_prediction_date: Optional[str], message: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ml_sync_log "
                "(kind,status,rows_synced,max_prediction_date,message) VALUES (?,?,?,?,?)",
                (kind, status, rows_synced, max_prediction_date, message),
            )
            # Bounded log — this table is written on a schedule, never read in bulk.
            conn.execute(
                "DELETE FROM ml_sync_log WHERE id NOT IN "
                "(SELECT id FROM ml_sync_log ORDER BY id DESC LIMIT 200)"
            )

    def load_ml_last_sync(self, kind: Optional[str] = None) -> Optional[dict]:
        sql = "SELECT * FROM ml_sync_log"
        params: list = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY id DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def load_ml_sync_log(self, limit: int = 25) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ml_sync_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── ML reconciliation ───────────────────────────────────────────────────

    _ML_RECON_COLS = (
        "target_date", "pair", "prediction_date", "predicted_signal",
        "implied_direction", "signal_confidence", "prob_buy", "prob_sell",
        "sql_base_close", "base_close", "actual_close", "actual_return_pct",
        "actual_pips", "actual_direction", "signal_outcome", "implied_outcome",
    )

    def save_ml_reconciliation(self, rows: List[dict]) -> int:
        if not rows:
            return 0
        placeholders = ",".join("?" * len(self._ML_RECON_COLS))
        payload = [tuple(r.get(col) for col in self._ML_RECON_COLS) for r in rows]
        with self._connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO ml_reconciliation "
                f"({','.join(self._ML_RECON_COLS)}, evaluated_at) "
                f"VALUES ({placeholders}, CURRENT_TIMESTAMP)",
                payload,
            )
        return len(payload)

    def load_ml_reconciled_pairs(self, target_date: str) -> set:
        """Pairs already scored for a session — the skip-list that keeps OANDA calls at zero."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pair FROM ml_reconciliation WHERE target_date = ?", (target_date,)
            ).fetchall()
        return {r["pair"] for r in rows}

    def load_ml_reconciliation(
        self, target_date: Optional[str] = None, limit: int = 500,
    ) -> list:
        sql = "SELECT * FROM ml_reconciliation"
        params: list = []
        if target_date:
            sql += " WHERE target_date = ?"
            params.append(target_date)
        sql += " ORDER BY target_date DESC, pair ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def load_ml_reconciled_dates(self, limit: int = 60) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT target_date FROM ml_reconciliation "
                "ORDER BY target_date DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r["target_date"] for r in rows]

    def load_ml_reconciliation_by_date(self, limit: int = 60) -> list:
        """
        Per-session hit rates.

        ``signal_*`` counts only the days the gate actually took a side;
        ``lean_*`` counts every row via the prob_buy/prob_sell lean, which is the
        only way HOLD/abstain rows — the bulk of the output — get measured.
        Rows the broker has no price data for are held out of both.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT target_date, "
                "       SUM(CASE WHEN signal_outcome <> 'NO_DATA' THEN 1 ELSE 0 END) AS pairs, "
                "       SUM(CASE WHEN signal_outcome IN ('HIT','MISS') THEN 1 ELSE 0 END) AS signal_calls, "
                "       SUM(CASE WHEN signal_outcome = 'HIT' THEN 1 ELSE 0 END) AS signal_hits, "
                "       SUM(CASE WHEN implied_outcome IN ('HIT','MISS') THEN 1 ELSE 0 END) AS lean_calls, "
                "       SUM(CASE WHEN implied_outcome = 'HIT' THEN 1 ELSE 0 END) AS lean_hits, "
                "       AVG(actual_return_pct) AS avg_return_pct "
                "FROM ml_reconciliation GROUP BY target_date "
                "ORDER BY target_date DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def load_ml_reconciliation_by_pair(self, limit: int = 50) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pair, "
                "       SUM(CASE WHEN signal_outcome <> 'NO_DATA' THEN 1 ELSE 0 END) AS sessions, "
                "       SUM(CASE WHEN signal_outcome IN ('HIT','MISS') THEN 1 ELSE 0 END) AS signal_calls, "
                "       SUM(CASE WHEN signal_outcome = 'HIT' THEN 1 ELSE 0 END) AS signal_hits, "
                "       SUM(CASE WHEN implied_outcome IN ('HIT','MISS') THEN 1 ELSE 0 END) AS lean_calls, "
                "       SUM(CASE WHEN implied_outcome = 'HIT' THEN 1 ELSE 0 END) AS lean_hits "
                "FROM ml_reconciliation GROUP BY pair ORDER BY pair LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
