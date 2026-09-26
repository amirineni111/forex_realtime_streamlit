"""
Headless real-time scanner: scan on every signal-bar close and push new alerts.

The dashboard only scans while a browser tab is open and its auto-refresh timer
fires, and that timer is not aligned to bar closes. This runner wakes a short lag
after each bar boundary of the signal timeframe (M15 by default, the moment a new
completed bar exists), runs the same ``run_scan``, and pushes whatever cleared the
entry-quality gate. OANDA publishes candles immediately, so the default lag is
20 seconds rather than the minute-plus a delayed feed needs.

Always run from the project root:

    python scripts/run_alerts.py                 # pairs + timeframe from dashboard prefs
    python scripts/run_alerts.py --once          # one scan now, then exit (a smoke test)
    python scripts/run_alerts.py --test-push     # send a test message to the push URL and exit
    python scripts/run_alerts.py --sample-alert  # send a sample alert in the real format and exit

Safe to run alongside the dashboard: both write to the same database, and the
alert dedupe (one alert per pair+direction per 45 minutes) means a setup is only
ever alerted once.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forex.alerts import Alert, URGENCY_HIGH, dispatch, make_sink, send_test  # noqa: E402
from forex.config import get_settings  # noqa: E402
from forex.market_sessions import US_EASTERN, is_forex_market_open  # noqa: E402
from forex.models import ScanRequest  # noqa: E402
from forex.pairs import UNIVERSE_MAP  # noqa: E402
from forex.scanner import run_scan  # noqa: E402
from forex.storage import Storage  # noqa: E402

PREFS_PATH = Path("data/app_preferences.json")
BAR_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60}


def _prefs() -> dict:
    try:
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def pairs_from_prefs(prefs: dict) -> list:
    """The pair list the dashboard sidebar would scan, parsed the same way."""
    choice = prefs.get("universe_choice", "Tight spread (recommended)")
    if choice == "Custom":
        raw = prefs.get("custom_pairs_raw", "")
        return [p.strip().upper() for p in raw.replace(",", "\n").splitlines() if p.strip()]
    return list(UNIVERSE_MAP.get(choice, UNIVERSE_MAP["Tight spread (recommended)"]))


def next_wake(now: datetime, bar_minutes: int, lag_seconds: float) -> datetime:
    """The next ``bar_minutes`` boundary plus ``lag_seconds`` strictly after ``now``."""
    base = now.replace(second=0, microsecond=0)
    minutes_into_day = base.hour * 60 + base.minute
    base -= timedelta(minutes=minutes_into_day % bar_minutes)
    wake = base + timedelta(seconds=lag_seconds)
    while wake <= now:
        wake += timedelta(minutes=bar_minutes)
    return wake


def _stamp() -> str:
    return datetime.now(US_EASTERN).strftime("%H:%M:%S")


def _scan_once(settings, storage, request, sinks, quiet: bool) -> None:
    summary = run_scan(settings, storage, request, alert_sinks=sinks)
    if not quiet or summary.alerts_raised:
        print(f"[{_stamp()}] scanned {summary.pairs_scanned}, "
              f"{summary.signals_found} signals, {summary.alerts_raised} new alerts, "
              f"{summary.errors} errors", flush=True)
    if summary.alerts_raised:
        for row in storage.load_alerts(limit=summary.alerts_raised, since_minutes=5):
            side = "LONG" if row["direction"] > 0 else "SHORT"
            status = "pushed" if row.get("delivered") else (
                f"push failed - {row['delivery_error']}" if row.get("delivery_error")
                else "console only")
            print(f"  {row['urgency']} {side} {row['pair']} @ {row['entry']:g} "
                  f"SL {row['stop']:g} TP {row['target']:g} ({status})", flush=True)


def _sample_alert() -> Alert:
    return Alert(
        pair="EUR_USD", signal="STRONG_BUY", direction=1,
        entry=1.08450, stop=1.08180, target=1.08855,
        stop_pips=27.0, target_pips=40.5, rr_ratio=1.5,
        spread_pips=1.2, cost_ratio=0.044, total_score=68.0,
        regime="TRENDING", session="London_NY_Overlap",
        extension_score=0.0, urgency=URGENCY_HIGH,
        reason="SAMPLE ALERT - not a real signal. Real alerts look exactly like this.",
    )


def main() -> int:
    prefs = _prefs()
    ap = argparse.ArgumentParser(description="Scan on each bar close and push new alerts")
    ap.add_argument("--pairs", nargs="*", help="defaults to the dashboard's pair universe")
    ap.add_argument("--timeframe", default=prefs.get("signal_timeframe", "M15"),
                    choices=sorted(BAR_MINUTES), help="signal timeframe (default: dashboard's)")
    ap.add_argument("--max-spread", type=float, default=2.0, help="max spread in pips")
    ap.add_argument("--lag", type=float, default=20.0,
                    help="seconds after each bar boundary to scan (default 20)")
    ap.add_argument("--weekend", action="store_true",
                    help="also scan while the forex market is closed")
    ap.add_argument("--once", action="store_true", help="run one scan now and exit")
    ap.add_argument("--test-push", action="store_true", help="send a test message and exit")
    ap.add_argument("--sample-alert", action="store_true",
                    help="send a made-up alert through the real alert path and exit")
    ap.add_argument("--quiet", action="store_true", help="only print scans that raised alerts")
    args = ap.parse_args()

    settings = get_settings()
    url = settings.alert_webhook_url or (prefs.get("alert_webhook") or "").strip()
    sink = make_sink(url)

    if args.test_push or args.sample_alert:
        if sink is None:
            print("FOREX_ALERT_WEBHOOK_URL is not set (see .env.example).")
            return 1
        if args.test_push:
            err = send_test(url, "Forex scanner test alert",
                            "If you can read this, forex push alerts are working.")
            print("sent" if err is None else f"failed: {err}")
            return 0 if err is None else 1
        report = dispatch([_sample_alert()], [sink])
        print("sent" if not report["errors"] else f"failed: {report['errors']}")
        return 0 if not report["errors"] else 1

    if not settings.oanda_api_key:
        print("OANDA_API_KEY is not set in .env.")
        return 1
    pairs = [p.upper() for p in (args.pairs or pairs_from_prefs(prefs))]
    if not pairs:
        print("No pairs: pass --pairs or pick a universe in the dashboard.")
        return 1

    storage = Storage(settings.db_path)
    request = ScanRequest(pairs=pairs, max_spread_pips=args.max_spread,
                          signal_timeframe=args.timeframe)
    sinks = [sink] if sink else []
    push = f"push -> {sink.name}" if sink else "console only (no FOREX_ALERT_WEBHOOK_URL)"
    print(f"Watching {len(pairs)} pairs on {args.timeframe}; {push}. Ctrl+C to stop.",
          flush=True)

    if args.once:
        _scan_once(settings, storage, request, sinks, quiet=False)
        return 0

    bar_minutes = BAR_MINUTES[args.timeframe]
    try:
        while True:
            now = datetime.now(timezone.utc)
            wake = next_wake(now, bar_minutes, args.lag)
            time.sleep(max(0.0, (wake - now).total_seconds()))
            if not args.weekend and not is_forex_market_open():
                continue
            try:
                _scan_once(settings, storage, request, sinks, args.quiet)
            except Exception as exc:  # keep the loop alive through network blips
                print(f"[{_stamp()}] scan failed: {exc}", flush=True)
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
