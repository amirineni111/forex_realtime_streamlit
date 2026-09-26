"""ntfy/webhook sink selection, the ntfy request shape, and the bar-close wake schedule."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from forex.alerts import (
    Alert, NtfySink, URGENCY_HIGH, URGENCY_NORMAL, WebhookSink, make_sink, send_test,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_alerts import next_wake, pairs_from_prefs  # noqa: E402


def _alert(**over) -> Alert:
    base = dict(
        pair="EUR_USD", signal="STRONG_BUY", direction=1, entry=1.0845, stop=1.0818,
        target=1.08855, stop_pips=27.0, target_pips=40.5, rr_ratio=1.5, spread_pips=1.2,
        cost_ratio=0.044, total_score=68.0, regime="TRENDING", session="London",
        extension_score=0.0, urgency=URGENCY_HIGH, reason="calm entry",
    )
    base.update(over)
    return Alert(**base)


def test_make_sink_routes_ntfy_to_plain_text_and_the_rest_to_json():
    assert isinstance(make_sink("https://ntfy.sh/topic"), NtfySink)
    assert isinstance(make_sink("https://ntfy.example.org/topic"), NtfySink)
    assert isinstance(make_sink("https://hooks.slack.com/services/x"), WebhookSink)
    assert make_sink("") is None and make_sink("   ") is None


def test_ntfy_request_is_plain_text_with_headers():
    req = NtfySink("https://ntfy.sh/topic").request(_alert())
    assert req.headers["Title"] == "LONG EUR/USD @ 1.0845 [HIGH]"
    assert req.headers["Priority"] == "high"
    assert req.headers["Tags"] == "chart_with_upwards_trend"
    body = req.content.decode()
    assert "SL 1.0818 / TP 1.08855" in body and not body.lstrip().startswith("{")


def test_ntfy_normal_short_gets_default_priority_and_down_arrow():
    req = NtfySink("https://ntfy.sh/topic").request(
        _alert(direction=-1, signal="STRONG_SHORT", urgency=URGENCY_NORMAL))
    assert req.headers["Title"] == "SHORT EUR/USD @ 1.0845"
    assert req.headers["Priority"] == "default"
    assert req.headers["Tags"] == "chart_with_downwards_trend"


def test_send_test_reports_instead_of_raising():
    assert send_test("", "T", "B") == "no URL configured"
    assert send_test("http://127.0.0.1:9/ntfy-nothing-listens", "T", "B", timeout=2) is not None


def test_next_wake_lands_just_after_the_next_m15_close():
    now = datetime(2026, 9, 24, 18, 37, 0, tzinfo=timezone.utc)
    assert next_wake(now, 15, 20) == datetime(2026, 9, 24, 18, 45, 20, tzinfo=timezone.utc)
    # Inside the lag window of the bar that just closed: wake for that bar.
    early = datetime(2026, 9, 24, 18, 45, 5, tzinfo=timezone.utc)
    assert next_wake(early, 15, 20) == datetime(2026, 9, 24, 18, 45, 20, tzinfo=timezone.utc)
    hourly = datetime(2026, 9, 24, 18, 0, 30, tzinfo=timezone.utc)
    assert next_wake(hourly, 60, 20) == datetime(2026, 9, 24, 19, 0, 20, tzinfo=timezone.utc)


def test_pairs_from_prefs_matches_the_dashboard_parsing():
    assert pairs_from_prefs({"universe_choice": "Custom",
                             "custom_pairs_raw": "eur_usd,\nGBP_USD\n\n"}) == ["EUR_USD", "GBP_USD"]
    assert "EUR_USD" in pairs_from_prefs({"universe_choice": "Majors"})
    assert pairs_from_prefs({}) == pairs_from_prefs({"universe_choice": "Tight spread (recommended)"})
