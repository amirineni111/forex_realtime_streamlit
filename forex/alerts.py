"""
Near-real-time alerting for setups that clear the entry-quality gate.

Design stance
-------------
An alert is *not* "the scanner found a signal". The scanner finds 20-30 signals a
day and their measured expectancy is -0.082R, so alerting on those would just be a
faster way to lose. An alert here means: the rules proposed a direction, the entry
is not extended (``forex.quality``), the spread is small enough relative to the risk
that the trade can clear its own cost, and we have not already alerted on this
pair+direction recently.

On the measured history that combination fires on roughly a fifth of what the
scanner currently arms -- about 1-2 alerts per pair per week rather than per hour.
That rarity is the point: the alert is worth interrupting someone for precisely
because the filter throws most things away.

Delivery
--------
Sinks are pluggable and every one of them is best-effort: a webhook that is down
must never take the scan with it, so ``dispatch`` catches per-sink failures and
reports them rather than raising. ``WebhookSink`` posts a JSON body that Slack and
Discord both accept (both read ``text``/``content``), and carries the full
structured payload alongside so a custom consumer has everything.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence

import httpx

# Alerts for the same pair+direction inside this window are suppressed. Matches
# Storage.REARM_COOLDOWN_MINUTES so the alert feed and the tracked-signal ledger
# agree about what counts as "the same setup".
DEFAULT_COOLDOWN_MINUTES = 45

# Urgency bands. ``extension_score`` is 0 for a calm entry and rises as the setup
# starts chasing; the gate has already rejected anything above its thresholds, so
# these split what survives into "textbook" and "acceptable".
URGENCY_HIGH = "HIGH"
URGENCY_NORMAL = "NORMAL"


@dataclass
class Alert:
    """One actionable setup, with everything needed to act without opening the app."""

    pair: str
    signal: str
    direction: int                     # +1 long, -1 short
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    stop_pips: Optional[float]
    target_pips: Optional[float]
    rr_ratio: Optional[float]
    spread_pips: Optional[float]
    cost_ratio: Optional[float]
    total_score: Optional[float]
    regime: Optional[str]
    session: Optional[str]
    extension_score: float
    urgency: str
    reason: str
    warnings: List[str] = field(default_factory=list)
    created_at: str = ""
    as_of: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def dedupe_key(self) -> str:
        return f"{self.pair}:{self.direction}"

    @property
    def side(self) -> str:
        return "LONG" if self.direction > 0 else "SHORT"

    def to_dict(self) -> dict:
        return asdict(self)

    def headline(self) -> str:
        """One line suitable for a push notification or a chat message."""
        bits = [f"{self.urgency} {self.side} {self.pair}"]
        if self.entry is not None:
            bits.append(f"@ {self.entry:g}")
        if self.stop is not None and self.target is not None:
            bits.append(f"SL {self.stop:g} / TP {self.target:g}")
        if self.stop_pips and self.target_pips:
            bits.append(f"({self.stop_pips:.0f}p risk / {self.target_pips:.0f}p reward)")
        return " ".join(bits)

    def body(self) -> str:
        """Multi-line detail: why this one, and what it costs to take."""
        lines = [self.headline()]
        if self.cost_ratio is not None:
            lines.append(
                f"cost {self.cost_ratio:.1%} of risk "
                f"(spread {self.spread_pips:.1f}p) -> needs "
                f"{(1 + self.cost_ratio) / (1 + (self.rr_ratio or 1.5)):.1%} win rate"
            )
        ctx = [c for c in (self.regime, self.session) if c]
        if ctx:
            lines.append(" | ".join(ctx) + f" | score {self.total_score:.0f}"
                         if self.total_score is not None else " | ".join(ctx))
        lines.append(f"entry quality: extension {self.extension_score:.2f} -- {self.reason}")
        for w in self.warnings:
            lines.append(f"note: {w}")
        return "\n".join(lines)


def classify_urgency(extension_score: float, total_score: Optional[float]) -> str:
    """
    HIGH when the entry is genuinely calm *and* the rules liked it independently.

    Both halves are required: a calm entry on a weak setup is just a quiet market,
    and a strong score on a stretched entry is the failure mode the gate exists to
    catch.
    """
    if extension_score <= 0.0 and (total_score or 0.0) >= 60:
        return URGENCY_HIGH
    return URGENCY_NORMAL


def build_alert(snapshot, verdict, features: Optional[dict] = None) -> Optional[Alert]:
    """
    Turn a scored snapshot plus its quality verdict into an Alert.

    Returns None when the setup is not alertable -- not actionable, rejected by the
    gate, or missing the levels needed to act on it. Returning None rather than a
    suppressed Alert keeps "was there an alert" a single unambiguous check.
    """
    if verdict is None or not verdict.passed:
        return None

    signal = getattr(snapshot, "trade_signal", None)
    if signal in (None, "AVOID", "WATCH_ONLY"):
        return None

    entry = getattr(snapshot, "suggested_entry", None)
    stop = getattr(snapshot, "suggested_stop", None)
    target = getattr(snapshot, "suggested_target", None)
    if entry is None or stop is None or target is None:
        return None

    direction = -1 if "SHORT" in signal else 1
    total_score = getattr(snapshot, "total_score", None)

    return Alert(
        pair=getattr(snapshot, "pair", "?"),
        signal=signal,
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        stop_pips=getattr(snapshot, "stop_pips", None),
        target_pips=getattr(snapshot, "target_pips", None),
        rr_ratio=getattr(snapshot, "rr_ratio", None),
        spread_pips=getattr(snapshot, "spread_pips", None),
        cost_ratio=getattr(snapshot, "cost_ratio", None),
        total_score=total_score,
        regime=getattr(snapshot, "regime", None),
        session=getattr(snapshot, "current_session", None),
        extension_score=verdict.extension_score,
        urgency=classify_urgency(verdict.extension_score, total_score),
        reason=verdict.reason,
        warnings=list(verdict.warnings),
        as_of=getattr(snapshot, "as_of", None),
    )


# -- delivery ---------------------------------------------------------------

class AlertSink:
    """A destination for alerts. Implementations must not raise from ``send``."""

    name = "sink"

    def send(self, alert: Alert) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class CallableSink(AlertSink):
    """Adapter so a plain function (``st.toast``, ``print``, a test spy) is a sink."""

    def __init__(self, fn: Callable[[Alert], None], name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    def send(self, alert: Alert) -> None:
        self._fn(alert)


class WebhookSink(AlertSink):
    """
    POST the alert to a webhook (Slack, Discord, or anything that takes JSON).

    ``text`` and ``content`` carry the human-readable form because Slack reads the
    first and Discord the second; ``alert`` carries the full structured payload for
    consumers that want to do something more than display it.
    """

    name = "webhook"

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert) -> None:
        body = alert.body()
        payload = {"text": body, "content": body, "alert": alert.to_dict()}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.url, json=payload)
            resp.raise_for_status()


def dispatch(alerts: Sequence[Alert], sinks: Sequence[AlertSink]) -> dict:
    """
    Send every alert to every sink, best-effort.

    A failing sink is recorded and skipped rather than propagated: alerting is a
    side-channel, and a dead webhook must not abort a scan that is also writing
    tracked signals and outcomes. Returns a report so the caller can surface
    failures in the UI instead of swallowing them silently.
    """
    sent = 0
    errors: List[str] = []
    for alert in alerts:
        for sink in sinks:
            try:
                sink.send(alert)
                sent += 1
            except Exception as exc:
                errors.append(f"{sink.name}/{alert.pair}: {exc}")
    return {"alerts": len(alerts), "sent": sent, "errors": errors}


def format_digest(alerts: Sequence[Alert]) -> str:
    """Compact multi-alert summary, for a single notification covering one scan."""
    if not alerts:
        return "No qualifying setups."
    head = f"{len(alerts)} qualifying setup{'s' if len(alerts) != 1 else ''}"
    return head + "\n" + "\n".join(f"  - {a.headline()}" for a in alerts)
