from __future__ import annotations
from typing import List, Optional
from datetime import datetime, timezone

import httpx

from .config import AppSettings
from .models import ForexBar, ForexQuote
from .pairs import spread_to_pips


class OandaClient:
    def __init__(self, settings: AppSettings) -> None:
        self._base_url = settings.base_url
        self._token = settings.oanda_api_key or ""
        self._account_id = settings.oanda_account_id or ""
        self._timeout = settings.request_timeout_seconds

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept-Datetime-Format": "RFC3339",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self._base_url}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            return resp.json()

    def resolve_account_id(self) -> str:
        """Fetch first account ID if not already set; validates the API key."""
        if self._account_id:
            return self._account_id
        data = self._get("/v3/accounts")
        accounts = data.get("accounts", [])
        if not accounts:
            raise ValueError("No OANDA accounts found for this API key.")
        self._account_id = accounts[0]["id"]
        return self._account_id

    def get_pricing(self, pairs: List[str]) -> List[ForexQuote]:
        """Fetch current bid/ask for a list of instruments in a single call."""
        account_id = self.resolve_account_id()
        instruments = ",".join(pairs)
        data = self._get(
            f"/v3/accounts/{account_id}/pricing",
            params={"instruments": instruments},
        )
        quotes: List[ForexQuote] = []
        now_str = datetime.now(timezone.utc).isoformat()
        for price in data.get("prices", []):
            pair = price.get("instrument", "")
            bids = price.get("bids", [])
            asks = price.get("asks", [])
            if not bids or not asks:
                continue
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            spread = ask - bid
            quotes.append(
                ForexQuote(
                    pair=pair,
                    bid=bid,
                    ask=ask,
                    spread_pips=spread_to_pips(pair, spread),
                    as_of=price.get("time", now_str),
                )
            )
        return quotes

    def get_candles(
        self,
        pair: str,
        granularity: str = "M5",
        count: int = 200,
    ) -> List[ForexBar]:
        """Fetch OHLCV candles for an instrument."""
        data = self._get(
            f"/v3/instruments/{pair}/candles",
            params={"granularity": granularity, "count": count, "price": "M"},
        )
        return self._parse_candles(data, pair, granularity)

    def get_candles_range(
        self,
        pair: str,
        start: str,
        end: str,
        granularity: str = "M5",
    ) -> List[ForexBar]:
        """
        Fetch every completed candle in ``[start, end]`` (RFC3339 strings).

        ``get_candles`` can only reach backwards from now by ``count``, which is fine
        for live scanning but useless for replaying a trade that closed weeks ago.
        OANDA caps a single response at 5000 candles, so this pages forward on the
        last returned timestamp until the window is covered. The API rejects
        ``from``+``to``+``count`` together, so paging uses ``from``+``count`` and the
        tail is trimmed against ``end`` here.
        """
        bars: List[ForexBar] = []
        cursor = start
        seen: set = set()
        while cursor < end:
            data = self._get(
                f"/v3/instruments/{pair}/candles",
                params={
                    "granularity": granularity, "price": "M",
                    "from": cursor, "count": 5000,
                },
            )
            page = self._parse_candles(data, pair, granularity)
            fresh = [b for b in page if b.timestamp not in seen]
            if not fresh:
                break  # no forward progress (weekend gap or end of history)
            seen.update(b.timestamp for b in fresh)
            bars.extend(b for b in fresh if b.timestamp <= end)
            cursor = fresh[-1].timestamp
        return bars

    @staticmethod
    def _parse_candles(data: dict, pair: str, granularity: str) -> List[ForexBar]:
        bars: List[ForexBar] = []
        for candle in data.get("candles", []):
            # Skip the still-forming bar — including it makes every indicator
            # (RSI/MACD/EMA/breakout) repaint until the candle closes.
            if not candle.get("complete", False):
                continue
            mid = candle.get("mid")
            if mid is None:
                continue
            bars.append(
                ForexBar(
                    pair=pair,
                    timeframe=granularity,
                    timestamp=candle["time"],
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=int(candle.get("volume", 0)),
                )
            )
        return bars
