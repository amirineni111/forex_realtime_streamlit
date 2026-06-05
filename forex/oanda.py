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
        bars: List[ForexBar] = []
        for candle in data.get("candles", []):
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
