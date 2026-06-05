from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
import os


class AppSettings(BaseModel):
    oanda_api_key: Optional[str] = None
    oanda_account_id: Optional[str] = None
    oanda_env: str = "practice"
    db_path: Path = Path("data/forex_data.sqlite3")
    request_timeout_seconds: float = 20.0

    @property
    def base_url(self) -> str:
        if self.oanda_env == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"


def get_settings() -> AppSettings:
    load_dotenv()
    raw_db = os.getenv("FOREX_DB_PATH", "data/forex_data.sqlite3")
    return AppSettings(
        oanda_api_key=os.getenv("OANDA_API_KEY"),
        oanda_account_id=os.getenv("OANDA_ACCOUNT_ID") or None,
        oanda_env=os.getenv("OANDA_ENV", "practice"),
        db_path=Path(os.path.expandvars(raw_db)),
    )
