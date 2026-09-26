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
    # Push endpoint for alerts: an ntfy topic URL (phone push), a Slack/Discord
    # webhook, or any URL taking a JSON POST. The dashboard sidebar overrides it;
    # the headless runner (scripts/run_alerts.py) uses it directly.
    alert_webhook_url: str = ""

    # ── Daily ML prediction source (SQL Server, written by sqlserver_copilot_forex) ──
    # Read-only, and read rarely: predictions land once per weekday after the
    # 20:55 ET run, so the scanner's 1-minute refresh serves them from the local
    # SQLite mirror and only reaches SQL Server when that mirror is behind.
    sql_server: Optional[str] = None
    sql_database: Optional[str] = None
    sql_username: Optional[str] = None
    sql_password: Optional[str] = None
    sql_driver: str = "ODBC Driver 17 for SQL Server"
    sql_trusted_connection: bool = False
    sql_login_timeout_seconds: int = 10
    ml_predictions_table: str = "forex_ml_predictions"
    ml_sync_min_interval_minutes: int = 30
    ml_initial_sync_days: int = 60

    @property
    def base_url(self) -> str:
        if self.oanda_env == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"

    @property
    def ml_source_configured(self) -> bool:
        """True when there is enough config to attempt a SQL Server connection."""
        if not self.sql_server or not self.sql_database:
            return False
        return self.sql_trusted_connection or bool(self.sql_username and self.sql_password)


def get_settings() -> AppSettings:
    load_dotenv()
    raw_db = os.getenv("FOREX_DB_PATH", "data/forex_data.sqlite3")
    return AppSettings(
        oanda_api_key=os.getenv("OANDA_API_KEY"),
        oanda_account_id=os.getenv("OANDA_ACCOUNT_ID") or None,
        oanda_env=os.getenv("OANDA_ENV", "practice"),
        db_path=Path(os.path.expandvars(raw_db)),
        alert_webhook_url=(os.getenv("FOREX_ALERT_WEBHOOK_URL") or "").strip(),
        sql_server=os.getenv("SQL_SERVER") or None,
        sql_database=os.getenv("SQL_DATABASE") or None,
        sql_username=os.getenv("SQL_USERNAME") or None,
        sql_password=os.getenv("SQL_PASSWORD") or None,
        sql_driver=os.getenv("SQL_DRIVER") or "ODBC Driver 17 for SQL Server",
        sql_trusted_connection=(os.getenv("SQL_TRUSTED_CONNECTION", "") or "").strip().lower()
        in ("yes", "true", "1"),
        ml_predictions_table=os.getenv("ML_PREDICTIONS_TABLE") or "forex_ml_predictions",
        ml_sync_min_interval_minutes=int(os.getenv("ML_SYNC_MIN_INTERVAL_MINUTES", "30")),
        ml_initial_sync_days=int(os.getenv("ML_INITIAL_SYNC_DAYS", "60")),
    )
