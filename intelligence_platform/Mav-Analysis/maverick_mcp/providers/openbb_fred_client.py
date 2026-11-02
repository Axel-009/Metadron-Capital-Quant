"""
OpenBB Platform FRED client for Maverick-MCP.

Routes all FRED (Federal Reserve Economic Data) requests through the OpenBB
Platform API instead of using the fredapi package directly. This centralizes
economic data access through the OpenBB gateway running on port 6900.

The client provides a drop-in replacement for fredapi.Fred.get_series(),
returning pandas Series with datetime index — identical to the fredapi interface.
"""

import logging
import os
from datetime import date

import pandas as pd
import requests

logger = logging.getLogger("maverick_mcp.openbb_fred_client")

# OpenBB Platform API configuration
OPENBB_API_HOST = os.getenv("OPENBB_API_HOST", "127.0.0.1")
OPENBB_API_PORT = os.getenv("OPENBB_API_PORT", "6900")
OPENBB_BASE_URL = os.getenv(
    "OPENBB_BASE_URL", f"http://{OPENBB_API_HOST}:{OPENBB_API_PORT}"
)

# Optional OpenBB auth (disabled by default for local use)
OPENBB_API_USERNAME = os.getenv("OPENBB_API_USERNAME", "")
OPENBB_API_PASSWORD = os.getenv("OPENBB_API_PASSWORD", "")


class OpenBBFredClient:
    """
    FRED data client that routes requests through the OpenBB Platform API.

    Drop-in replacement for fredapi.Fred — provides get_series() with the same
    return type (pd.Series with datetime index) but fetches data via OpenBB's
    /api/v1/economy/fred_series endpoint.

    This enables:
    - Centralized data routing through OpenBB's provider infrastructure
    - Access to all 34+ OpenBB data providers from a single gateway
    - Unified API key management via OpenBB's credential system
    - Consistent rate limiting and caching at the platform level
    """

    def __init__(
        self,
        base_url: str = OPENBB_BASE_URL,
        api_key: str | None = None,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        # Set up auth if configured
        if OPENBB_API_USERNAME and OPENBB_API_PASSWORD:
            self.session.auth = (OPENBB_API_USERNAME, OPENBB_API_PASSWORD)

        # Pass FRED API key as a header if provided (OpenBB can use it)
        if api_key:
            self.session.headers["X-FRED-API-KEY"] = api_key

        logger.info(
            f"OpenBB FRED client initialized: {self.base_url}"
        )

    def get_series(
        self,
        series_id: str,
        observation_start: str | date | None = None,
        observation_end: str | date | None = None,
        **kwargs,
    ) -> pd.Series:
        """
        Fetch a FRED series via OpenBB Platform API.

        Compatible with fredapi.Fred.get_series() — returns a pandas Series
        with datetime index and float values.

        Args:
            series_id: FRED series identifier (e.g., "SP500", "UNRATE", "CPILFESL")
            observation_start: Start date (str YYYY-MM-DD or date object)
            observation_end: End date (str YYYY-MM-DD or date object)
            **kwargs: Additional parameters passed to OpenBB

        Returns:
            pd.Series with datetime index containing the series observations
        """
        params: dict = {
            "symbol": series_id,
            "provider": "fred",
            "limit": 100000,
        }

        if observation_start:
            params["start_date"] = (
                observation_start.isoformat()
                if isinstance(observation_start, date)
                else str(observation_start)
            )
        if observation_end:
            params["end_date"] = (
                observation_end.isoformat()
                if isinstance(observation_end, date)
                else str(observation_end)
            )

        # Map any extra kwargs OpenBB supports
        if "frequency" in kwargs:
            params["frequency"] = kwargs["frequency"]
        if "aggregation_method" in kwargs:
            params["aggregation_method"] = kwargs["aggregation_method"]

        url = f"{self.base_url}/api/v1/economy/fred_series"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return self._parse_response(data, series_id)

        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to OpenBB Platform API at {self.base_url}. "
                "Ensure the OpenBB API server is running (port 6900)."
            )
            return pd.Series([], dtype=float)
        except requests.exceptions.Timeout:
            logger.error(
                f"Timeout fetching {series_id} from OpenBB Platform API"
            )
            return pd.Series([], dtype=float)
        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP error fetching {series_id} from OpenBB: {e.response.status_code} - {e.response.text}"
            )
            return pd.Series([], dtype=float)
        except Exception as e:
            logger.error(f"Error fetching {series_id} from OpenBB: {e}")
            return pd.Series([], dtype=float)

    def _parse_response(self, data: dict, series_id: str) -> pd.Series:
        """
        Parse OpenBB API JSON response into a pandas Series.

        OpenBB returns:
        {
            "results": [
                {"date": "2024-01-01", "value": 123.45},
                ...
            ],
            "provider": "fred",
            "warnings": [...],
            "chart": null,
            "extra": {...}
        }

        Args:
            data: Raw JSON response from OpenBB
            series_id: Series ID for logging

        Returns:
            pd.Series with datetime index
        """
        results = data.get("results", [])

        if not results:
            logger.warning(f"No data returned from OpenBB for FRED series {series_id}")
            return pd.Series([], dtype=float)

        dates = []
        values = []

        for record in results:
            record_date = record.get("date")
            # OpenBB fred_series returns the value in a field named after the series
            # or in a generic "value" field — handle both
            value = record.get("value")
            if value is None:
                # Try the series ID as lowercase key
                value = record.get(series_id.lower())
            if value is None:
                # Try common OpenBB field names
                for field in ("close", "rate", "level"):
                    value = record.get(field)
                    if value is not None:
                        break

            if record_date is not None and value is not None:
                try:
                    dates.append(pd.Timestamp(record_date))
                    values.append(float(value))
                except (ValueError, TypeError):
                    continue

        if not dates:
            logger.warning(
                f"Could not parse any observations for FRED series {series_id}"
            )
            return pd.Series([], dtype=float)

        series = pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)
        series = series.sort_index()

        logger.debug(
            f"Fetched {len(series)} observations for {series_id} via OpenBB"
        )
        return series

    def search(self, query: str, **kwargs) -> list[dict]:
        """
        Search FRED series via OpenBB Platform API.

        Args:
            query: Search term
            **kwargs: Additional search parameters

        Returns:
            List of matching series metadata dicts
        """
        params: dict = {
            "query": query,
            "provider": "fred",
        }
        if "limit" in kwargs:
            params["limit"] = kwargs["limit"]

        url = f"{self.base_url}/api/v1/economy/fred_search"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error searching FRED via OpenBB: {e}")
            return []

    def health_check(self) -> bool:
        """
        Check if the OpenBB Platform API is reachable.

        Returns:
            True if the API is responding, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/economy/fred_series",
                params={"symbol": "DGS10", "provider": "fred", "limit": 1},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False
