from app.stocks.libraries.base_library import FinanceLibrary
from datetime import date, timedelta

try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False


class OpenBBProvider(FinanceLibrary):
    """
    Client for searching stock information using the OpenBB SDK, implementing the FinanceLibrary interface.
    Used as the primary data source with graceful degradation if OpenBB is not installed.
    """

    @staticmethod
    def is_available() -> bool:
        """Returns True if the OpenBB SDK is installed and importable."""
        return OPENBB_AVAILABLE

    @staticmethod
    def get_ticker(cusip: str, **kwargs) -> str | None:
        """
        Searches for a ticker for a given CUSIP using OpenBB's equity search.

        Args:
            cusip (str): The CUSIP of the stock.

        Returns:
            str | None: The ticker symbol if found, otherwise None.
        """
        if not OPENBB_AVAILABLE:
            return None

        try:
            result = obb.equity.search(query=cusip)
            if result and hasattr(result, 'results') and result.results:
                return result.results[0].symbol
            print(f"🚨 OpenBB: No ticker found for CUSIP {cusip}.")
            return None
        except Exception as e:
            print(f"❌ ERROR: Failed to get ticker for CUSIP {cusip} using OpenBB: {e}")
            return None

    @staticmethod
    def get_company(cusip: str, **kwargs) -> str | None:
        """
        Searches for a company name for a given CUSIP using OpenBB's equity search.

        Args:
            cusip (str): The CUSIP of the stock.
            ticker (str): The stock ticker (optional kwarg).

        Returns:
            str | None: The company name if found, otherwise None.
        """
        if not OPENBB_AVAILABLE:
            return None

        ticker = kwargs.get('ticker')
        query = ticker if ticker else cusip

        try:
            result = obb.equity.search(query=query)
            if result and hasattr(result, 'results') and result.results:
                name = getattr(result.results[0], 'name', None)
                if name:
                    return name
            print(f"🚨 OpenBB: No company found for CUSIP {cusip}.")
            return None
        except Exception as e:
            print(f"❌ ERROR: Failed to get company for CUSIP {cusip} using OpenBB: {e}")
            return None

    @staticmethod
    def get_current_price(ticker: str, **kwargs) -> float | None:
        """
        Gets the current (latest) market price for a ticker using OpenBB's equity quote.

        Args:
            ticker (str): The stock ticker.

        Returns:
            float | None: The current price if found, otherwise None.
        """
        if not OPENBB_AVAILABLE:
            return None

        try:
            result = obb.equity.price.quote(symbol=ticker)
            if result and hasattr(result, 'results') and result.results:
                quote = result.results[0]
                # Try last_price first, then previous_close as fallback
                price = getattr(quote, 'last_price', None) or getattr(quote, 'prev_close', None)
                if price is not None:
                    return float(price)

            print(f"🚨 OpenBB: No current price found for {ticker}.")
            return None
        except Exception as e:
            print(f"❌ ERROR: Failed to get current price for {ticker} using OpenBB: {e}")
            return None

    @staticmethod
    def get_avg_price(ticker: str, date_obj: date, **kwargs) -> float | None:
        """
        Gets the average daily price for a ticker on a specific date using OpenBB's historical data.
        The average price is calculated as (High + Low) / 2.

        Args:
            ticker (str): The stock ticker.
            date_obj (date): The date for which to fetch the price.

        Returns:
            float | None: The average price if found, otherwise None.
        """
        if not OPENBB_AVAILABLE:
            return None

        try:
            start_date = date_obj.isoformat()
            end_date = (date_obj + timedelta(days=1)).isoformat()

            result = obb.equity.price.historical(
                symbol=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if result and hasattr(result, 'results') and result.results:
                bar = result.results[0]
                high = getattr(bar, 'high', None)
                low = getattr(bar, 'low', None)

                if high is not None and low is not None:
                    return round((float(high) + float(low)) / 2, 2)

            # If no data for the specific date, fall back to current price
            print(f"🚨 OpenBB: No historical data for {ticker} on {date_obj}. Falling back to current price.")
            return OpenBBProvider.get_current_price(ticker)
        except Exception as e:
            print(f"❌ ERROR: Failed to get avg price for {ticker} on {date_obj} using OpenBB: {e}")
            return None
