"""HeatmapEngine — Full GICS sector ASCII heatmap with 150+ tickers.

Provides:
    - Sector heatmap grouped by 11 GICS sectors (ANSI color-coded)
    - Factor ETF performance heatmap
    - Cross-asset correlation matrix in ASCII
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("numpy not available — HeatmapEngine degraded")

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("pandas not available — HeatmapEngine degraded")

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_WHITE = "\033[97m"

# ---------------------------------------------------------------------------
# GICS sector universe (150+ tickers across 11 sectors)
# ---------------------------------------------------------------------------
GICS_SECTORS = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "CSCO", "ORCL",
        "ACN", "INTC", "AMD", "TXN", "QCOM", "NOW", "INTU", "AMAT",
    ],
    "Health Care": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK",
    ],
    "Financials": [
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",
        "SPGI", "BLK", "C", "AXP", "SCHW", "CB", "MMC",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX",
        "BKNG", "CMG", "ORLY", "ROST", "DHI", "GM", "F",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
        "CHTR", "EA", "TTWO", "WBD", "PARA", "MTCH",
    ],
    "Industrials": [
        "GE", "CAT", "HON", "UNP", "BA", "RTX", "DE", "LMT",
        "MMM", "UPS", "FDX", "GD", "NOC", "WM", "ETN",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ",
        "CL", "EL", "KHC", "GIS", "SJM", "HSY", "K",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
        "PXD", "OXY", "HAL", "DVN", "HES", "FANG",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC", "XEL",
        "ED", "WEC", "PEG", "ES", "AWK", "DTE",
    ],
    "Real Estate": [
        "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL",
        "DLR", "VICI", "AVB", "EQR", "ARE", "MAA",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW",
        "DD", "VMC", "MLM", "PPG", "ALB", "EMN",
    ],
}


def _color_value(val: float, fmt: str = "+.2f") -> str:
    """Return ANSI-colored string for a numeric value."""
    if val > 0.1:
        color = ANSI_GREEN
    elif val < -0.1:
        color = ANSI_RED
    else:
        color = ANSI_YELLOW
    return f"{color}{val:{fmt}}%{ANSI_RESET}"


def _bar_block(val: float, max_width: int = 8) -> str:
    """Return a small horizontal bar for the value."""
    blocks = min(int(abs(val) * max_width / 5.0), max_width)
    char = "+" if val >= 0 else "-"
    color = ANSI_GREEN if val >= 0 else ANSI_RED
    return f"{color}{char * blocks}{ANSI_RESET}"


class HeatmapEngine:
    """Full GICS sector ASCII heatmap with 150+ tickers."""

    def __init__(self):
        self.sectors = GICS_SECTORS
        self._all_tickers = []
        for tickers in self.sectors.values():
            self._all_tickers.extend(tickers)
        logger.info("HeatmapEngine initialized — %d tickers across %d GICS sectors",
                     len(self._all_tickers), len(self.sectors))

    # ------------------------------------------------------------------
    # Sector heatmap
    # ------------------------------------------------------------------
    def generate_sector_heatmap(
        self,
        securities: list,
        returns_df: "pd.DataFrame" = None,
    ) -> str:
        """Generate ASCII heatmap grouped by 11 GICS sectors.

        Parameters
        ----------
        securities : list
            Ticker list (used to filter displayed tickers).
        returns_df : pd.DataFrame, optional
            DataFrame with columns = tickers and rows = dates.
            Must contain at least 22 rows for 1M lookback.
            If None, a placeholder heatmap is generated.
        """
        try:
            lines = []
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 90}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}  GICS SECTOR HEATMAP  —  {ts}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 90}{ANSI_RESET}")
            lines.append(f"  {'Ticker':<8} {'1D':>8} {'1W':>8} {'1M':>8}  {'Bar'}")
            lines.append(f"  {'-' * 50}")

            sec_set = set(securities) if securities else set(self._all_tickers)

            for sector, tickers in self.sectors.items():
                active = [t for t in tickers if t in sec_set]
                if not active:
                    continue
                lines.append(f"\n{ANSI_BOLD}{ANSI_WHITE}  [{sector}]{ANSI_RESET}")

                for ticker in active:
                    d1, w1, m1 = self._get_returns(ticker, returns_df)
                    bar = _bar_block(d1)
                    lines.append(
                        f"  {ticker:<8} {_color_value(d1):>18} "
                        f"{_color_value(w1):>18} {_color_value(m1):>18}  {bar}"
                    )

            lines.append(f"\n{ANSI_DIM}  {len(sec_set)} tickers displayed{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 90}{ANSI_RESET}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("generate_sector_heatmap failed: %s", e)
            return f"[HeatmapEngine] Sector heatmap error: {e}"

    # ------------------------------------------------------------------
    # Factor heatmap
    # ------------------------------------------------------------------
    def generate_factor_heatmap(
        self,
        factor_etfs: dict,
        returns_df: "pd.DataFrame" = None,
    ) -> str:
        """Factor ETF performance heatmap.

        Parameters
        ----------
        factor_etfs : dict
            Mapping of factor name -> ETF ticker (e.g. {"Momentum": "MTUM"}).
        returns_df : pd.DataFrame, optional
            Returns data.
        """
        try:
            lines = []
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}  FACTOR ETF HEATMAP{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")
            lines.append(f"  {'Factor':<20} {'ETF':<8} {'1D':>8} {'1W':>8} {'1M':>8}")
            lines.append(f"  {'-' * 60}")

            for factor_name, etf in factor_etfs.items():
                d1, w1, m1 = self._get_returns(etf, returns_df)
                lines.append(
                    f"  {factor_name:<20} {etf:<8} "
                    f"{_color_value(d1):>18} {_color_value(w1):>18} {_color_value(m1):>18}"
                )

            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("generate_factor_heatmap failed: %s", e)
            return f"[HeatmapEngine] Factor heatmap error: {e}"

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------
    def generate_correlation_heatmap(
        self,
        returns_df: "pd.DataFrame",
        top_n: int = 20,
    ) -> str:
        """Cross-asset correlation matrix in ASCII.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Returns DataFrame (columns = tickers, rows = dates).
        top_n : int
            Number of tickers to include.
        """
        try:
            if pd is None or np is None:
                return "[HeatmapEngine] numpy/pandas required for correlation heatmap"

            cols = list(returns_df.columns[:top_n])
            corr = returns_df[cols].corr()

            lines = []
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * (10 + 7 * len(cols))}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}  CORRELATION MATRIX (top {top_n}){ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * (10 + 7 * len(cols))}{ANSI_RESET}")

            # Header row
            header = "          " + "".join(f"{c:>7}" for c in cols)
            lines.append(header)

            for row_ticker in cols:
                row_str = f"  {row_ticker:<7} "
                for col_ticker in cols:
                    val = corr.loc[row_ticker, col_ticker]
                    if abs(val - 1.0) < 1e-6:
                        cell = f"{ANSI_DIM}  1.00{ANSI_RESET}"
                    elif val > 0.5:
                        cell = f"{ANSI_GREEN}{val:>6.2f}{ANSI_RESET}"
                    elif val < -0.2:
                        cell = f"{ANSI_RED}{val:>6.2f}{ANSI_RESET}"
                    else:
                        cell = f"{ANSI_YELLOW}{val:>6.2f}{ANSI_RESET}"
                    row_str += cell + " "
                lines.append(row_str)

            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * (10 + 7 * len(cols))}{ANSI_RESET}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("generate_correlation_heatmap failed: %s", e)
            return f"[HeatmapEngine] Correlation heatmap error: {e}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_returns(
        self, ticker: str, returns_df: "pd.DataFrame" = None
    ) -> tuple:
        """Extract 1D, 1W, 1M returns for a ticker from a DataFrame.

        Returns (day_pct, week_pct, month_pct) as floats.
        """
        if returns_df is not None and pd is not None and ticker in returns_df.columns:
            try:
                col = returns_df[ticker].dropna()
                d1 = float(col.iloc[-1] * 100) if len(col) >= 1 else 0.0
                w1 = float(col.iloc[-5:].sum() * 100) if len(col) >= 5 else 0.0
                m1 = float(col.iloc[-22:].sum() * 100) if len(col) >= 22 else 0.0
                return d1, w1, m1
            except Exception:
                pass
        return 0.0, 0.0, 0.0
