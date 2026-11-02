"""PortfolioAnalytics — Deep portfolio analytics (pure numpy/pandas).

Provides:
    - Sharpe, Sortino, Calmar ratios
    - Max drawdown computation
    - Factor exposure (beta to each factor)
    - Correlation matrix
    - Rolling metrics (Sharpe, vol, drawdown)
    - Risk decomposition (systematic vs idiosyncratic)
    - Comprehensive analytics report
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("numpy not available — PortfolioAnalytics degraded")

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("pandas not available — PortfolioAnalytics degraded")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"


class PortfolioAnalytics:
    """Deep portfolio analytics — pure numpy/pandas."""

    # ------------------------------------------------------------------
    # Risk-adjusted return ratios
    # ------------------------------------------------------------------
    @staticmethod
    def compute_sharpe(returns, rf: float = 0.04) -> float:
        """Annualized Sharpe ratio.

        Parameters
        ----------
        returns : array-like
            Daily returns series.
        rf : float
            Annual risk-free rate (default 4%).

        Returns
        -------
        float
            Annualized Sharpe ratio.
        """
        try:
            if np is None:
                return 0.0
            r = np.asarray(returns, dtype=float)
            r = r[~np.isnan(r)]
            if len(r) < 2:
                return 0.0
            daily_rf = rf / TRADING_DAYS_PER_YEAR
            excess = r - daily_rf
            std = np.std(excess, ddof=1)
            if std < 1e-10:
                return 0.0
            return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS_PER_YEAR))
        except Exception as e:
            logger.error("compute_sharpe failed: %s", e)
            return 0.0

    @staticmethod
    def compute_sortino(returns, rf: float = 0.04) -> float:
        """Annualized Sortino ratio (downside deviation).

        Parameters
        ----------
        returns : array-like
            Daily returns series.
        rf : float
            Annual risk-free rate.

        Returns
        -------
        float
            Annualized Sortino ratio.
        """
        try:
            if np is None:
                return 0.0
            r = np.asarray(returns, dtype=float)
            r = r[~np.isnan(r)]
            if len(r) < 2:
                return 0.0
            daily_rf = rf / TRADING_DAYS_PER_YEAR
            excess = r - daily_rf
            downside = excess[excess < 0]
            if len(downside) < 1:
                return float("inf") if np.mean(excess) > 0 else 0.0
            down_std = np.sqrt(np.mean(downside ** 2))
            if down_std < 1e-10:
                return 0.0
            return float(np.mean(excess) / down_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        except Exception as e:
            logger.error("compute_sortino failed: %s", e)
            return 0.0

    @staticmethod
    def compute_calmar(returns) -> float:
        """Calmar ratio (annualized return / max drawdown).

        Parameters
        ----------
        returns : array-like
            Daily returns series.

        Returns
        -------
        float
            Calmar ratio.
        """
        try:
            if np is None:
                return 0.0
            r = np.asarray(returns, dtype=float)
            r = r[~np.isnan(r)]
            if len(r) < 2:
                return 0.0
            annual_ret = np.mean(r) * TRADING_DAYS_PER_YEAR
            max_dd = PortfolioAnalytics.compute_max_drawdown(r)
            if abs(max_dd) < 1e-10:
                return 0.0
            return float(annual_ret / abs(max_dd))
        except Exception as e:
            logger.error("compute_calmar failed: %s", e)
            return 0.0

    @staticmethod
    def compute_max_drawdown(returns) -> float:
        """Maximum drawdown from peak.

        Parameters
        ----------
        returns : array-like
            Daily returns series.

        Returns
        -------
        float
            Maximum drawdown as a negative fraction (e.g., -0.15 = -15%).
        """
        try:
            if np is None:
                return 0.0
            r = np.asarray(returns, dtype=float)
            r = r[~np.isnan(r)]
            if len(r) < 1:
                return 0.0
            cumulative = np.cumprod(1.0 + r)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative / running_max - 1.0
            return float(np.min(drawdowns))
        except Exception as e:
            logger.error("compute_max_drawdown failed: %s", e)
            return 0.0

    # ------------------------------------------------------------------
    # Factor analysis
    # ------------------------------------------------------------------
    @staticmethod
    def compute_factor_exposure(returns, factor_returns) -> dict:
        """Compute beta to each factor via OLS regression.

        Parameters
        ----------
        returns : array-like
            Portfolio daily returns.
        factor_returns : dict or pd.DataFrame
            Factor name -> daily returns array, or DataFrame with factor columns.

        Returns
        -------
        dict
            Factor name -> beta value.
        """
        try:
            if np is None:
                return {}
            r = np.asarray(returns, dtype=float)

            if pd is not None and isinstance(factor_returns, pd.DataFrame):
                factors = {col: factor_returns[col].values for col in factor_returns.columns}
            elif isinstance(factor_returns, dict):
                factors = {k: np.asarray(v, dtype=float) for k, v in factor_returns.items()}
            else:
                return {}

            betas = {}
            for name, f_ret in factors.items():
                try:
                    min_len = min(len(r), len(f_ret))
                    if min_len < 5:
                        betas[name] = 0.0
                        continue
                    y = r[:min_len]
                    x = f_ret[:min_len]
                    # Remove NaN
                    mask = ~(np.isnan(y) | np.isnan(x))
                    y, x = y[mask], x[mask]
                    if len(y) < 5:
                        betas[name] = 0.0
                        continue
                    cov = np.cov(y, x)
                    var_x = cov[1, 1]
                    if var_x < 1e-12:
                        betas[name] = 0.0
                    else:
                        betas[name] = float(cov[0, 1] / var_x)
                except Exception:
                    betas[name] = 0.0

            return betas
        except Exception as e:
            logger.error("compute_factor_exposure failed: %s", e)
            return {}

    @staticmethod
    def compute_correlation_matrix(returns) -> "pd.DataFrame":
        """Compute pairwise correlation matrix.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame with columns = tickers, rows = dates.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        try:
            if pd is None:
                return None
            if isinstance(returns, pd.DataFrame):
                return returns.corr()
            return pd.DataFrame(np.corrcoef(np.asarray(returns, dtype=float)))
        except Exception as e:
            logger.error("compute_correlation_matrix failed: %s", e)
            return pd.DataFrame() if pd else None

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_rolling_metrics(returns, window: int = 63) -> "pd.DataFrame":
        """Compute rolling Sharpe, volatility, and drawdown.

        Parameters
        ----------
        returns : array-like or pd.Series
            Daily returns.
        window : int
            Rolling window in trading days (default 63 ~ 3 months).

        Returns
        -------
        pd.DataFrame
            Columns: rolling_sharpe, rolling_vol, rolling_drawdown.
        """
        try:
            if pd is None or np is None:
                return None

            if isinstance(returns, pd.Series):
                r = returns
            else:
                r = pd.Series(np.asarray(returns, dtype=float))

            n = len(r)
            if n < window:
                logger.warning("Returns length (%d) < window (%d)", n, window)
                window = max(n, 2)

            rolling_mean = r.rolling(window).mean()
            rolling_std = r.rolling(window).std()
            rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)).fillna(0)
            rolling_vol = (rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)).fillna(0)

            # Rolling drawdown
            cumulative = (1 + r).cumprod()
            rolling_max = cumulative.rolling(window, min_periods=1).max()
            rolling_dd = (cumulative / rolling_max - 1).fillna(0)

            return pd.DataFrame({
                "rolling_sharpe": rolling_sharpe,
                "rolling_vol": rolling_vol,
                "rolling_drawdown": rolling_dd,
            })
        except Exception as e:
            logger.error("compute_rolling_metrics failed: %s", e)
            return pd.DataFrame() if pd else None

    # ------------------------------------------------------------------
    # Risk decomposition
    # ------------------------------------------------------------------
    @staticmethod
    def compute_risk_decomposition(returns, factor_returns) -> dict:
        """Decompose risk into systematic and idiosyncratic components.

        Uses multi-factor regression: R = alpha + sum(beta_i * F_i) + epsilon

        Parameters
        ----------
        returns : array-like
            Portfolio daily returns.
        factor_returns : dict or pd.DataFrame
            Factor returns (same as compute_factor_exposure).

        Returns
        -------
        dict
            Keys: total_var, systematic_var, idiosyncratic_var,
                  systematic_pct, idiosyncratic_pct, r_squared.
        """
        try:
            if np is None:
                return {}

            r = np.asarray(returns, dtype=float)

            # Build factor matrix
            if pd is not None and isinstance(factor_returns, pd.DataFrame):
                F = factor_returns.values
                min_len = min(len(r), F.shape[0])
                r = r[:min_len]
                F = F[:min_len]
            elif isinstance(factor_returns, dict):
                arrays = [np.asarray(v, dtype=float) for v in factor_returns.values()]
                min_len = min(len(r), *[len(a) for a in arrays])
                r = r[:min_len]
                F = np.column_stack([a[:min_len] for a in arrays])
            else:
                return {}

            # Remove NaN rows
            mask = ~np.any(np.isnan(F), axis=1) & ~np.isnan(r)
            r = r[mask]
            F = F[mask]

            if len(r) < F.shape[1] + 2:
                return {}

            # OLS: R = F @ beta + alpha + epsilon
            X = np.column_stack([np.ones(len(r)), F])
            beta, residuals, _, _ = np.linalg.lstsq(X, r, rcond=None)
            predicted = X @ beta
            epsilon = r - predicted

            total_var = float(np.var(r, ddof=1))
            idio_var = float(np.var(epsilon, ddof=1))
            sys_var = max(total_var - idio_var, 0.0)

            r_sq = 1.0 - (idio_var / total_var) if total_var > 1e-12 else 0.0

            return {
                "total_var": total_var,
                "systematic_var": sys_var,
                "idiosyncratic_var": idio_var,
                "systematic_pct": sys_var / total_var * 100 if total_var > 1e-12 else 0.0,
                "idiosyncratic_pct": idio_var / total_var * 100 if total_var > 1e-12 else 0.0,
                "r_squared": r_sq,
            }
        except Exception as e:
            logger.error("compute_risk_decomposition failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_analytics_report(
        self,
        portfolio_returns,
        benchmark_returns=None,
    ) -> str:
        """Generate comprehensive analytics report.

        Parameters
        ----------
        portfolio_returns : array-like or pd.Series
            Portfolio daily returns.
        benchmark_returns : array-like or pd.Series, optional
            Benchmark daily returns for comparison.

        Returns
        -------
        str
            Formatted ASCII analytics report.
        """
        try:
            if np is None:
                return "[PortfolioAnalytics] numpy required"

            r = np.asarray(portfolio_returns, dtype=float)
            r = r[~np.isnan(r)]

            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            lines = []
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}  PORTFOLIO ANALYTICS REPORT  —  {ts}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")

            # Return stats
            ann_ret = float(np.mean(r) * TRADING_DAYS_PER_YEAR * 100)
            ann_vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)
            sharpe = self.compute_sharpe(r)
            sortino = self.compute_sortino(r)
            calmar = self.compute_calmar(r)
            max_dd = self.compute_max_drawdown(r) * 100
            cumulative = float((np.cumprod(1 + r)[-1] - 1) * 100) if len(r) > 0 else 0.0

            def _c(v, fmt="+.2f"):
                color = ANSI_GREEN if v > 0 else ANSI_RED if v < 0 else ANSI_YELLOW
                return f"{color}{v:{fmt}}{ANSI_RESET}"

            lines.append(f"\n  {ANSI_BOLD}Return & Risk{ANSI_RESET}")
            lines.append(f"  {'Cumulative Return:':<30} {_c(cumulative)}%")
            lines.append(f"  {'Annualized Return:':<30} {_c(ann_ret)}%")
            lines.append(f"  {'Annualized Volatility:':<30} {ann_vol:.2f}%")
            lines.append(f"  {'Max Drawdown:':<30} {_c(max_dd)}%")

            lines.append(f"\n  {ANSI_BOLD}Risk-Adjusted Metrics{ANSI_RESET}")
            lines.append(f"  {'Sharpe Ratio:':<30} {_c(sharpe)}")
            lines.append(f"  {'Sortino Ratio:':<30} {_c(sortino)}")
            lines.append(f"  {'Calmar Ratio:':<30} {_c(calmar)}")

            # Distribution stats
            skew = float(np.mean(((r - np.mean(r)) / np.std(r)) ** 3)) if np.std(r) > 1e-10 else 0.0
            kurt = float(np.mean(((r - np.mean(r)) / np.std(r)) ** 4) - 3) if np.std(r) > 1e-10 else 0.0
            win_rate = float(np.sum(r > 0) / len(r) * 100) if len(r) > 0 else 0.0

            lines.append(f"\n  {ANSI_BOLD}Distribution{ANSI_RESET}")
            lines.append(f"  {'Win Rate:':<30} {win_rate:.1f}%")
            lines.append(f"  {'Skewness:':<30} {skew:.3f}")
            lines.append(f"  {'Excess Kurtosis:':<30} {kurt:.3f}")
            lines.append(f"  {'Observations:':<30} {len(r)}")

            # Benchmark comparison
            if benchmark_returns is not None:
                b = np.asarray(benchmark_returns, dtype=float)
                b = b[~np.isnan(b)]
                min_len = min(len(r), len(b))
                if min_len > 5:
                    r_b, b_b = r[:min_len], b[:min_len]
                    b_sharpe = self.compute_sharpe(b_b)
                    b_ann_ret = float(np.mean(b_b) * TRADING_DAYS_PER_YEAR * 100)
                    b_cum = float((np.cumprod(1 + b_b)[-1] - 1) * 100)
                    excess_ret = ann_ret - b_ann_ret
                    info_ratio = 0.0
                    tracking_err = np.std(r_b - b_b, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                    if tracking_err > 1e-10:
                        info_ratio = (np.mean(r_b - b_b) * TRADING_DAYS_PER_YEAR) / tracking_err

                    lines.append(f"\n  {ANSI_BOLD}vs Benchmark{ANSI_RESET}")
                    lines.append(f"  {'Benchmark Cumulative:':<30} {_c(b_cum)}%")
                    lines.append(f"  {'Benchmark Sharpe:':<30} {_c(b_sharpe)}")
                    lines.append(f"  {'Excess Return (ann.):':<30} {_c(excess_ret)}%")
                    lines.append(f"  {'Information Ratio:':<30} {_c(float(info_ratio))}")
                    lines.append(f"  {'Tracking Error:':<30} {float(tracking_err) * 100:.2f}%")

            lines.append(f"\n{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("generate_analytics_report failed: %s", e)
            return f"[PortfolioAnalytics] Report error: {e}"
