"""PortfolioMCGate — Monte Carlo pre-trade risk gate (decision matrix + L7 execution).

One shared, fast, pure-numpy Monte Carlo used in three places:

    research   : MonteCarloRiskEngine output (per-ticker vol + tail scores) is
                 ingested via ``ingest_research()`` so the gate uses the
                 research-calibrated volatility and tail thickness.
    decision   : DecisionMatrix calls ``evaluate_trade()`` for every approved
                 trade with its proposed size, and scales the size down until
                 the portfolio VaR/CVaR limits hold (or rejects).
    execution  : L7RiskEngine calls ``check_order()`` as gate G11 right before
                 an order is sent, against the live book.

Model
-----
One-factor model with fat tails (Student-t, df≈4):
    r_i = beta_i * m + e_i
    m   ~ t(df) scaled to the market vol over the horizon
    e_i ~ t(df) scaled to the idiosyncratic vol of ticker i
Stocks/ETFs/futures: P&L = signed exposure × r_i.
Options: full Black-Scholes repricing of the contract at the horizon.

Limits (fractions of NAV, losses reported as POSITIVE numbers)
-------------------------------------------------------------
    MC_VAR95_MAX   1.5 %  1-day 95% VaR   (= $0.30M on $20M NAV, CLAUDE.md policy)
    MC_CVAR95_MAX  2.5 %  1-day 95% CVaR  (expected loss in the worst 5% of days)
A trade that LOWERS portfolio VaR/CVaR (e.g. a hedge) always passes the
VaR/CVaR checks even if the book is already above the limit.

Probability-of-profit (over the trade's holding horizon, per instrument)
------------------------------------------------------------------------
    EQUITY / ETF / FUTURE : P(profit) ≥ 52 %  and expected P&L > 0
    OPTION_LONG           : P(profit) ≥ 25 %  and expected P&L > 0
    OPTION_SHORT          : P(profit) ≥ 65 %  and expected P&L > 0
Hedges (``is_hedge=True``) skip the probability/EV test: they are negative-EV
insurance by design and are judged only on reducing tail risk.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("metadron.risk.portfolio_mc_gate")

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MC_VAR95_MAX = 0.015          # 1-day 95% VaR ≤ 1.5% of NAV
MC_CVAR95_MAX = 0.025         # 1-day 95% CVaR ≤ 2.5% of NAV
MC_PROB_PROFIT_MIN: Dict[str, float] = {
    "EQUITY": 0.52,
    "ETF": 0.52,
    "FUTURE": 0.52,
    "OPTION_LONG": 0.25,
    "OPTION_SHORT": 0.65,
}
VAR_HORIZON_DAYS = 1          # VaR / CVaR horizon
DEFAULT_HOLD_DAYS = 21        # P(profit) horizon for linear trades (alpha_pred is ~1M forward)
MARKET_VOL = 0.16             # annual market factor vol
DEFAULT_VOL = 0.30            # annual vol when nothing better is known
DEFAULT_BETA = 1.0
T_DF = 4.0                    # Student-t degrees of freedom (fat tails)
N_PATHS = 4000
TRADING_DAYS = 252
CONTRACT_MULTIPLIER = 100


# ---------------------------------------------------------------------------
# Black-Scholes (vectorised, no scipy dependency)
# ---------------------------------------------------------------------------
def _norm_cdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _erf(x):
    # Abramowitz-Stegun 7.1.26 (max error 1.5e-7), vectorised
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def bs_price(spot, strike, t_years, vol, is_call: bool, r: float = 0.04):
    """Vectorised Black-Scholes price; intrinsic value at/after expiry."""
    spot = np.asarray(spot, dtype=float)
    if t_years <= 1e-6 or vol <= 1e-6:
        return np.maximum(spot - strike, 0.0) if is_call else np.maximum(strike - spot, 0.0)
    sq = vol * math.sqrt(t_years)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(spot, 1e-12) / strike) + (r + 0.5 * vol * vol) * t_years) / sq
    d2 = d1 - sq
    disc = math.exp(-r * t_years)
    if is_call:
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Holding:
    """One exposure in the simulated book."""
    ticker: str
    instrument: str = "EQUITY"          # EQUITY | ETF | FUTURE | OPTION
    exposure: float = 0.0               # signed $ exposure for linear products
    # option fields
    contracts: float = 0.0              # signed: + long, - short
    spot: float = 0.0
    strike: float = 0.0
    dte: float = 0.0                    # calendar days to expiry
    iv: float = 0.0
    is_call: bool = True
    # risk inputs
    vol: float = 0.0                    # annual vol of the underlying
    beta: float = DEFAULT_BETA
    underlying: str = ""                # options: underlying ticker (defaults to ticker)


@dataclass
class MCResult:
    passed: bool = True
    instrument_class: str = "EQUITY"
    var95_pre: float = 0.0
    var95_post: float = 0.0
    cvar95_pre: float = 0.0
    cvar95_post: float = 0.0
    prob_profit: float = 0.0
    expected_pnl: float = 0.0
    hold_days: float = 0.0
    is_hedge: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "instrument_class": self.instrument_class,
            "var95_pre": round(self.var95_pre, 6),
            "var95_post": round(self.var95_post, 6),
            "cvar95_pre": round(self.cvar95_pre, 6),
            "cvar95_post": round(self.cvar95_post, 6),
            "prob_profit": round(self.prob_profit, 4),
            "expected_pnl": round(self.expected_pnl, 2),
            "hold_days": self.hold_days,
            "is_hedge": self.is_hedge,
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
class PortfolioMCGate:
    """Monte Carlo VaR / CVaR / P(profit) gate shared by decision + execution."""

    def __init__(self, n_paths: int = N_PATHS, seed: int = 7,
                 var_max: float = MC_VAR95_MAX, cvar_max: float = MC_CVAR95_MAX,
                 prob_min: Optional[Dict[str, float]] = None):
        self.n_paths = n_paths
        self.seed = seed
        self.var_max = var_max
        self.cvar_max = cvar_max
        self.prob_min = dict(prob_min or MC_PROB_PROFIT_MIN)
        self._holdings: List[Holding] = []
        self._vol_override: Dict[str, float] = {}
        self._tail_df: Dict[str, float] = {}
        self._research_ts: Optional[str] = None
        self._lock = threading.Lock()
        self.last_book_risk: dict = {}

    # -- research ingestion ----------------------------------------------
    def ingest_research(self, portfolio_risk) -> int:
        """Use MonteCarloRiskEngine output (PortfolioRisk or dict) to calibrate.

        Per-ticker annualized vol replaces the default/proposal vol, and the
        tail_risk_score (0..1) thickens the tails (lower Student-t df).
        Returns the number of tickers calibrated.
        """
        ticker_risks = getattr(portfolio_risk, "ticker_risks", None)
        if ticker_risks is None and isinstance(portfolio_risk, dict):
            ticker_risks = portfolio_risk.get("ticker_risks", {})
        n = 0
        for tkr, tr in (ticker_risks or {}).items():
            vol = getattr(tr, "annualized_vol", None)
            if vol is None and isinstance(tr, dict):
                vol = tr.get("annualized_vol")
            tail = getattr(tr, "tail_risk_score", None)
            if tail is None and isinstance(tr, dict):
                tail = tr.get("tail_risk_score")
            if vol and vol > 0:
                self._vol_override[tkr] = float(vol)
                n += 1
            if tail is not None:
                # tail 0 → df 8 (near normal) … tail 1 → df 3 (very fat)
                self._tail_df[tkr] = float(np.clip(8.0 - 5.0 * float(tail), 3.0, 8.0))
        ts = getattr(portfolio_risk, "timestamp", None)
        self._research_ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        logger.info("PortfolioMCGate: research calibration for %d tickers", n)
        return n

    # -- book ---------------------------------------------------------------
    def set_holdings(self, holdings: List[Holding]):
        with self._lock:
            self._holdings = list(holdings)

    def set_holdings_from_positions(self, positions: dict, nav: float = 0.0):
        """Build the book from broker positions {ticker: Position-like}."""
        hs: List[Holding] = []
        for tkr, pos in (positions or {}).items():
            qty = float(getattr(pos, "quantity", 0) or 0)
            if qty == 0:
                continue
            price = float(getattr(pos, "current_price", 0) or getattr(pos, "avg_cost", 0) or 0)
            opt = parse_option_symbol(tkr)
            if opt:
                hs.append(Holding(
                    ticker=tkr, instrument="OPTION", contracts=qty,
                    spot=float(getattr(pos, "underlying_price", 0) or opt["strike"]),
                    strike=opt["strike"], dte=opt["dte"],
                    iv=float(getattr(pos, "iv", 0) or 0.0) or self._vol_for(opt["underlying"], DEFAULT_VOL),
                    is_call=opt["is_call"], underlying=opt["underlying"],
                    vol=self._vol_for(opt["underlying"], DEFAULT_VOL),
                ))
            else:
                hs.append(Holding(ticker=tkr, instrument="EQUITY", exposure=qty * price,
                                  vol=self._vol_for(tkr, DEFAULT_VOL)))
        self.set_holdings(hs)

    def _vol_for(self, ticker: str, fallback: float) -> float:
        return float(self._vol_override.get(ticker, fallback or DEFAULT_VOL))

    # -- simulation -----------------------------------------------------------
    def _simulate_pnl(self, holdings: List[Holding], horizon_days: float,
                      drift: Optional[Dict[str, float]] = None,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulated P&L ($) of `holdings` over `horizon_days` (n_paths,)."""
        rng = rng or np.random.default_rng(self.seed)
        n = self.n_paths
        if not holdings:
            return np.zeros(n)
        h = max(horizon_days, 1e-3) / TRADING_DAYS

        def t_draw(df, size):
            df = max(df, 2.5)
            z = rng.standard_t(df, size=size)
            return z / math.sqrt(df / (df - 2.0))        # unit variance

        m = t_draw(T_DF, n) * MARKET_VOL * math.sqrt(h)
        idio: Dict[str, np.ndarray] = {}
        pnl = np.zeros(n)
        drift = drift or {}
        for hd in holdings:
            und = hd.underlying or hd.ticker
            vol = self._vol_override.get(und, hd.vol or DEFAULT_VOL)
            beta = hd.beta if hd.beta is not None else DEFAULT_BETA
            if und not in idio:
                idio_var = max(vol * vol - (beta * MARKET_VOL) ** 2, (0.25 * vol) ** 2)
                idio[und] = t_draw(self._tail_df.get(und, T_DF), n) * math.sqrt(idio_var * h)
            r = beta * m + idio[und] + drift.get(und, 0.0)
            if hd.instrument == "OPTION":
                spot = hd.spot if hd.spot > 0 else hd.strike
                iv = hd.iv if hd.iv > 0 else vol
                t0 = max(hd.dte, 0.0) / 365.0
                t1 = max(hd.dte - horizon_days * 365.0 / TRADING_DAYS, 0.0) / 365.0
                p0 = float(bs_price(spot, hd.strike, t0, iv, hd.is_call))
                p1 = bs_price(spot * np.exp(r), hd.strike, t1, iv, hd.is_call)
                pnl += hd.contracts * CONTRACT_MULTIPLIER * (p1 - p0)
            else:
                pnl += hd.exposure * r
        return pnl

    @staticmethod
    def _var_cvar(pnl: np.ndarray, nav: float) -> tuple:
        if nav <= 0 or pnl.size == 0:
            return 0.0, 0.0
        loss = -pnl / nav
        var = float(np.percentile(loss, 95))
        tail = loss[loss >= var]
        cvar = float(tail.mean()) if tail.size else var
        return max(var, 0.0), max(cvar, 0.0)

    def book_risk(self, nav: float) -> dict:
        """1-day VaR/CVaR of the current book (fractions of NAV)."""
        with self._lock:
            holdings = list(self._holdings)
        pnl = self._simulate_pnl(holdings, VAR_HORIZON_DAYS, rng=np.random.default_rng(self.seed))
        var, cvar = self._var_cvar(pnl, nav)
        self.last_book_risk = {"var95": var, "cvar95": cvar, "n_holdings": len(holdings),
                               "var_limit": self.var_max, "cvar_limit": self.cvar_max,
                               "research_calibrated_at": self._research_ts}
        return dict(self.last_book_risk)

    # -- trade evaluation -------------------------------------------------------
    @staticmethod
    def trade_to_holding(trade: dict, units: float) -> Holding:
        """Convert a proposal/order dict + size into a Holding."""
        side = str(trade.get("side", "long")).lower()
        sign = -1.0 if side in ("short", "sell", "sell_short") else 1.0
        instrument = str(trade.get("instrument_type", trade.get("instrument", "EQUITY"))).upper()
        ticker = trade.get("ticker", "UNKNOWN")
        vol = float(trade.get("volatility", 0.0) or trade.get("vol", 0.0) or DEFAULT_VOL)
        beta = float(trade.get("beta", DEFAULT_BETA) or DEFAULT_BETA)
        if instrument in ("OPTION", "OPTIONS"):
            und = trade.get("underlying") or ticker
            return Holding(
                ticker=ticker, instrument="OPTION", contracts=sign * units,
                spot=float(trade.get("underlying_price", 0) or trade.get("spot", 0) or 0),
                strike=float(trade.get("strike", 0) or 0),
                dte=float(trade.get("dte", 7) or 7),
                iv=float(trade.get("iv", 0) or 0) or vol,
                is_call=str(trade.get("option_type", "call")).lower().startswith("c"),
                vol=vol, beta=beta, underlying=und,
            )
        price = float(trade.get("price", 0) or 0)
        return Holding(ticker=ticker, instrument=instrument if instrument in ("ETF", "FUTURE") else "EQUITY",
                       exposure=sign * units * price, vol=vol, beta=beta)

    @staticmethod
    def instrument_class(trade: dict) -> str:
        instrument = str(trade.get("instrument_type", trade.get("instrument", "EQUITY"))).upper()
        if instrument in ("OPTION", "OPTIONS"):
            side = str(trade.get("side", "long")).lower()
            return "OPTION_SHORT" if side in ("short", "sell", "sell_short") else "OPTION_LONG"
        return instrument if instrument in ("ETF", "FUTURE") else "EQUITY"

    @staticmethod
    def horizon_return(trade: dict, hold_days: float = DEFAULT_HOLD_DAYS) -> float:
        """Expected underlying return over `hold_days` (drift for P(profit)).

        ``expected_return`` is taken as already matching the hold horizon.
        Alpha predictions (alpha_signal / alpha / alpha_pred) are ~1-month
        forward returns, so they are scaled linearly to the hold horizon.
        """
        v = trade.get("expected_return")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        for k in ("alpha_signal", "alpha", "alpha_pred"):
            v = trade.get(k)
            if v is not None:
                try:
                    return float(v) * (hold_days / DEFAULT_HOLD_DAYS)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def evaluate_trade(self, trade: dict, units: float, nav: float) -> MCResult:
        """Evaluate a trade of `units` shares/contracts against the current book."""
        res = MCResult(instrument_class=self.instrument_class(trade),
                       is_hedge=bool(trade.get("is_hedge", False)))
        if nav <= 0 or units <= 0:
            res.passed = False
            res.reasons.append("MC: no NAV or zero size")
            return res
        new = self.trade_to_holding(trade, units)
        with self._lock:
            book = list(self._holdings)

        # 1-day VaR / CVaR, same random draws pre and post (common random numbers)
        pre = self._simulate_pnl(book, VAR_HORIZON_DAYS, rng=np.random.default_rng(self.seed))
        post = self._simulate_pnl(book + [new], VAR_HORIZON_DAYS, rng=np.random.default_rng(self.seed))
        res.var95_pre, res.cvar95_pre = self._var_cvar(pre, nav)
        res.var95_post, res.cvar95_post = self._var_cvar(post, nav)

        risk_reducing_var = res.var95_post <= res.var95_pre + 1e-9
        risk_reducing_cvar = res.cvar95_post <= res.cvar95_pre + 1e-9
        if res.var95_post > self.var_max and not risk_reducing_var:
            res.reasons.append(f"MC VaR95 {res.var95_post:.2%} > {self.var_max:.2%} NAV")
        if res.cvar95_post > self.cvar_max and not risk_reducing_cvar:
            res.reasons.append(f"MC CVaR95 {res.cvar95_post:.2%} > {self.cvar_max:.2%} NAV")
        if res.is_hedge and not (risk_reducing_var or risk_reducing_cvar):
            res.reasons.append("MC: hedge does not reduce tail risk")

        # P(profit) + expected P&L of the trade alone over its holding horizon
        if new.instrument == "OPTION":
            hold = max(min(float(trade.get("hold_days", new.dte * TRADING_DAYS / 365.0)),
                           new.dte * TRADING_DAYS / 365.0), 1e-3)
        else:
            hold = float(trade.get("hold_days", DEFAULT_HOLD_DAYS))
        res.hold_days = round(hold, 2)
        und = new.underlying or new.ticker
        drift = {und: self.horizon_return(trade, hold)}
        alone = self._simulate_pnl([new], hold, drift=drift, rng=np.random.default_rng(self.seed + 1))
        # costs: half the quoted spread on entry and exit
        cost = 0.0
        spread_frac = trade.get("spread_frac")
        if spread_frac:
            notional = abs(new.exposure) if new.instrument != "OPTION" else abs(
                new.contracts) * CONTRACT_MULTIPLIER * float(trade.get("premium", trade.get("price", 0)) or 0)
            cost = float(spread_frac) * notional
        alone = alone - cost
        res.prob_profit = float((alone > 0).mean())
        res.expected_pnl = float(alone.mean())
        if not res.is_hedge:
            pmin = self.prob_min.get(res.instrument_class, 0.52)
            if res.prob_profit < pmin:
                res.reasons.append(f"MC P(profit) {res.prob_profit:.0%} < {pmin:.0%} ({res.instrument_class})")
            if res.expected_pnl <= 0:
                res.reasons.append(f"MC expected P&L ${res.expected_pnl:,.2f} ≤ 0")

        res.passed = not res.reasons
        return res

    def size_to_fit(self, trade: dict, units: int, nav: float, min_units: int = 1) -> tuple:
        """Halve `units` until VaR/CVaR fit (prob/EV are size-independent).

        Returns (units, MCResult). units == 0 means rejected.
        """
        res = self.evaluate_trade(trade, units, nav)
        u = units
        while not res.passed and u > min_units and all(
                r.startswith("MC VaR95") or r.startswith("MC CVaR95") for r in res.reasons):
            u = max(min_units, u // 2)
            res = self.evaluate_trade(trade, u, nav)
        return (u if res.passed else 0), res

    def check_order(self, ticker: str, side: str, quantity: float, price: float,
                    nav: float, product_type: str = "EQUITY", **option_fields) -> MCResult:
        """Execution-time check (L7 gate G11): VaR/CVaR only, live book."""
        trade = {"ticker": ticker, "side": "short" if side in ("SELL", "SHORT") else "long",
                 "price": price, "instrument_type": product_type, "is_hedge": True}
        trade.update(option_fields)
        res = self.evaluate_trade(trade, quantity, nav)
        # execution gate enforces tail risk only; P(profit) was decided upstream
        res.reasons = [r for r in res.reasons if r.startswith("MC VaR95") or r.startswith("MC CVaR95")]
        res.passed = not res.reasons
        return res


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_option_symbol(symbol: str) -> Optional[dict]:
    """Parse an OCC option symbol, e.g. 'AAPL  250117C00150000' or 'AAPL250117C00150000'."""
    import re
    from datetime import date
    if not symbol:
        return None
    m = re.match(r"^([A-Z.]{1,6})\s*(\d{6})([CP])(\d{8})$", symbol.strip().upper())
    if not m:
        return None
    und, ymd, cp, k = m.groups()
    try:
        exp = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    except ValueError:
        return None
    return {
        "underlying": und,
        "expiry": exp.isoformat(),
        "dte": (exp - date.today()).days,
        "is_call": cp == "C",
        "strike": int(k) / 1000.0,
    }


_shared_gate: Optional[PortfolioMCGate] = None
_shared_lock = threading.Lock()


def get_shared_mc_gate() -> PortfolioMCGate:
    """Process-wide gate so research, decision and execution share calibration."""
    global _shared_gate
    with _shared_lock:
        if _shared_gate is None:
            _shared_gate = PortfolioMCGate()
        return _shared_gate
