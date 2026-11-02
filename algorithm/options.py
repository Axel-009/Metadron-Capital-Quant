"""Options engine — Black-Scholes pricing, greeks, implied vol, real-world P(profit), contract choice.

Pure Python (math only). Port of the engine's OptionsEngine essentials for the LEAN algorithm.

    price / greeks      Black-Scholes with continuous dividend yield q
    implied_vol         bisection on price (robust, no derivative)
    real-world gate     P(profit at expiry) and expected P&L under a drift μ (the alpha forecast), not r:
                          long call profits if S_T > K + premium;  long put if S_T < K − premium
                          E[payoff] = S·e^{μT}·N(d1μ) − K·N(d2μ)  (call), analogous for puts
    selection           among quoted contracts in a DTE window: |Δ| closest to the target, spread ≤ 10% of
                        mid (no quote → skip), open interest ≥ MIN_OPEN_INTEREST when known
    sizing              full Kelly on premium; if one contract's premium exceeds the Kelly/cap budget, the
                        1-contract exception buys exactly 1 contract when it still passes the P(profit) gate
                        and costs ≤ ONE_CONTRACT_MAX_NAV of NAV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

MAX_SPREAD_FRACTION = 0.10
MIN_OPEN_INTEREST = 100
P_PROFIT_LONG = 0.25
P_PROFIT_SHORT = 0.65
ONE_CONTRACT_MAX_NAV = 0.02     # the 1-contract exception never risks more than 2% NAV in premium
MULTIPLIER = 100


def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _n(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _d1d2(S, K, T, sigma, r, q):
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / st
    return d1, d1 - st


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.04, q: float = 0.0, call: bool = True) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, sigma, r, q)
    if call:
        return S * math.exp(-q * T) * _N(d1) - K * math.exp(-r * T) * _N(d2)
    return K * math.exp(-r * T) * _N(-d2) - S * math.exp(-q * T) * _N(-d1)


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float


def greeks(S, K, T, sigma, r=0.04, q=0.0, call=True) -> Greeks:
    if T <= 0 or sigma <= 0:
        itm = (S > K) if call else (S < K)
        return Greeks((1.0 if itm else 0.0) * (1 if call else -1), 0.0, 0.0, 0.0)
    d1, d2 = _d1d2(S, K, T, sigma, r, q)
    eq, er = math.exp(-q * T), math.exp(-r * T)
    delta = eq * _N(d1) if call else eq * (_N(d1) - 1)
    gamma = eq * _n(d1) / (S * sigma * math.sqrt(T))
    vega = S * eq * _n(d1) * math.sqrt(T) / 100
    if call:
        theta = (-S * eq * _n(d1) * sigma / (2 * math.sqrt(T)) - r * K * er * _N(d2) + q * S * eq * _N(d1)) / 365
    else:
        theta = (-S * eq * _n(d1) * sigma / (2 * math.sqrt(T)) + r * K * er * _N(-d2) - q * S * eq * _N(-d1)) / 365
    return Greeks(delta, gamma, vega, theta)


def implied_vol(price: float, S: float, K: float, T: float, r: float = 0.04, q: float = 0.0,
                call: bool = True) -> Optional[float]:
    if price <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0) if call else max(K - S, 0.0)
    if price < intrinsic - 1e-6:
        return None
    lo, hi = 1e-4, 5.0
    if bs_price(S, K, T, hi, r, q, call) < price:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, r, q, call) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def real_world(S: float, K: float, T: float, sigma: float, premium: float, mu: float, call: bool = True):
    """(P(profit at expiry), expected P&L per share) for a LONG option under drift μ."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0, -premium
    st = sigma * math.sqrt(T)
    be = K + premium if call else K - premium
    if be <= 0:
        p = 1.0 if call else 0.0
    else:
        z = (math.log(S / be) + (mu - 0.5 * sigma * sigma) * T) / st
        p = _N(z) if call else _N(-z)
    d1 = (math.log(S / K) + (mu + 0.5 * sigma * sigma) * T) / st
    d2 = d1 - st
    fwd = S * math.exp(mu * T)
    payoff = fwd * _N(d1) - K * _N(d2) if call else K * _N(-d2) - fwd * _N(-d1)
    return p, payoff - premium


@dataclass
class Quote:
    symbol: object
    strike: float
    dte: int
    call: bool
    bid: float
    ask: float
    open_interest: Optional[float] = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_ok(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.mid > 0 and (self.ask - self.bid) / self.mid <= MAX_SPREAD_FRACTION


@dataclass
class Pick:
    quote: Quote
    iv: float
    delta: float


def select(quotes: Iterable[Quote], S: float, call: bool, target_delta: float, dte_lo: int, dte_hi: int,
           r: float = 0.04) -> Optional[Pick]:
    best = None
    for qt in quotes:
        if qt.call != call or not (dte_lo <= qt.dte <= dte_hi) or not qt.spread_ok:
            continue
        if qt.open_interest is not None and qt.open_interest < MIN_OPEN_INTEREST:
            continue
        T = qt.dte / 365
        iv = implied_vol(qt.mid, S, qt.strike, T, r, 0.0, call)
        if iv is None:
            continue
        d = greeks(S, qt.strike, T, iv, r, 0.0, call).delta
        err = abs(abs(d) - abs(target_delta))
        if best is None or err < best[0]:
            best = (err, Pick(qt, iv, d))
    return best[1] if best else None


def contracts(budget: float, nav: float, premium: float, p_profit: float, exp_pnl: float,
              long: bool = True) -> int:
    """Full-Kelly premium budget (fraction of NAV) → number of contracts, with the 1-contract exception."""
    cost = premium * MULTIPLIER
    if cost <= 0 or exp_pnl <= 0 or p_profit < (P_PROFIT_LONG if long else P_PROFIT_SHORT):
        return 0
    n = int(budget * nav // cost)
    if n >= 1:
        return n
    return 1 if cost <= ONE_CONTRACT_MAX_NAV * nav else 0


def kelly_option_fraction(p_profit: float, exp_pnl: float, premium: float) -> float:
    """Kelly for a binary-ish long option: f = p − (1 − p)/b with b = average win / premium."""
    if premium <= 0 or p_profit <= 0:
        return 0.0
    avg_win = (exp_pnl + premium) / max(p_profit, 1e-9)      # E[payoff | profit] approx
    b = max(avg_win / premium - 1, 1e-9)
    return max(0.0, p_profit - (1 - p_profit) / b)
