"""EventTailHedge — protective option hedges ahead of single-name events.

For every stock position with a scheduled binary event (earnings, FDA
decision, court ruling, deal vote …) inside the hedge window, buy a
protective option that expires just after the event:

    long stock  → buy puts     short stock → buy calls

Sizing
------
    gap scenario   = EVENT_SHOCK_MULT × expected event move (min 8%)
    gap loss       = position value × gap scenario
    hedge only if  gap loss ≥ MIN_GAP_LOSS_NAV (0.25% of NAV)
    strike         = spot × (1 ∓ 0.5 × expected move)  (slightly OTM)
    expiry         = first weekly Friday ≥ event date + 1 day
    contracts      = ceil(shares × HEDGE_RATIO / 100)
    premium budget = ≤ HEDGE_BUDGET_PCT_OF_POSITION (3%) of the position's
                     value and ≤ HEDGE_BUDGET_NAV (1%) of NAV for all hedges;
                     if over budget, the hedge ratio is cut, but never below
                     1 contract when the position is ≥ 100 shares.

Expected event move
-------------------
    supplied "expected_move" (e.g. from an ATM straddle) if present,
    else implied/realised vol × sqrt(days to expiry / 365) × EVENT_VOL_BUMP.

The premium is estimated with Black-Scholes at an event-inflated IV.
Orders are flagged is_hedge=True so the Monte Carlo gate judges them only on
reducing tail risk (VaR/CVaR), not on expected profit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("metadron.execution.event_tail_hedge")

try:
    from ..risk.portfolio_mc_gate import bs_price, parse_option_symbol
except Exception:  # degraded mode
    bs_price = None
    parse_option_symbol = None

EVENT_TYPES = {"EARNINGS", "FDA_CLINICAL", "LEGAL_REGULATORY", "MERGER_VOTE",
               "GUIDANCE", "INVESTOR_DAY", "CATALYST"}
HEDGE_WINDOW_DAYS = 7               # hedge events within the next 7 calendar days
EVENT_SHOCK_MULT = 2.0              # tail scenario = 2× the expected move
MIN_EVENT_SHOCK = 0.08              # never assume less than an 8% gap
MIN_GAP_LOSS_NAV = 0.0025           # ignore if the gap would cost < 0.25% NAV
HEDGE_RATIO = 1.0                   # hedge 100% of the shares
HEDGE_BUDGET_PCT_OF_POSITION = 0.03 # ≤ 3% of position value per event
HEDGE_BUDGET_NAV = 0.01             # ≤ 1% NAV across all event hedges
EVENT_VOL_BUMP = 1.5                # IV uplift into an event
DEFAULT_VOL = 0.35


@dataclass
class EventHedgeOrder:
    underlying: str
    option_type: str               # PUT | CALL
    strike: float
    expiry: str                    # YYYY-MM-DD
    contracts: int
    est_premium: float             # per share
    est_cost: float                # contracts × premium × 100
    event_type: str
    event_date: str
    gap_loss_hedged: float         # $ loss avoided in the tail scenario (approx.)
    reason: str = ""

    @property
    def occ_symbol(self) -> str:
        exp = date.fromisoformat(self.expiry)
        return (f"{self.underlying:<6}{exp:%y%m%d}{'P' if self.option_type == 'PUT' else 'C'}"
                f"{int(round(self.strike * 1000)):08d}")

    def to_dict(self) -> dict:
        return {
            "underlying": self.underlying, "option_type": self.option_type,
            "strike": self.strike, "expiry": self.expiry, "contracts": self.contracts,
            "est_premium": round(self.est_premium, 4), "est_cost": round(self.est_cost, 2),
            "event_type": self.event_type, "event_date": self.event_date,
            "gap_loss_hedged": round(self.gap_loss_hedged, 2), "occ_symbol": self.occ_symbol,
            "reason": self.reason,
        }


def _next_friday_after(d: date) -> date:
    d = d + timedelta(days=1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _round_strike(x: float) -> float:
    step = 1.0 if x < 100 else (2.5 if x < 250 else 5.0)
    return round(round(x / step) * step, 2)


class EventTailHedger:
    """Builds event-driven protective hedges for the current book."""

    def __init__(self):
        self.hedged: Dict[str, str] = {}      # underlying → event_date already hedged
        self.last_orders: List[EventHedgeOrder] = []

    def build(self, positions: dict, nav: float, events: Iterable[dict],
              vols: Optional[Dict[str, float]] = None,
              today: Optional[date] = None) -> List[EventHedgeOrder]:
        """positions: {ticker: Position-like}; events: [{ticker, date, type, expected_move?}]"""
        today = today or date.today()
        vols = vols or {}
        budget_left = HEDGE_BUDGET_NAV * nav
        by_ticker: Dict[str, dict] = {}
        for ev in events or []:
            tkr = str(ev.get("ticker", "")).upper()
            etype = str(getattr(ev.get("type"), "value", ev.get("type", "EARNINGS"))).upper()
            if not tkr or (etype not in EVENT_TYPES and "EARN" not in etype):
                continue
            try:
                ed = date.fromisoformat(str(ev.get("date", ""))[:10])
            except ValueError:
                continue
            if not (0 <= (ed - today).days <= HEDGE_WINDOW_DAYS):
                continue
            if tkr not in by_ticker or ed < date.fromisoformat(by_ticker[tkr]["date"][:10]):
                by_ticker[tkr] = {**ev, "type": etype, "date": ed.isoformat()}

        # positions already carrying protection (existing long puts/calls on the underlying)
        protected = set()
        for sym, pos in (positions or {}).items():
            opt = parse_option_symbol(sym) if parse_option_symbol else None
            if opt and int(getattr(pos, "quantity", 0) or 0) > 0:
                protected.add((opt["underlying"], "PUT" if not opt["is_call"] else "CALL"))

        orders: List[EventHedgeOrder] = []
        candidates = []
        for tkr, ev in by_ticker.items():
            pos = (positions or {}).get(tkr)
            if pos is None:
                continue
            qty = int(getattr(pos, "quantity", 0) or 0)
            spot = float(getattr(pos, "current_price", 0) or 0)
            if qty == 0 or spot <= 0 or abs(qty) < 100:
                continue          # < 100 shares cannot be hedged with listed contracts
            otype = "PUT" if qty > 0 else "CALL"
            if (tkr, otype) in protected or self.hedged.get(tkr) == ev["date"]:
                continue
            ed = date.fromisoformat(ev["date"])
            expiry = _next_friday_after(ed)
            dte = max((expiry - today).days, 1)
            vol = float(vols.get(tkr, 0) or DEFAULT_VOL)
            exp_move = float(ev.get("expected_move") or vol * math.sqrt(dte / 365.0) * EVENT_VOL_BUMP)
            shock = max(EVENT_SHOCK_MULT * exp_move, MIN_EVENT_SHOCK)
            value = abs(qty) * spot
            gap_loss = value * shock
            if nav <= 0 or gap_loss < MIN_GAP_LOSS_NAV * nav:
                continue
            candidates.append((gap_loss, tkr, ev, qty, spot, otype, expiry, dte, vol, exp_move, shock, value))

        # biggest tail exposures first, until the NAV budget runs out
        for gap_loss, tkr, ev, qty, spot, otype, expiry, dte, vol, exp_move, shock, value in sorted(
                candidates, reverse=True):
            k = _round_strike(spot * (1 - 0.5 * exp_move) if otype == "PUT" else spot * (1 + 0.5 * exp_move))
            iv = vol * EVENT_VOL_BUMP
            if bs_price is None:
                break
            prem = float(bs_price(spot, k, dte / 365.0, iv, is_call=(otype == "CALL")))
            if prem <= 0:
                continue
            contracts = math.ceil(abs(qty) * HEDGE_RATIO / 100)
            per_contract = prem * 100
            cap = min(HEDGE_BUDGET_PCT_OF_POSITION * value, budget_left)
            if contracts * per_contract > cap:
                contracts = int(cap // per_contract)
                if contracts < 1:
                    contracts = 1 if per_contract <= budget_left else 0
            if contracts < 1:
                logger.info("Event hedge %s skipped: 1 contract $%.0f exceeds remaining budget $%.0f",
                            tkr, per_contract, budget_left)
                continue
            cost = contracts * per_contract
            budget_left -= cost
            # payoff in the tail scenario (intrinsic at the shocked price)
            shocked = spot * (1 - shock) if otype == "PUT" else spot * (1 + shock)
            payoff = contracts * 100 * max((k - shocked) if otype == "PUT" else (shocked - k), 0.0)
            order = EventHedgeOrder(
                underlying=tkr, option_type=otype, strike=k, expiry=expiry.isoformat(),
                contracts=contracts, est_premium=prem, est_cost=cost,
                event_type=ev["type"], event_date=ev["date"], gap_loss_hedged=payoff,
                reason=(f"{ev['type']} {ev['date']}: {abs(qty)} sh, expected move ±{exp_move:.1%}, "
                        f"tail gap {shock:.0%} = ${gap_loss:,.0f}; {contracts}x {otype} {k} "
                        f"exp {expiry} ≈ ${cost:,.0f}"),
            )
            orders.append(order)
            self.hedged[tkr] = ev["date"]
            logger.info("EVENT HEDGE %s", order.reason)
        self.last_orders = orders
        return orders


def earnings_events_from_calendar(df, tickers: Optional[Iterable[str]] = None) -> List[dict]:
    """Convert an OpenBB earnings-calendar frame to event dicts."""
    out: List[dict] = []
    if df is None or getattr(df, "empty", True):
        return out
    want = {t.upper() for t in tickers} if tickers else None
    sym_col = next((c for c in ("symbol", "ticker") if c in df.columns), None)
    date_col = next((c for c in ("report_date", "date", "earnings_date") if c in df.columns), None)
    if sym_col is None:
        return out
    for idx, row in df.iterrows():
        tkr = str(row[sym_col]).upper()
        if want is not None and tkr not in want:
            continue
        d = row[date_col] if date_col else idx
        out.append({"ticker": tkr, "date": str(d)[:10], "type": "EARNINGS"})
    return out
