"""PositionProtection — drawdown protection on CURRENT positions + expiry checks.

Runs every live-loop heartbeat (including while the kill switch is active,
because every action it produces reduces risk). It looks at the real broker
book, not just conviction-override records, and returns concrete orders.

Position level (drawdown measured from the higher of avg cost and the best
price seen since the position was opened):
    EQUITY / ETF   reduce 50% at -6%   ·  close at -10%
    LONG OPTION    reduce 50% at -40%  ·  close at -60%   (of premium)
    SHORT OPTION   close when the option has doubled against us (-100%)
Each REDUCE fires once per position until a new high is made.

Portfolio level (NAV drawdown from peak — same thresholds as the
conviction-override system):
    ≥ 4%  reduce every losing position by 50% (once per drawdown episode)
    ≥ 8%  close every position except hedges

Option expiries (OCC symbols, e.g. "AAPL  261120C00150000"):
    DTE ≤ 0                          close (expiring today)
    DTE ≤ 1 and short                close (assignment / pin risk)
    DTE ≤ 1 and long                 close (avoid exercise / overnight gap)
    DTE ≤ 3                          flag ROLL candidate (no order)

Conviction overrides: OverrideManager.check_expiries() and
check_drawdown_protection() are run and their results converted to orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger("metadron.execution.position_protection")

try:
    from ..risk.portfolio_mc_gate import parse_option_symbol
except Exception:  # degraded mode
    parse_option_symbol = None

# Position-level thresholds (fractions)
EQUITY_REDUCE_DD = 0.06
EQUITY_CLOSE_DD = 0.10
OPTION_REDUCE_DD = 0.40
OPTION_CLOSE_DD = 0.60
SHORT_OPTION_CLOSE_LOSS = 1.00
REDUCE_FRACTION = 0.50

# Portfolio-level thresholds (match conviction_override.py)
PORTFOLIO_REDUCE_DD = 0.04
PORTFOLIO_HARD_STOP_DD = 0.08

# Expiry rules (calendar days)
EXPIRY_CLOSE_DTE = 1
EXPIRY_ROLL_FLAG_DTE = 3


@dataclass
class ProtectionAction:
    ticker: str
    action: str                 # CLOSE | REDUCE | FLAG
    quantity: int = 0           # shares/contracts to trade (0 for FLAG)
    side: str = ""              # SELL (close long) | COVER (close short)
    reason: str = ""
    rule: str = ""
    is_option: bool = False

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "action": self.action, "quantity": self.quantity,
                "side": self.side, "reason": self.reason, "rule": self.rule,
                "is_option": self.is_option}


@dataclass
class _Track:
    peak_price: float = 0.0
    reduced: bool = False


class PositionProtection:
    """Stateful protection engine (one instance per live loop)."""

    def __init__(self, hedge_tickers: Optional[Set[str]] = None):
        self._tracks: Dict[str, _Track] = {}
        self._peak_nav: float = 0.0
        self._portfolio_reduced: bool = False
        self.hedge_tickers: Set[str] = set(hedge_tickers or ())
        self.last_actions: List[ProtectionAction] = []

    # ------------------------------------------------------------------
    def evaluate(self, positions: dict, nav: float, today: Optional[date] = None,
                 override_manager=None) -> List[ProtectionAction]:
        """Return protection orders for the current book.

        positions: {symbol: Position-like with quantity, avg_cost, current_price}
        """
        today = today or date.today()
        actions: Dict[str, ProtectionAction] = {}   # one action per ticker (strongest wins)
        flags: List[ProtectionAction] = []

        def put(a: ProtectionAction):
            prev = actions.get(a.ticker)
            rank = {"CLOSE": 2, "REDUCE": 1}
            if prev is None or rank.get(a.action, 0) > rank.get(prev.action, 0):
                actions[a.ticker] = a

        # ---- portfolio drawdown --------------------------------------
        self._peak_nav = max(self._peak_nav, nav)
        port_dd = 1.0 - nav / self._peak_nav if self._peak_nav > 0 else 0.0
        if port_dd < PORTFOLIO_REDUCE_DD / 2:
            self._portfolio_reduced = False      # episode over → re-arm

        live = set()
        for sym, pos in (positions or {}).items():
            qty = int(getattr(pos, "quantity", 0) or 0)
            if qty == 0:
                continue
            live.add(sym)
            px = float(getattr(pos, "current_price", 0) or 0)
            cost = float(getattr(pos, "avg_cost", 0) or 0)
            if px <= 0 or cost <= 0:
                continue
            opt = parse_option_symbol(sym) if parse_option_symbol else None
            is_short = qty < 0
            side = "COVER" if is_short else "SELL"
            abs_qty = abs(qty)
            is_hedge = sym in self.hedge_tickers

            # ---- expiries (options) ------------------------------------
            if opt is not None:
                dte = (date.fromisoformat(opt["expiry"]) - today).days
                if dte <= 0:
                    put(ProtectionAction(sym, "CLOSE", abs_qty, side,
                                         f"expires today ({opt['expiry']})", "EXPIRY_DTE0", True))
                elif dte <= EXPIRY_CLOSE_DTE:
                    why = "assignment/pin risk" if is_short else "avoid exercise/overnight gap"
                    put(ProtectionAction(sym, "CLOSE", abs_qty, side,
                                         f"DTE={dte}: {why}", "EXPIRY_DTE1", True))
                elif dte <= EXPIRY_ROLL_FLAG_DTE:
                    flags.append(ProtectionAction(sym, "FLAG", 0, "",
                                                  f"DTE={dte}: roll candidate", "EXPIRY_ROLL", True))

            # ---- position drawdown --------------------------------------
            tr = self._tracks.setdefault(sym, _Track(peak_price=max(px, cost) if not is_short else min(px, cost)))
            if is_short:
                tr.peak_price = min(tr.peak_price, px)          # best price for a short
                loss = (px - tr.peak_price) / tr.peak_price
            else:
                if px > tr.peak_price:
                    tr.peak_price = px
                    tr.reduced = False                         # new high → re-arm
                loss = (tr.peak_price - px) / tr.peak_price

            if is_hedge:
                continue                                       # never stop out a hedge on its own decay
            if opt is not None:
                if is_short:
                    if loss >= SHORT_OPTION_CLOSE_LOSS:
                        put(ProtectionAction(sym, "CLOSE", abs_qty, side,
                                             f"short option +{loss:.0%} against", "OPT_SHORT_STOP", True))
                elif loss >= OPTION_CLOSE_DD:
                    put(ProtectionAction(sym, "CLOSE", abs_qty, side,
                                         f"option -{loss:.0%} from peak", "OPT_CLOSE", True))
                elif loss >= OPTION_REDUCE_DD and not tr.reduced:
                    q = max(1, int(abs_qty * REDUCE_FRACTION))
                    put(ProtectionAction(sym, "REDUCE", q, side,
                                         f"option -{loss:.0%} from peak", "OPT_REDUCE", True))
                    tr.reduced = True
            else:
                if loss >= EQUITY_CLOSE_DD:
                    put(ProtectionAction(sym, "CLOSE", abs_qty, side,
                                         f"-{loss:.1%} from peak/cost", "EQ_CLOSE"))
                elif loss >= EQUITY_REDUCE_DD and not tr.reduced:
                    q = max(1, int(abs_qty * REDUCE_FRACTION))
                    put(ProtectionAction(sym, "REDUCE", q, side,
                                         f"-{loss:.1%} from peak/cost", "EQ_REDUCE"))
                    tr.reduced = True

            # ---- portfolio drawdown actions ------------------------------
            pnl = (px - cost) * qty
            if port_dd >= PORTFOLIO_HARD_STOP_DD:
                put(ProtectionAction(sym, "CLOSE", abs_qty, side,
                                     f"portfolio drawdown {port_dd:.1%} ≥ {PORTFOLIO_HARD_STOP_DD:.0%}",
                                     "PORT_HARD_STOP", opt is not None))
            elif port_dd >= PORTFOLIO_REDUCE_DD and not self._portfolio_reduced and pnl < 0:
                q = max(1, int(abs_qty * REDUCE_FRACTION))
                put(ProtectionAction(sym, "REDUCE", q, side,
                                     f"portfolio drawdown {port_dd:.1%}: trim loser",
                                     "PORT_REDUCE", opt is not None))

        if port_dd >= PORTFOLIO_REDUCE_DD:
            self._portfolio_reduced = True

        # forget closed positions
        for sym in list(self._tracks):
            if sym not in live:
                del self._tracks[sym]

        # ---- conviction overrides --------------------------------------
        if override_manager is not None:
            try:
                for ov in override_manager.check_expiries():
                    if ov.ticker in live:
                        q = abs(int(getattr(positions[ov.ticker], "quantity", 0)))
                        put(ProtectionAction(ov.ticker, "CLOSE", q, "SELL",
                                             "conviction override expired", "OVERRIDE_EXPIRY"))
                for ov in override_manager.check_drawdown_protection(nav):
                    if ov.ticker not in live:
                        continue
                    held = abs(int(getattr(positions[ov.ticker], "quantity", 0)))
                    status = getattr(getattr(ov, "status", None), "value", str(getattr(ov, "status", "")))
                    if status == "STOPPED_OUT":
                        put(ProtectionAction(ov.ticker, "CLOSE", held, "SELL",
                                             "override hard stop (portfolio DD)", "OVERRIDE_STOP"))
                    else:
                        put(ProtectionAction(ov.ticker, "REDUCE", max(1, held // 2), "SELL",
                                             "override drawdown reduction", "OVERRIDE_REDUCE"))
            except Exception as exc:
                logger.warning("Override protection checks failed: %s", exc)

        result = list(actions.values()) + flags
        self.last_actions = result
        for a in result:
            if a.action != "FLAG":
                logger.warning("PROTECTION %s %s x%d (%s) — %s", a.action, a.ticker,
                               a.quantity, a.rule, a.reason)
        return result

    def state(self) -> dict:
        return {
            "peak_nav": self._peak_nav,
            "portfolio_reduced": self._portfolio_reduced,
            "tracked_positions": len(self._tracks),
            "last_actions": [a.to_dict() for a in self.last_actions],
            "as_of": datetime.now().isoformat(),
        }
