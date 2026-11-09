# region imports
from AlgorithmImports import *
# endregion
"""Options desk: plans (rotation / escalation / LEAPS / straddles / alpha calls), contract selection, real-world
gate + Kelly sizing with the 1-contract exception, premium caps, exits — mixed into MetadronCorridor."""

import feed_guard as fg
import schedule as hb
import options as op
import decision as dm
import protection as pr
from rotation import realised_vol as name_vol
from sectors import SECTOR_ETF_FALLBACK


class OptionsDeskMixin:
    # ------------------------------------------------------------------ options
    def _sleeve_of(self, ticker):
        d = self.decisions.get(ticker)
        if d is not None:
            return d.sleeve
        sc = self.scores.get(ticker)
        return sc.capital.sleeve if sc is not None else "IG"

    def _option_plans(self, plan, ev, kron):
        """Options on SINGLE NAMES: rotation calls into the favoured GICS sectors, escalation calls on the best
        Energy names, LEAPS / straddles on crashed names, alpha calls on the strongest names."""
        plans = []
        approved = sorted((d for d in self.decisions.values() if d.approved), key=lambda d: -d.composite)
        by_sector = {}
        for d in approved:
            by_sector.setdefault(d.sector, []).append(d)
        for a in plan.actions:
            if a.kind == "ROTATE":                       # a theme crashed: calls on the best names of the destination sectors
                for d in by_sector.get(a.ticker, [])[:2]:
                    plans.append(dict(kind="ROTATION_CALL", ticker=d.ticker, call=True, delta=0.5, dte=(45, 120),
                                      budget=None, mu=d.mu_alpha + 0.04, rv=d.idio_vol, iv_max=1.25,
                                      reason=f"rotation into {a.ticker}: composite {d.composite:.2f}"))
            elif a.kind == "ESCALATION_CALLS":
                names = [d.ticker for d in by_sector.get(a.ticker, [])[:2]] or \
                        [SECTOR_ETF_FALLBACK.get(a.ticker, "XLE")]
                for t in names:
                    plans.append(dict(kind=a.kind, ticker=t, call=True, delta=0.4, dte=(45, 75),
                                      budget=a.premium / len(names), mu=0.3, reason=a.reason))
            elif a.kind in ("LEAPS", "LEAPS_SPREAD"):
                plans.append(dict(kind=a.kind, ticker=a.ticker, call=True, delta=0.6, dte=(270, 540), budget=a.premium,
                                  mu=0.25, reason=a.reason, short_delta=0.3 if a.kind == "LEAPS_SPREAD" else None))
            elif a.kind == "STRADDLE":
                for call in (True, False):
                    plans.append(dict(kind="STRADDLE", ticker=a.ticker, call=call, delta=0.5, dte=(7, 30),
                                      budget=a.premium / 2, mu=0.0, reason=a.reason, straddle=True))
        for d in approved[:5]:                           # alpha calls: only when implied ≤ realised (cheap convexity)
            plans.append(dict(kind="ALPHA_CALL", ticker=d.ticker, call=True, delta=0.6, dte=(30, 90),
                              budget=None, mu=d.mu_alpha + 0.04, rv=d.idio_vol, iv_max=1.0,
                              reason=f"top composite {d.composite:.2f} ({d.sector})"))
        for p in plans:
            p["sleeve"] = self._sleeve_of(p["ticker"])
        return plans

    def _option_premium(self, sleeve=None):
        """Market value of open long options as a fraction of NAV — total (incl. hedges) or per credit sleeve."""
        tot = 0.0
        for sym, m in self.option_book.items():
            if m["kind"] == "SPREAD_SHORT" or self.portfolio[sym].quantity == 0:
                continue
            if sleeve is None or (m.get("sleeve") == sleeve and m["kind"] != "HEDGE"):
                tot += max(self.portfolio[sym].holdings_value, 0.0)
        return tot / self._nav()

    def options_select(self):
        if self.is_warming_up or not self.option_plans:
            return
        self.option_candidates = {}
        for p in self.option_plans:
            und = self._symbol(p["ticker"])
            if und is not None and not self.securities.contains_key(und):
                try:                                       # option plans on names not yet subscribed
                    und = self.add_equity(p["ticker"], self.res).symbol
                    self.sym_of[p["ticker"]] = und
                except Exception:
                    und = None
            if und is None or not self.securities.contains_key(und):
                continue
            if self._ensure_raw(und):
                continue                                   # just switched to raw prices: pick strikes next window
            S = self.securities[und].price
            try:
                chain = self.option_chain_provider.get_option_contract_list(und, self.time)
            except Exception:
                chain = []
            lo, hi = p["dte"]
            picks = [c for c in chain if lo <= (c.id.date.date() - self.time.date()).days <= hi
                     and (c.id.option_right == OptionRight.CALL) == p["call"] and 0.7 * S <= c.id.strike_price <= 1.3 * S]
            picks = sorted(picks, key=lambda c: abs(c.id.strike_price - S))[:12]
            for c in picks:
                if not self.securities.contains_key(c):
                    self.add_option_contract(c, Resolution.MINUTE)
            self.option_candidates[(p["ticker"], p["kind"], p["call"])] = picks

    def options_execute(self):
        if self.is_warming_up or not self.option_plans or self.breaker_on or not self._feed_check():
            return
        nav, done = self._nav(), []
        for p in self.option_plans:
            if any(m["ticker"] == p["ticker"] and m["kind"] == p["kind"] and m.get("call") == p["call"]
                   and self.portfolio[s].quantity != 0 for s, m in self.option_book.items()):
                done.append(p)                             # idempotent: one open position per (ticker, kind, side)
                continue
            cands = self.option_candidates.get((p["ticker"], p["kind"], p["call"]), [])
            und = self._symbol(p["ticker"])
            if not cands or und is None:
                continue
            if not self.securities.contains_key(und):
                continue
            S = self.securities[und].price
            quotes = []
            for c in cands:
                if not self.securities.contains_key(c):
                    continue
                s = self.securities[c]
                quotes.append(op.Quote(c, c.id.strike_price, (c.id.date.date() - self.time.date()).days, p["call"],
                                       s.bid_price, s.ask_price, getattr(s, "open_interest", None) or None))
            pick = op.select(quotes, S, p["call"], p["delta"], *p["dte"])
            if pick is None:
                continue
            q = pick.quote
            T = q.dte / 365
            rv = p.get("rv") or name_vol(self._closes(und, 64), 63) or pick.iv
            if p.get("iv_max") is not None and pick.iv > p["iv_max"] * rv:
                continue                                   # alpha / rotation calls only when implied vol is not rich
            prob, pnl = op.real_world(S, q.strike, T, max(rv, 0.05), q.mid, p["mu"], p["call"])
            if p.get("straddle") or p["kind"] == "HEDGE":
                prob, pnl = max(prob, op.P_PROFIT_LONG), max(pnl, 1e-6)   # hedges / vol trades exempt from P(profit)
            budget = p["budget"] if p["budget"] is not None else op.kelly_option_fraction(prob, pnl, q.mid)
            n = op.contracts(budget, nav, q.mid, prob, pnl)
            room = dm.OPTIONS_TOTAL - self._option_premium()       # premium ≤ 10% of NAV (hedges included)
            if p["kind"] != "HEDGE":                                # per credit sleeve: IG 4 · HY 4 · distressed 2
                sleeve = p.get("sleeve", "IG")
                room = min(room, dm.OPTION_PREMIUM.get(sleeve, dm.OPTION_PREMIUM["IG"]) - self._option_premium(sleeve))
            n = min(n, int(max(room, 0.0) * nav // max(q.mid * op.MULTIPLIER, 1e-9)))
            if n <= 0:
                continue
            half = (q.ask - q.bid) / 2
            limit = round(q.mid + hb.OPTION_LIMIT_FRACTION * half, 2)
            self.limit_order(q.symbol, n, limit, tag=f"{p['kind']} {p['ticker']} Δ{pick.delta:.2f} IV {pick.iv:.0%} "
                                                      f"P(profit) {prob:.0%}: {p['reason']}")
            self.option_book[q.symbol] = dict(kind=p["kind"], ticker=p["ticker"], entry_dte=q.dte, cost=q.mid * n * 100,
                                              call=p["call"], sleeve=p.get("sleeve", "IG"))
            if p.get("short_delta"):
                short = op.select([x for x in quotes if x.dte == q.dte and x.strike > q.strike], S, True,
                                  p["short_delta"], q.dte, q.dte)
                if short is not None:
                    self.limit_order(short.quote.symbol, -n, round(short.quote.mid - hb.OPTION_LIMIT_FRACTION *
                                                                    (short.quote.ask - short.quote.bid) / 2, 2),
                                     tag=f"LEAPS_SPREAD short leg {p['ticker']}")
                    self.option_book[short.quote.symbol] = dict(kind="SPREAD_SHORT", ticker=p["ticker"],
                                                                entry_dte=q.dte, cost=0.0)
            done.append(p)
        self.option_plans = [p for p in self.option_plans if p not in done]

    def _option_exits(self, only_expiry=False):
        if not self.feed_ok and not only_expiry:
            return                                         # under a feed halt only expiry closes run
        ev = self.events()
        for sym, meta in list(self.option_book.items()):
            h = self.portfolio[sym]
            if h.quantity == 0:
                if (sym.id.date.date() - self.time.date()).days < 0 or not self.transactions.get_open_orders(sym):
                    self.option_book.pop(sym, None)
                continue
            dte = (sym.id.date.date() - self.time.date()).days
            if only_expiry:
                why = f"expiry in {dte}d" if dte <= hb.EXPIRY_CLOSE_DTE else None
            else:
                why = pr.option_exit(dte, meta["entry_dte"], abs(h.holdings_value), meta["cost"], meta["kind"],
                                     ev["days"].get(meta["ticker"]))
            if why and meta["kind"] != "SPREAD_SHORT":
                self.liquidate(sym, tag=f"OPTION EXIT {meta['kind']} {meta['ticker']}: {why}")
                for s2, m2 in list(self.option_book.items()):           # close the spread's short leg with it
                    if m2["kind"] == "SPREAD_SHORT" and m2["ticker"] == meta["ticker"] and s2.id.date == sym.id.date:
                        self.liquidate(s2, tag="close spread short leg")
                        self.option_book.pop(s2, None)
                self.option_book.pop(sym, None)

