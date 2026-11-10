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
import shorts as sh
from rotation import realised_vol as name_vol
from sectors import SECTOR_ETF_FALLBACK


class OptionsDeskMixin:   # plain holder of methods (not a base class — see main.py)
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
        approved = sorted((d for d in self.decisions.values() if d.approved and d.ticker not in self.protect
                           and d.sleeve != "SHORT" and not getattr(d, "filled", False)),
                          key=lambda d: -d.composite)             # calls: entry-gate longs, not in a cooldown
        bears = sorted((d for d in self.decisions.values() if d.approved and d.sleeve == "SHORT"
                        and d.target < 0 and d.ticker not in self.protect), key=lambda d: -d.composite)
        by_sector = {}
        for d in approved:
            by_sector.setdefault(d.sector, []).append(d)
        for a in plan.actions:
            if a.kind == "ROTATE":                       # a theme crashed: calls on the best names of the destination sectors
                for d in by_sector.get(a.ticker, [])[:2]:
                    plans.append(dict(kind="ROTATION_CALL", ticker=d.ticker, call=True, delta=0.5, dte=(45, 120),
                                      budget=None, mu=d.mu_alpha + 0.04, iv_max=1.25,
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
                              budget=None, mu=d.mu_alpha + 0.04, iv_max=1.0,
                              reason=f"top composite {d.composite:.2f} ({d.sector})"))
        for d in bears[:sh.BEAR_PUTS]:                    # bearish turn: puts on the strongest short ideas
            plans.append(dict(kind="BEAR_PUT", ticker=d.ticker, call=False, delta=0.4, dte=(45, 120),
                              budget=None, mu=-(d.mu_alpha + 0.04), iv_max=1.25,
                              reason=f"short composite {d.composite:.2f} ({d.sector})"))
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
                    und = self.add_equity(p["ticker"], self.res, data_normalization_mode=DataNormalizationMode.RAW).symbol
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
            added = []
            for c in picks:
                if self.securities.contains_key(c):
                    added.append(c)
                    continue
                try:
                    self.add_option_contract(c, Resolution.MINUTE)
                    added.append(c)
                except Exception as exc:                   # never let one refused contract stop the algorithm
                    self.debug(f"option contract refused for {p['ticker']}: {exc}")
                    break
            self.option_candidates[(p["ticker"], p["kind"], p["call"])] = added

    def options_execute(self):
        if self.is_warming_up or not self.option_plans or self.breaker_on or not self._feed_check():
            return
        nav, done = self._nav(), []
        for p in self.option_plans:
            if any(m["ticker"] == p["ticker"] and m["kind"] == p["kind"] and m.get("call") == p["call"]
                   and (self.portfolio[s].quantity != 0 or self._working(s)) for s, m in self.option_book.items()):
                done.append(p)                             # idempotent: one open position OR working order per
                                                           # (ticker, kind, side) — never stack entries
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
                self._opt_reject("no tradable contract (spread / open interest / quote)")
                continue
            q = pick.quote
            T = q.dte / 365
            rv = p.get("rv") or name_vol(self._closes(und, 64), 63) or pick.iv   # TOTAL realised vol (implied vol is
                                                                               # total too; never compare with idio)
            if p.get("iv_max") is not None and pick.iv > p["iv_max"] * rv:
                self._opt_reject(f"implied vol rich (> {p['iv_max']:g} x realised)")
                continue                                   # alpha / rotation calls only when implied vol is not rich
            # Black-Scholes values the contract at realised vol; Monte Carlo (4,000 real-world paths) gives the
            # probability of profit and the expected P&L the gate and the Kelly size use
            bs = op.bs_price(S, q.strike, T, max(rv, 0.05), call=p["call"])
            mc = op.monte_carlo(S, q.strike, T, max(rv, 0.05), q.mid, p["mu"], p["call"])
            prob, pnl = mc["p_profit"], mc["exp_pnl"]
            exempt = bool(p.get("straddle")) or p["kind"] == "HEDGE"
            if exempt:
                prob, pnl = max(prob, op.P_PROFIT_LONG), max(pnl, 1e-6)   # hedges / vol trades exempt from P(profit)
            elif pnl <= 0 or prob < op.P_PROFIT_LONG:
                self._opt_reject(f"Monte Carlo gate (P(profit) < {op.P_PROFIT_LONG:.0%} or expected P&L ≤ 0)")
            budget = p["budget"] if p["budget"] is not None else op.kelly_option_fraction(prob, pnl, q.mid)
            n = op.contracts(budget, nav, q.mid, prob, pnl)
            room = dm.OPTIONS_TOTAL - self._option_premium()       # premium ≤ 10% of NAV (hedges included)
            if p["kind"] != "HEDGE":                                # per credit sleeve: IG 4 · HY 4 · distressed 2
                sleeve = p.get("sleeve", "IG")
                room = min(room, dm.OPTION_PREMIUM.get(sleeve, dm.OPTION_PREMIUM["IG"]) - self._option_premium(sleeve))
            n = min(n, int(max(room, 0.0) * nav // max(q.mid * op.MULTIPLIER, 1e-9)))
            if n <= 0:
                if pnl > 0 and prob >= op.P_PROFIT_LONG:
                    self._opt_reject("premium budget full / contract too large")
                continue
            rec = dict(time=str(self.time)[:16], kind=p["kind"], ticker=p["ticker"], right="C" if p["call"] else "P",
                       strike=q.strike, dte=q.dte, spot=round(S, 2), mid=round(q.mid, 2), bs_value=round(bs, 2),
                       iv=round(pick.iv, 3), rv=round(rv, 3), delta=round(pick.delta, 2), drift=round(p["mu"], 3),
                       mc_p_profit=round(mc["p_profit"], 3), mc_exp_pnl=round(mc["exp_pnl"], 2),
                       mc_p5=round(mc["p5"], 2), mc_p95=round(mc["p95"], 2), mc_total_loss=round(mc["p_total_loss"], 3),
                       paths=mc["paths"], contracts=n, premium_nav=round(n * q.mid * op.MULTIPLIER / nav, 4),
                       reason=p["reason"])
            self.option_log = (self.option_log + [rec])[-200:]
            self.debug(f"OPTION PICK {rec['kind']} {rec['ticker']} {q.strike:g}{rec['right']} {q.dte}d x{n} | spot "
                       f"{S:.2f} mid {q.mid:.2f} BS@RV {bs:.2f} IV {pick.iv:.0%} RV {rv:.0%} Δ{pick.delta:.2f} | MC "
                       f"{mc['paths']} paths: P(profit) {mc['p_profit']:.0%} E[P&L] {mc['exp_pnl']:+.2f} 5%/95% "
                       f"{mc['p5']:+.2f}/{mc['p95']:+.2f} P(lose all) {mc['p_total_loss']:.0%} | premium "
                       f"{rec['premium_nav']:.2%} NAV | {p['reason']}")
            self._stat("Option last", f"{rec['kind']} {rec['ticker']} {q.strike:g}{rec['right']} {q.dte}d MC "
                                      f"P {mc['p_profit']:.0%} E {mc['exp_pnl']:+.2f}")
            self.plot("Options", "MC P(profit) %", mc["p_profit"] * 100)
            self.plot("Options", "IV / RV", pick.iv / max(rv, 1e-6))
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

    def _opt_reject(self, why):
        self.opt_rejects[why] = self.opt_rejects.get(why, 0) + 1
        top = sorted(self.opt_rejects.items(), key=lambda x: -x[1])[:2]
        self._stat("Options", f"picked {len(self.option_log)} | rejected: " + "; ".join(f"{k[:22]} x{n}" for k, n in top))

    def _open_orders(self, sym):
        try:
            return list(self.transactions.get_open_orders(sym))
        except Exception:
            return []

    def _option_close(self, sym, dte, tag):
        """Close an option position with a limit order (option spreads are wide — never a raw market order):
        mid ∓ 0.3 × half-spread; at the bid / ask when ≤ 2 DTE; market only on expiry day or with no quote.
        Re-priced on every risk check until flat."""
        h = self.portfolio[sym]
        if h.quantity == 0:
            return
        self._cancel_working(sym)
        s = self.securities[sym]
        bid, ask = getattr(s, "bid_price", 0) or 0, getattr(s, "ask_price", 0) or 0
        if dte <= 0 or bid <= 0 or ask < bid:
            if dte <= hb.EXPIRY_CLOSE_DTE:                 # must be out before expiry even without a quote
                self.liquidate(sym, tag=tag)
            return
        mid, half, sell = (bid + ask) / 2, (ask - bid) / 2, h.quantity > 0
        if dte <= hb.EXPIRY_CLOSE_DTE:
            px = bid if sell else ask
        else:
            px = mid - hb.OPTION_LIMIT_FRACTION * half if sell else mid + hb.OPTION_LIMIT_FRACTION * half
        self.limit_order(sym, -h.quantity, max(round(px, 2), 0.01), tag=tag)

    def _option_exits(self, only_expiry=False):
        if not self.feed_ok and not only_expiry:
            return                                         # under a feed halt only expiry closes run
        ev = self.events()
        days = lambda s: (s.id.date.date() - self.time.date()).days
        for sym, meta in list(self.option_book.items()):
            h = self.portfolio[sym]
            dte = days(sym)
            if h.quantity == 0:
                working = self._open_orders(sym)
                if working and only_expiry:                # 15:30: an unfilled entry never rests overnight
                    self._cancel_working(sym)
                    working = []
                if dte < 0 or not working:
                    self.option_book.pop(sym, None)
                continue
            if meta["kind"] == "SPREAD_SHORT":
                continue
            if only_expiry:
                why = f"expiry in {dte}d" if dte <= hb.EXPIRY_CLOSE_DTE else None
            else:
                why = pr.option_exit(dte, meta["entry_dte"], abs(h.holdings_value), meta["cost"], meta["kind"],
                                     ev["days"].get(meta["ticker"]))
            if why:                                        # the book entry stays until the position is flat
                self._option_close(sym, dte, f"OPTION EXIT {meta['kind']} {meta['ticker']}: {why}")
                for s2, m2 in list(self.option_book.items()):           # close the spread's short leg with it
                    if m2["kind"] == "SPREAD_SHORT" and m2["ticker"] == meta["ticker"] and s2.id.date == sym.id.date:
                        self._option_close(s2, dte, "close spread short leg")
        for s2, m2 in list(self.option_book.items()):      # a short call must always be covered by its long leg
            short = -self.portfolio[s2].quantity
            if m2["kind"] != "SPREAD_SHORT" or short <= 0:
                continue
            legs = [s for s, m in self.option_book.items() if m["kind"] == "LEAPS_SPREAD"
                    and m["ticker"] == m2["ticker"] and s.id.date == s2.id.date]
            if any(self._open_orders(s) for s in legs) or self._open_orders(s2):
                continue                                   # long leg still being entered / exit already working
            covered = sum(max(self.portfolio[s].quantity, 0) for s in legs)
            if short > covered:
                self._cancel_working(s2)
                ask = getattr(self.securities[s2], "ask_price", 0) or 0
                if ask > 0:
                    self.limit_order(s2, int(short - covered), round(ask, 2), tag=f"cover short leg {m2['ticker']}")
                else:
                    self.market_order(s2, int(short - covered), tag=f"cover short leg {m2['ticker']}")
