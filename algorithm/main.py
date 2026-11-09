# region imports
from AlgorithmImports import *
# endregion

import json
import math
from datetime import timedelta

from corridor import target_leverage, should_rebalance, annualised_drift, realised_vol, R_LOW, R_HIGH, RISK_FREE, MAX_GROSS_LEVERAGE
from risk_budget import (beta_cap, hard_stops, daily_var, check_alpha_trade, AlphaPosition, momentum_confirms,
                         alpha_var, REGIME_VAR_CEILING, MAX_FUTURES_LEVERAGE)
from regime import RegimeModel, REGIMES
from forward_rm import estimate as forward_rm
from fundamentals import top_down, analyze, cap_structure_map
from universe import (SP1500_ETFS, EXPECTED_MEMBERS, SECTOR_TYPE, extract, subsector_for, passes_liquidity,
                      passes_fallback, shortlist, EpsHistory, PriceBuffer, SHORTLIST_N)
from profiles import THEMES as THEME_PROFILES, theme_members, watch_candidate
from rotation import (DEFENSE_SIGNAL_FALLBACK, ESCALATION_SLEEVE, THEME_SUBSECTOR, build_plan, escalation_score,
                      theme_index, theme_state, realised_vol as name_vol)
from macro_regime import factor_states, diagnose, market_bias, stock_bond_regime
from sector_rotation import evaluate, scenarios_from_json, summary
from market_map import CYCLICAL, SECULAR, DEFENSIVE, SUBSECTORS
from sectors import SECTORS, sector_scores, sector_budgets, rotation_sectors, SECTOR_ETF_FALLBACK
import schedule as hb
import decision as dm
import options as op
import protection as pr
import alpha as al
import feed_guard as fg

# ETF buckets (allocation_engine: TLTW/cash-flow 15%, FI/macro 5%) — the only ETFs held as positions
CASHFLOW_ETFS = {"income": ["TLTW", "JEPI", "JEPQ", "DIVO"], "commodities": ["GLD", "DBC"]}
FI_ETFS = {"TREASURIES": ["TLT", "IEF"], "OTHER": ["SHY"]}
ALL_ETFS = {t for v in list(CASHFLOW_ETFS.values()) + list(FI_ETFS.values()) for t in v}
MIN_TRADE = 0.005
STOP_COOLDOWN_DAYS = 5          # after a −8% close-out, stay flat 5 days, then re-arm at the new NAV


class MetadronCorridor(QCAlgorithm):
    """Metadron Capital — full LEAN build (steps 1–7).

    Heartbeat (schedule.py, aligned to intraday liquidity):
        09:31 risk-open · 09:50 research · 10:00 core (MES overlay + 40% of equity deltas)
        10:15/10:45 and 14:15/14:45 options windows · every 30 min risk · 15:30 expiry + beta true-up
        15:45 closing auction (remaining equity deltas, market-on-close)

    Research (09:50): HMM regime → forward Rm → corridor leverage L (full Kelly, 4.5x, regime ceiling, burst)
        → macro question tree + scenarios → sub-sector edges → 11 GICS sector scores + budgets (bullish / bearish
        reasoning; crashed themes marked down) → theme plan (cut crashed names, destination sectors)
        → decision matrix on the fundamentals shortlist (4 gates, spread, P(profit), full Kelly, sleeves,
        2.5% alpha VaR) → MES = L − Σ wᵢβᵢ (CVaR-capped, breaker-halved) → option plans.
    Parameters: kelly_fraction (1.0), beta_instrument (MES|SPY), rm_lookback, rm_smoothing, rotation_trading,
        universe_mode, fundamental_refresh_days, shortlist_size, trade_equities, trade_options, data_resolution.
    """

    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        p = lambda k, d: self.get_parameter(k, d)
        self.k = float(p("kelly_fraction", 1.0))
        dm.KELLY_FRACTION = self.k
        # PHASE 2: the MES futures beta overlay (corridor L − sleeve beta, ≤ 2.0x NAV) is built but OFF by default
        # (owner, Nov 2026). beta_instrument = NONE (phase 1: no overlay, corridor computed + reported only),
        # MES (phase 2: futures overlay), SPY (≤ 2x ETF overlay).
        self.instrument = p("beta_instrument", "NONE").upper()
        self.rm_lookback = int(p("rm_lookback", 252))
        self.rm_smoothing = float(p("rm_smoothing", 0.10))
        self.rotation_trading = str(p("rotation_trading", "true")).lower() == "true"
        self.universe_mode = str(p("universe_mode", "sp1500")).lower()
        self.refresh_days = int(p("fundamental_refresh_days", 5))
        self.shortlist_n = int(p("shortlist_size", SHORTLIST_N))
        self.trade_equities = str(p("trade_equities", "true")).lower() == "true"
        self.trade_options = str(p("trade_options", "true")).lower() == "true"
        self.res = Resolution.MINUTE if str(p("data_resolution", "minute")).lower() == "minute" else Resolution.DAILY

        spy = self.add_equity("SPY", self.res)
        spy.set_leverage(2.0)
        self.spy = spy.symbol
        self.set_benchmark(self.spy)
        if self.instrument == "MES":
            self.future = self.add_future(Futures.Indices.MICRO_SP_500_E_MINI, self.res,
                                          data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
                                          data_mapping_mode=DataMappingMode.OPEN_INTEREST, contract_depth_offset=0)
            self.future.set_leverage(MAX_GROSS_LEVERAGE + 0.5)
        else:
            self.future = None
        self.ref_index = None                     # phase 1 cross-feed reference for the feed guard: S&P 500 index
        if self.future is None:
            try:
                self.ref_index = self.add_index("SPX", self.res).symbol
            except Exception as exc:
                self.debug(f"SPX index unavailable ({exc}) — feed guard runs staleness checks only")

        tickers = sorted(set(ESCALATION_SLEEVE["oil_option_underlyings"]) | ALL_ETFS
                         | {"HYG", "LQD", DEFENSE_SIGNAL_FALLBACK})
        self.rot_symbols = {}
        for t in tickers:
            try:
                sec = self.add_equity(t, self.res)
                sec.set_leverage(2.0)
                self.rot_symbols[t] = sec.symbol
            except Exception as exc:
                self.debug(f"cannot add {t}: {exc}")
        try:
            self.vix = self.add_index("VIX", Resolution.DAILY).symbol
        except Exception:
            self.vix = None

        self.rm_ewma, self.regime_model = None, RegimeModel()
        self.last_views, self.last_plan_sig, self.sector_score, self.sector_budget = {}, None, {}, {}
        self.theme_members, self.defense_names, self.watch, self.last_credit_stress = {}, [], set(), False
        self.sym_of = {}
        self._init_universe()
        self.wf, self.quality, self.scorecard = al.WalkForward(), al.QualityEnsemble(), al.Scorecard()
        self.protector = pr.Protector()
        self.targets, self.decisions, self.option_plans, self.option_candidates = {}, {}, [], {}
        self.option_book = {}             # Symbol → {kind, underlying, entry_dte, cost, ticker}
        self.hedged = set()
        self.current_leverage, self.beta_target, self.state = 0.0, 0.0, None
        self.peak_nav = self.portfolio.total_portfolio_value
        self.day_open_nav, self.breaker_on, self.week_step = None, False, 0
        self.feed, self.stop_tickets, self.feed_ok = fg.FeedGuard(daily=self.res == Resolution.DAILY), {}, True
        try:
            self.default_order_properties.time_in_force = TimeInForce.GOOD_TIL_CANCELED   # resting stops survive days
        except Exception:
            pass
        self.set_warm_up(self.rm_lookback + 5, Resolution.DAILY)

        on = lambda m, fn: self.schedule.on(self.date_rules.every_day(self.spy),
                                            self.time_rules.after_market_open(self.spy, m), fn)
        on(hb.RISK_OPEN, self.risk_open)
        on(hb.RESEARCH, self.research)
        on(hb.CORE, self.core)
        for m in hb.OPTIONS_SELECT:
            on(m, self.options_select)
        for m in hb.OPTIONS_EXECUTE:
            on(m, self.options_execute)
        for m in hb.risk_minutes():
            on(m, self.risk_check)
        on(hb.EXPIRY, self.expiry)
        on(hb.CLOSE, self.close_auction)

    # ------------------------------------------------------------------ universe (step 3)
    def _init_universe(self):
        self.members = {}                     # etf → set(ticker)
        self.eps_hist, self.px_buf = EpsHistory(), PriceBuffer()
        self.scores, self.td, self.health = {}, None, {}
        self.selected, self.last_score_time = [], None
        self.subsector_overrides = {}
        self.universe_settings.resolution = self.res
        if self.universe_mode == "sp1500":
            for etf in SP1500_ETFS:
                try:
                    self.add_universe(self.universe.etf(etf, self.universe_settings,
                                                        lambda c, e=etf: self._etf_members(e, c)))
                except Exception as exc:
                    self.debug(f"universe: ETF constituents {etf} unavailable ({exc}) — fallback filter")
                    self.universe_mode = "fundamental"
        self.add_universe(self._fundamental_selection)

    def _etf_members(self, etf, constituents):
        self.members[etf] = {c.symbol.value for c in constituents}      # tickers (same-day feeds)
        return []                              # membership only; subscriptions come from the shortlist

    def _fundamental_selection(self, fundamental):
        members = set().union(*self.members.values()) if self.members else set()
        theme_subs = {sub for spec in THEME_PROFILES.values() for sub in spec["subsectors"]}
        watch_new = set()
        days = (self.time - self.last_score_time).days if self.last_score_time is not None else 999
        rescore = days >= self.refresh_days
        rows, missing_n, expected = {}, 0, 0
        for f in fundamental:
            sym = f.symbol
            t = sym.value
            in_universe = t in members if self.universe_mode == "sp1500" else bool(getattr(f, "has_fundamental_data", False))
            off_index = False
            if not in_universe:                       # off-index theme watch: profile-matched names only
                if not rescore:
                    if t in self.watch:
                        self.px_buf.update(t, getattr(f, "price", None))
                    continue
                ac = getattr(f, "asset_classification", None)
                if not watch_candidate(getattr(ac, "morningstar_sector_code", None), getattr(f, "market_cap", 0),
                                       getattr(f, "dollar_volume", 0)):
                    continue
                d, missing = extract(f)
                if missing or subsector_for(t, d, self.subsector_overrides) not in theme_subs:
                    continue
                watch_new.add(t)
                off_index = True
            self.px_buf.update(t, getattr(f, "price", None))
            if not rescore:
                continue
            if not off_index:
                d, missing = extract(f)
            if self.universe_mode != "sp1500" and not off_index and not passes_fallback(d):
                continue
            expected += 1
            if missing:
                missing_n += 1
                continue
            self.eps_hist.update(t, d.get("eps_ttm"))
            rows[t] = (sym, d)
        if not rescore:
            return self.selected
        if self.subsector_overrides == {}:
            raw = self._store_read("metadron/subsector_map.json")
            try:
                self.subsector_overrides = json.loads(raw) if raw else {"_": "_"}
            except Exception:
                self.subsector_overrides = {"_": "_"}
        macro = self.macro_inputs()
        vix = None
        if getattr(self, "vix", None) is not None:
            v = self._closes(self.vix, 3)
            vix = v[-1] if v else None
        self.td = top_down([d for _, d in rows.values()], treasury_10y=macro.get("treasury_10y", 0.042), vix=vix)
        scores = {}
        for t, (sym, d) in rows.items():
            code = d.get("sector_code")
            try:
                stype = SECTOR_TYPE.get(int(code), "") if code is not None else ""
            except (TypeError, ValueError):
                stype = ""
            sc = analyze(t, d, self.td, equity_vol=self.px_buf.vol(t), eps_history=self.eps_hist.annual(t),
                         sector_type=stype)
            sc.subsector = subsector_for(t, d, self.subsector_overrides)
            if t in watch_new or (passes_liquidity(d) or self.universe_mode != "sp1500"):
                scores[t] = sc
        self.scores = scores
        self.watch = watch_new
        caps = {t: d.get("market_cap", 0.0) for t, (_, d) in rows.items()}
        self.theme_members = theme_members({t: sc.subsector for t, sc in scores.items()}, caps)
        self.defense_names = sorted((t for t, sc in scores.items() if sc.subsector == "DEFENSE"),
                                    key=lambda t: -caps.get(t, 0.0))[:8]
        extra = watch_new
        held = [kv.key.value for kv in self.portfolio if kv.value.invested and kv.value.type == SecurityType.EQUITY]
        sl = shortlist({t: s for t, s in scores.items() if t not in extra}, self.last_views, self.shortlist_n, held,
                       self.sector_budget or None)
        sym_of = {t: sym for t, (sym, _) in rows.items()}
        self.sym_of.update(sym_of)
        self.selected = [sym_of[t] for t in sl.names if t in sym_of]
        self.last_score_time = self.time
        self.health = {"members": len(members) if self.universe_mode == "sp1500" else expected,
                       "expected": EXPECTED_MEMBERS if self.universe_mode == "sp1500" else expected,
                       "scored": len(scores), "missing_fields": missing_n,
                       "gate_pass": sum(1 for s in scores.values() if s.gate_pass),
                       "shortlist": len(sl.names), "vetoed": len(sl.vetoed)}
        self._publish_universe(sl)
        return self.selected

    def _publish_universe(self, sl):
        h, td = self.health, self.td
        for k in ("members", "scored", "missing_fields", "gate_pass", "shortlist"):
            self.plot("Data health", k, h.get(k, 0))
        self.plot("Valuation", "earnings yield %", td.earnings_yield * 100)
        self.plot("Valuation", "ERP %", td.erp * 100)
        self.plot("Valuation", "market P/E", min(td.market_pe, 60))
        self.plot("Valuation", "max investment P/E", td.max_pe)
        if h.get("members", 0) < 0.9 * h.get("expected", 1) or h.get("missing_fields", 0) > 0.2 * max(h.get("members", 1), 1):
            self.debug(f"DATA-HEALTH alert {h}")
        top = sorted(sl.rank.items(), key=lambda x: -x[1])[:100]
        report = {"time": str(self.time), "mode": self.universe_mode, "health": h, "top_down": td.__dict__,
                  "shortlist": sl.by_sleeve, "vetoed": sl.vetoed[:50],
                  "scores": {t: self.scores[t].to_dict() for t, _ in top},
                  "themes": {th: {"reasoning": THEME_PROFILES[th]["reasoning"], "members": names}
                             for th, names in self.theme_members.items()},
                  "watch": {t: self.scores[t].to_dict() for names in self.theme_members.values() for t in names}}
        try:
            self.object_store.save("metadron/universe_scores.json", json.dumps(report, default=str))
        except Exception:
            pass
        self.debug(f"UNIVERSE {h} | EY {td.earnings_yield:.2%} ERP {td.erp:+.2%} {td.regime} | "
                   + " ".join(f"{k}:{','.join(v[:5])}" for k, v in sl.by_sleeve.items()))

    def subsector_fundamentals(self) -> dict:
        """{subsector: −1..1} = 2 × (mean gate score − 0.5) over scored names → sector_rotation input."""
        agg = {}
        for s in self.scores.values():
            agg.setdefault(s.subsector, []).append(s.gate_score)
        return {k: max(-1.0, min(1.0, 2 * (sum(v) / len(v) - 0.5))) for k, v in agg.items() if len(v) >= 3}

    def macro_inputs(self) -> dict:
        """Numeric readings written by the macro agent (FRED, GMTF, Kronos) — {} if absent/stale."""
        raw = self._store_read("metadron/macro_inputs.json")
        if not raw:
            return {}
        try:
            return {k: float(v) for k, v in json.loads(raw).items() if isinstance(v, (int, float))}
        except Exception as exc:
            self.debug(f"macro_inputs.json unreadable: {exc}")
            return {}

    def _set_beta_leverage(self, leverage: float, tag: str):
        nav = self.portfolio.total_portfolio_value
        if self.instrument == "NONE":
            return                                     # phase 1: no beta overlay (MES = phase 2)
        if self.future is None:
            self.set_holdings(self.spy, min(leverage, 2.0), tag=tag)
            return
        contract = self.future.mapped
        if contract is None:
            return
        sec = self.securities[contract]
        price = sec.price
        if price <= 0:
            return
        mult = sec.symbol_properties.contract_multiplier
        target_qty = int(round(leverage * nav / (price * mult)))
        current = int(self.portfolio[contract].quantity)
        delta = target_qty - current
        if delta == 0:
            return
        if not self.feed_ok and abs(target_qty) > abs(current):
            self.debug(f"FEED HALT: MES increase blocked ({current} → {target_qty})")
            return
        self._limit(contract, delta, "FUTURE", tag)

    def on_symbol_changed_events(self, symbol_changed_events):
        """Roll the MES position into the newly mapped contract (phase 2).

        LEAN also raises this event for equity ticker changes (renames / mergers, e.g. DOW in 2017); those need
        no action — LEAN carries the holding over — and their old ticker is a plain string that cannot be used
        to index the portfolio. So only the continuous future is handled, with tickers resolved to Symbols."""
        if self.future is None:
            return
        for symbol, changed in symbol_changed_events.items():
            if symbol != self.future.symbol:
                continue
            try:
                old, new = self.symbol(changed.old_symbol), self.symbol(changed.new_symbol)
            except Exception as exc:
                self.debug(f"roll: cannot resolve {changed.old_symbol} → {changed.new_symbol}: {exc}")
                continue
            qty = self.portfolio[old].quantity if self.portfolio.contains_key(old) else 0
            self._drop_stop(old)                           # no orphan stop on the expiring contract
            if qty != 0:
                self.liquidate(old, tag="roll")
                self.market_order(new, qty, tag="roll")


    # ------------------------------------------------------------------ data helpers
    def _bars(self, symbol, n=300):
        """(closes, volumes) of daily bars."""
        try:
            h = self.history(symbol, n, Resolution.DAILY)
            if h is None or h.empty or "close" not in h.columns:
                return [], []
            vol = [float(x) for x in h["volume"].values] if "volume" in h.columns else []
            return [float(x) for x in h["close"].values], vol
        except Exception:
            return [], []

    def _closes(self, symbol, n=260):
        return self._bars(symbol, n)[0] if symbol is not None else []

    def _store_read(self, key):
        try:
            if self.object_store.contains_key(key):
                return self.object_store.read(key)
        except Exception:
            pass
        return None

    def _store_json(self, key):
        raw = self._store_read(key)
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def _nav(self):
        return max(self.portfolio.total_portfolio_value, 1.0)

    def _weight(self, symbol):
        return self.portfolio[symbol].holdings_value / self._nav() if symbol is not None else 0.0

    def _symbol(self, ticker):
        return self.rot_symbols.get(ticker) or self.sym_of.get(ticker)

    def events(self) -> dict:
        """events.json (events agent): days_to_event aged by the days since as_of; penalties; escalation flag."""
        ev = self._store_json("metadron/events.json")
        out = {"days": {}, "penalty": {}, "flags": {}, "escalation_flag": ev.get("escalation_flag")}
        try:
            age = (self.time.date() - __import__("datetime").date.fromisoformat(ev.get("as_of", ""))).days
        except Exception:
            age = 0
        for t, e in (ev.get("tickers") or {}).items():
            if e.get("days_to_event") is not None:
                out["days"][t] = int(e["days_to_event"]) - age
            out["penalty"][t] = float(e.get("penalty", 0.0) or 0.0)
            out["flags"][t] = e.get("flags", [])
        return out

    # ------------------------------------------------------------------ feed guard (feed_guard.py)
    def _minutes(self, t):
        return t.timestamp() / 60.0 if t is not None and hasattr(t, "timestamp") else None

    def _last_bar(self, sym):
        try:
            d = self.securities[sym].get_last_data()
            return self._minutes(d.end_time) if d is not None else None
        except Exception:
            return None

    def _feed_check(self) -> bool:
        """Staleness + SPY/MES cross-feed + jump checks; flips self.feed_ok and logs transitions."""
        keys = {"SPY": self.spy}
        mes_sym = self.future.mapped if self.future is not None else self.ref_index
        ref_name = "MES" if self.future is not None else "SPX"
        if mes_sym is not None:
            keys[ref_name] = mes_sym
        for kv in self.portfolio:
            if kv.value.invested and kv.value.type == SecurityType.EQUITY:
                keys[kv.key.value] = kv.key
        last = {k: self._last_bar(s) for k, s in keys.items()}
        spy_px = self.securities[self.spy].price
        mes_px = self.securities[mes_sym].price if mes_sym is not None else None
        was = self.feed_ok
        st = self.feed.check(self._minutes(self.time), last, spy_px, mes_px, ref_name)
        self.feed_ok = st.ok
        if was and not st.ok:
            self.debug("FEED HALT: " + "; ".join(st.reasons[:5]) + " — no new entries, resting stops stay active")
        elif not was and st.ok:
            self.debug("FEED OK: halt cleared")
        self.plot("Feed", "ok", 1.0 if st.ok else 0.0)
        return st.ok

    def _drop_stop(self, sym):
        """Cancel a name's resting stop before trading it (re-placed by _sync_stops once no order is working),
        so a stop and a sell can never both execute and flip the position short."""
        m = self.stop_tickets.pop(sym, None)
        if m is not None:
            try:
                m["ticket"].cancel()
            except Exception:
                pass

    def _working(self, sym) -> bool:
        keep = {m["ticket"].order_id for m in self.stop_tickets.values()}
        try:
            return any(o.id not in keep for o in self.transactions.get_open_orders(sym))
        except Exception:
            return False

    def _cancel_working(self, sym):
        """Cancel working (non-stop) orders for a symbol so cancel-and-replace never stacks orders."""
        keep = {m["ticket"].order_id for m in self.stop_tickets.values()}
        try:
            for o in self.transactions.get_open_orders(sym):
                if o.id not in keep:
                    self.transactions.cancel_order(o.id)
        except Exception:
            pass

    def _limit(self, sym, qty, kind, tag):
        """Collared limit order around the last trusted price (never a raw market order)."""
        px = self.securities[sym].price
        if px <= 0 or qty == 0:
            return
        self._cancel_working(sym)
        self._drop_stop(sym)
        self.limit_order(sym, qty, fg.collar(px, qty > 0, kind), tag=tag)

    def _sync_stops(self):
        """Resting broker-side stops: long stocks at peak −10%, MES overlay at the ~8%-NAV adverse move."""
        want = {}
        for kv in self.portfolio:
            sym, h = kv.key, kv.value
            if not h.invested:
                continue
            if h.type == SecurityType.EQUITY and h.quantity > 0:
                st = self.protector.states.get(sym.value)
                peak = st.peak if st is not None else self.securities[sym].price
                want[sym] = (-int(h.quantity), fg.position_stop(peak))
            elif h.type == SecurityType.FUTURE and h.quantity != 0:
                stop = fg.futures_stop(self.securities[sym].price, self.current_leverage)
                if stop is not None:
                    want[sym] = (-int(h.quantity), stop)
        for sym in list(self.stop_tickets):
            m = self.stop_tickets[sym]
            done = str(getattr(m["ticket"], "status", "")).upper().endswith(("FILLED", "CANCELED", "INVALID"))
            w = want.get(sym)
            if done or w is None or w[0] != m["qty"] or abs(w[1] / m["stop"] - 1) > fg.STOP_REPLACE:
                if not done:
                    try:
                        m["ticket"].cancel()
                    except Exception:
                        pass
                self.stop_tickets.pop(sym, None)
        for sym, (qty, stop) in want.items():
            if sym not in self.stop_tickets and qty != 0 and not self._working(sym):
                t = self.stop_market_order(sym, qty, stop, tag=f"RESTING STOP {sym.value} @ {stop}")
                self.stop_tickets[sym] = {"ticket": t, "qty": qty, "stop": stop}

    # ------------------------------------------------------------------ 09:31 risk-open
    def risk_open(self):
        if self.is_warming_up:
            return
        self._feed_check()
        self._sync_stops()                                 # names traded in yesterday's closing auction
        self.day_open_nav, self.breaker_on = self._nav(), False
        self.risk_check()

    # ------------------------------------------------------------------ 09:50 research
    def research(self):
        if self.is_warming_up:
            return
        closes = self._closes(self.spy, self.rm_lookback + 1)
        if len(closes) < 64:
            self.debug("DATA-HEALTH no SPY history — research skipped")
            return
        drift = annualised_drift(closes)
        self.rm_ewma = drift if self.rm_ewma is None else self.rm_smoothing * drift + (1 - self.rm_smoothing) * self.rm_ewma
        sigma = realised_vol(closes[-64:])
        long = self._closes(self.spy, 800) or closes
        vix_hist = self._closes(self.vix, 800) if self.vix is not None else []
        hyg, lqd = self._closes(self.rot_symbols.get("HYG"), 800), self._closes(self.rot_symbols.get("LQD"), 800)
        reg = self.regime_model.classify(long, vix_hist or None, hyg or None, lqd or None)
        macro, ev = self.macro_inputs(), self.events()
        kron = al.kronos_alpha(self._store_read("metadron/kronos.json"))
        ey = macro.get("earnings_yield")
        if ey is None and self.td is not None and self.td.n_names >= 100:
            ey = self.td.earnings_yield
        fwd = forward_rm(self.rm_ewma, regime_probs=reg.probs, gmtf_score=macro.get("gmtf_score"), earnings_yield=ey,
                         inflation=macro.get("inflation_yoy"), kronos=macro.get("kronos_spx_annual", kron.get("SPY")))
        burst = reg.regime != "CRASH" and momentum_confirms(long)
        cap = beta_cap(reg.regime, sigma, rm=fwd.rm, burst=burst)
        state = target_leverage(fwd.rm, sigma, k=self.k, cap=cap)
        self.state = dict(rm=fwd.rm, sigma=sigma, regime=reg.regime, probs=reg.probs, method=reg.method, cap=cap,
                          burst=burst, zone=state.zone, L=state.leverage, components=fwd.components,
                          participation=state.participation, gamma=state.gamma, kelly=state.kelly)
        if reg.regime != getattr(self, "last_regime", None):
            self.last_regime = reg.regime
            self.debug(f"REGIME {reg.regime} ({reg.method}, p={reg.confidence:.2f}) Rm={fwd.rm:.2%} "
                       f"components={ {k: round(v, 4) for k, v in fwd.components.items()} }")

        # theme states → sector view → GICS sector scores → rotation plan
        prices = {t: c for t, s in self.rot_symbols.items() if len(c := self._closes(s)) > 21}
        theme_names = {t for v in self.theme_members.values() for t in v} | set(self.defense_names)
        for t in theme_names:                               # profiled theme members (S&P 1500 + off-index watch)
            if t not in prices and self.sym_of.get(t) is not None and len(c := self._closes(self.sym_of[t])) > 21:
                prices[t] = c
        vix = vix_hist[-1] if vix_hist else None
        esc = escalation_score(oil=prices.get("USO"), gold=prices.get("GLD"),
                               defense_idx=theme_index(prices, self.defense_names) or theme_index(prices, [DEFENSE_SIGNAL_FALLBACK]) or None,
                               spy=closes, vix=vix,
                               agent_flag=ev["escalation_flag"])
        states = [theme_state(n, tks, prices, closes, self.last_credit_stress) for n, tks in self.theme_members.items()]
        bias = self.run_sector_view(state.zone, prices, closes, esc, vix, states, macro)
        crashing = [THEME_SUBSECTOR[s.theme] for s in states if s.state != "NORMAL" and s.theme in THEME_SUBSECTOR]
        self.sector_score = sector_scores({k: v["edge"] for k, v in self.last_views.items()}, reg.regime, bias, esc, crashing)
        self.sector_budget = sector_budgets(self.sector_score)
        nav = self._nav()
        holdings = {t: self._weight(self.sym_of[t]) for t in theme_names if self.sym_of.get(t) is not None}
        plan = build_plan(state.zone, prices, closes, holdings, escalation=esc,
                          cap_structure=cap_structure_map(self.scores) if self.scores else None,
                          days_to_event=ev["days"], sector_scores=self.sector_score, themes=self.theme_members,
                          credit_stress=self.last_credit_stress)
        self._log_plan(plan, esc)

        # decision matrix on the shortlist
        self.caps = dm.sleeve_caps()
        self.decisions = self._decide(closes, reg.regime, ev, kron)
        targets = {t: d.target for t, d in self.decisions.items() if d.approved}
        if self.rotation_trading:                               # crashed-theme names are cut; the freed weight
            for a in plan.actions:                              # flows through the sector budgets into single names
                if a.kind == "CUT":
                    targets[a.ticker] = a.target_weight
        targets.update(self._etf_sleeve(prices, closes))
        targets = self._fit_alpha_budget(targets, prices)
        for t, s in list(self.sym_of.items()) + list(self.rot_symbols.items()):      # exit what is no longer wanted
            if t not in targets and self.portfolio[s].invested and self.portfolio[s].quantity > 0 \
                    and s.security_type == SecurityType.EQUITY:
                targets[t] = 0.0
        self.targets = targets

        # market beta overlay: corridor L minus the sleeves' beta, CVaR-capped, breaker-halved
        betas = {t: d.beta for t, d in self.decisions.items()}
        sleeve_beta = dm.book_beta({t: w for t, w in targets.items() if w > 0}, betas)
        book = [AlphaPosition(t, w, self.decisions[t].idio_vol if t in self.decisions else 0.25)
                for t, w in targets.items() if w > 0]
        beta_lev = state.leverage - sleeve_beta
        scale = pr.cvar_scale(state.leverage, sigma, alpha_var(book), REGIME_VAR_CEILING.get(reg.regime, 0.045))
        beta_lev *= scale
        nav_now = self._nav()
        self.peak_nav = max(self.peak_nav, nav_now)
        stop = hard_stops(nav_now, self.peak_nav, self.current_leverage)
        if stop.action == "CLOSE":
            beta_lev, self.targets = 0.0, {t: 0.0 for t in targets}
            self.stop_time = getattr(self, "stop_time", None) or self.time
            if (self.time - self.stop_time).days >= STOP_COOLDOWN_DAYS or nav_now >= self.peak_nav * 0.97:
                self.peak_nav, self.stop_time = nav_now, None      # re-arm after the cooldown (a flat book can't recover)
                self.debug(f"HARD STOP re-armed at NAV {nav_now:,.0f}")
        elif stop.action == "TRIM":
            beta_lev *= 0.5
            self.targets = {t: w * 0.5 for t, w in targets.items()}
        if self.breaker_on:
            beta_lev *= 0.5
        overlay = max(-MAX_FUTURES_LEVERAGE, min(MAX_FUTURES_LEVERAGE, beta_lev))   # owner cap: 2.0x NAV
        self.beta_target = 0.0 if self.instrument == "NONE" else overlay              # phase 1: not executed
        self.state.update(sleeve_beta=sleeve_beta, beta_overlay=self.beta_target, overlay_phase2=overlay,
                          beta_instrument=self.instrument, cvar_scale=scale, stop=stop.action,
                          escalation=esc, bias=bias)
        self.option_plans = self._option_plans(plan, ev, kron) if self.trade_options and stop.action != "CLOSE" else []
        self._weekly_learning(closes, kron)
        self._plots(drift, fwd, reg, state, cap, sigma)
        self._sync_stops()
        self._publish_state()

    def _decide(self, spy_closes, regime, ev, kron):
        weights = self.scorecard.weights()
        out = {}
        for t, s in self.scores.items():
            sym = self.sym_of.get(t)
            if sym is None or not self.securities.contains_key(sym) or s.capital.sleeve not in dm.SLEEVES:
                continue
            if self.selected and sym not in self.selected:
                continue
            px, vol = self._bars(sym, 300)
            if len(px) < 64:
                continue
            patt, _ = al.patterns(px, vol)
            view = self.last_views.get(s.subsector, {})
            feats = dict(al.price_features(px), composite=s.composite / 100, mos=s.margin_of_safety,
                         roic_wacc=s.roic_wacc, edge=view.get("edge", 0.0), pattern=patt)
            ml = self.wf.predict(feats)
            q = self.quality.prob_top(al.quality_features(s))
            k = kron.get(t)
            parts = [(weights.get("ml", 0), ml), (weights.get("quality", 0), None if q is None else 0.3 * (q - 1 / 3)),
                     (weights.get("kronos", 0), k)]
            parts = [(w, v) for w, v in parts if v is not None and w > 0]
            mw = sum(w for w, _ in parts)
            model_alpha = sum(w * v for w, v in parts) / mw if mw > 0 else None
            sec = self.securities[sym]
            fund_score = max(0.0, s.gate_score - ev["penalty"].get(t, 0.0))
            nature = SUBSECTORS.get(s.subsector, {}).get("nature", SECULAR)
            d = dm.decide(t, s.capital.sleeve, s.gate_pass and fund_score > 0, fund_score, px, vol, spy_closes,
                          view.get("edge"), nature, regime, model_alpha, mw, patt,
                          dm.spread_bps(getattr(sec, "bid_price", None), getattr(sec, "ask_price", None)))
            d.features = feats
            d.sector = s.sector
            catalyst = 0 <= ev["days"].get(t, 99) <= 21 or any("1.01" in f for f in ev["flags"].get(t, []))
            if d.approved and catalyst and s.capital.sleeve != "DISTRESSED":
                d.sleeve = "EVENT"                              # event-driven bucket (10%)
            out[t] = d
        dm.fit_sleeves(list(out.values()), getattr(self, "caps", None), self.sector_budget or None)
        return out

    def _etf_sleeve(self, prices, spy_closes):
        """TLTW / cash-flow 15% (income ETFs; commodities when bonds stop hedging) + FI / macro 5% (bonds)."""
        sb = stock_bond_regime(spy_closes, prices.get("TLT", []))
        live = lambda names: [t for t in names if len(prices.get(t, [])) > 21]   # listed with history
        income, comm = live(CASHFLOW_ETFS["income"]), live(CASHFLOW_ETFS["commodities"])
        comm_share = {"COMMODITIES/GOLD": 0.5, "GOLD/CASH": 1 / 3}.get(sb["hedge"], 0.0)
        if not income:
            comm_share = 1.0
        out = {t: 0.0 for t in ALL_ETFS}
        cf = self.caps.get("CASHFLOW", dm.SLEEVES["CASHFLOW"])
        for names, share in ((income, 1 - comm_share), (comm, comm_share)):
            for t in names:
                out[t] = cf * share / len(names)
        fi = live(FI_ETFS["TREASURIES"] if sb["hedge"] == "TREASURIES" else FI_ETFS["OTHER"]) or live(FI_ETFS["TREASURIES"])
        for t in fi:
            out[t] = self.caps.get("FI_MACRO", dm.SLEEVES["FI_MACRO"]) / len(fi)
        self.state_etf = {"hedge": sb["hedge"], "cashflow": {t: w for t, w in out.items() if w}}
        return out

    def _fit_alpha_budget(self, targets, prices):
        """Scale each new/increased equity target into the 2.5% stock-specific VaR budget (largest first)."""
        book, out = [], {}
        for t, w in sorted(targets.items(), key=lambda x: -abs(x[1])):
            if w <= 0:
                out[t] = 0.0
                continue
            d = self.decisions.get(t)
            etf = t in ALL_ETFS
            vol = d.idio_vol if d else (name_vol(prices.get(t, []), 63) or 0.25) * (0.4 if etf else 1.0)
            chk = check_alpha_trade(book, AlphaPosition(t, w, vol, getattr(self.scores.get(t), "subsector", "")))
            out[t] = chk.max_weight_allowed
            book.append(AlphaPosition(t, out[t], vol, getattr(self.scores.get(t), "subsector", "")))
        return out

    # ------------------------------------------------------------------ execution
    def _trade_equities(self, fraction, moc=False):
        if not self.trade_equities:
            return
        nav = self._nav()
        for t, target in self.targets.items():
            sym = self._symbol(t)
            if sym is None or not self.securities.contains_key(sym):
                continue
            sec = self.securities[sym]
            price = sec.price
            if price <= 0 or ((self.breaker_on or not self.feed_ok) and target > self._weight(sym)):
                continue                                   # no new risk under the breaker or a feed halt
            delta_w = target - self._weight(sym)
            if abs(delta_w) < MIN_TRADE:
                continue
            qty = int(delta_w * fraction * nav / price)
            if target == 0 and moc:
                qty = -int(self.portfolio[sym].quantity)
            if qty == 0:
                continue
            why = self.decisions[t].reasons[-1] if t in self.decisions and self.decisions[t].reasons else "rotation/sleeve"
            if moc:
                self._cancel_working(sym)
                self._drop_stop(sym)
                self.market_on_close_order(sym, qty, tag=f"MOC {t} → {target:.1%}: {why}")
            else:
                self._limit(sym, qty, "EQUITY", f"OPEN {t} → {target:.1%}: {why}")

    def core(self):
        if self.is_warming_up or self.state is None:
            return
        self._feed_check()
        if should_rebalance(self.current_leverage, self.beta_target):
            self._set_beta_leverage(self.beta_target, tag=f"{self.state['zone']} Rm={self.state['rm']:.2%} "
                                    f"{self.state['regime']}{' burst' if self.state['burst'] else ''} L={self.state['L']:.2f} "
                                    f"overlay={self.beta_target:.2f}")
            self.current_leverage = self.beta_target
        self._trade_equities(hb.OPEN_SLICE)
        self._sync_stops()

    def close_auction(self):
        if self.is_warming_up or self.state is None:
            return
        if not self._feed_check():
            self.debug("FEED HALT: closing-auction orders skipped (market-on-close has no price collar)")
            return
        self._trade_equities(1.0, moc=True)
        self._sync_stops()

    def expiry(self):
        if self.is_warming_up:
            return
        self._feed_check()
        self._option_exits(only_expiry=True)
        if self.state is not None and should_rebalance(self.current_leverage, self.beta_target):
            self._set_beta_leverage(self.beta_target, tag="beta true-up")
            self.current_leverage = self.beta_target

    # ------------------------------------------------------------------ risk heartbeat (every 30 min)
    def risk_check(self):
        if self.is_warming_up:
            return
        self._feed_check()
        nav = self._nav()
        if self.day_open_nav and not self.breaker_on and pr.breaker(nav, self.day_open_nav):
            self.breaker_on = True
            self.beta_target *= 0.5
            self._set_beta_leverage(self.beta_target, tag=f"daily-loss breaker {nav / self.day_open_nav - 1:.1%}")
            self.current_leverage = self.beta_target
            self.debug(f"BREAKER daily loss {nav / self.day_open_nav - 1:.2%}: no new entries, overlay halved")
        for kv in self.portfolio:
            sym, h = kv.key, kv.value
            if not h.invested or h.type != SecurityType.EQUITY or h.quantity <= 0:
                continue
            a = self.protector.update(sym.value, self.securities[sym].price, h.quantity)
            if a is None:
                continue
            qty = -int(h.quantity * a.fraction) if a.kind == "TRIM" else -int(h.quantity)
            if qty:
                self._limit(sym, qty, "EQUITY", f"PROTECT {a.kind} {sym.value}: {a.reason}")
            if sym.value in self.targets:
                self.targets[sym.value] = self._weight(sym) * (1 - a.fraction) if a.kind == "TRIM" else 0.0
        self._option_exits()
        ev = self.events()
        for kv in self.portfolio:
            sym, h = kv.key, kv.value
            if h.type == SecurityType.EQUITY and h.quantity > 0:
                act = pr.needs_hedge(self._weight(sym), ev["days"].get(sym.value), sym.value in self.hedged)
                if act == "HEDGE" and self.trade_options:
                    self.option_plans.append(dict(kind="HEDGE", ticker=sym.value, call=False, delta=0.3,
                                                  dte=pr.HEDGE_DTE, budget=pr.HEDGE_PREMIUM, mu=-0.2,
                                                  reason=f"event tail hedge (event in {ev['days'][sym.value]}d)"))
                    self.hedged.add(sym.value)
                elif act == "UNHEDGE":
                    self.hedged.discard(sym.value)
        self._sync_stops()

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
            if und is None or not self.securities.contains_key(und):
                continue
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

    # ------------------------------------------------------------------ learning (weekly)
    def _weekly_learning(self, spy_closes, kron):
        if self.time.weekday() != 0 or not self.decisions:
            return
        self.week_step += 1
        rows, qrows = {}, {}
        for t, d in self.decisions.items():
            sym = self.sym_of.get(t)
            if sym is None:
                continue
            price = self.securities[sym].price
            rows[t] = (getattr(d, "features", {}), price)
            if t in self.scores:
                qrows[t] = (al.quality_features(self.scores[t]), price)
        matured = self.wf.observe(rows)
        self.quality.observe(qrows)
        if self.week_step % self.wf.horizon_steps == 1:
            if matured:
                self.scorecard.settle(dict(matured))
            for t, d in self.decisions.items():
                f = getattr(d, "features", {})
                self.scorecard.record("score", t, dm.ALPHA_SCALE * (d.composite - 0.5))
                self.scorecard.record("ml", t, self.wf.predict(f))
                q = self.quality.prob_top(al.quality_features(self.scores[t])) if t in self.scores else None
                self.scorecard.record("quality", t, q)
                self.scorecard.record("kronos", t, kron.get(t))

    # ------------------------------------------------------------------ reporting
    def _log_plan(self, plan, esc):
        for ts in plan.themes:
            self.plot("Themes", f"{ts.theme} state", {"NORMAL": 0, "STRESS": 1, "CRASH": 2}[ts.state])
            self.plot("Themes", f"{ts.theme} drawdown %", ts.drawdown * 100)
        self.plot("Themes", "escalation score", esc)
        sig = tuple((a.kind, a.ticker, round(a.target_weight, 3)) for a in plan.actions)
        if sig != self.last_plan_sig:
            self.last_plan_sig = sig
            for a in plan.actions:
                self.debug(f"ROTATION {a.kind} {a.ticker} w={a.target_weight:.3f} premium={a.premium:.3%} "
                           f"dte={a.dte} — {a.reason}")
        self.last_plan = plan

    def _plots(self, drift, fwd, reg, state, cap, sigma):
        nav = self._nav()
        held = (self.portfolio[self.future.mapped].holdings_value if self.future is not None and self.future.mapped
                else self.portfolio[self.spy].holdings_value)
        actual = held / nav
        self.plot("Rm vs corridor", "Rm (forward)", state.rm * 100)
        self.plot("Rm vs corridor", "trailing EWMA", self.rm_ewma * 100)
        self.plot("Rm vs corridor", "7% floor", R_LOW * 100)
        self.plot("Rm vs corridor", "12% ceiling", R_HIGH * 100)
        for name in REGIMES:
            self.plot("Regime", f"P({name})", reg.probs.get(name, 0.0))
        self.plot("Corridor", "participation s(Rm)", state.participation)
        self.plot("Corridor", "gamma ds/dRm per 1%", state.gamma * 0.01)
        self.plot("Leverage", "corridor L", state.leverage)
        self.plot("Leverage", "corridor cap", cap)
        self.plot("Leverage", "MES overlay", self.beta_target)
        self.plot("Leverage", "futures cap", MAX_FUTURES_LEVERAGE)
        self.plot("Leverage", "sleeve beta", self.state.get("sleeve_beta", 0.0))
        self.plot("Leverage", "actual overlay", actual)
        self.plot("Risk", "realised vol %", sigma * 100)
        self.plot("Risk", "beta VaR95 1d %", daily_var(actual, sigma) * 100)
        self.plot("Risk", "drawdown %", (1 - nav / self.peak_nav) * 100)
        self.plot("Risk", "options premium % NAV", self._option_premium() * 100)
        self.plot("Risk", "MES margin % NAV", dm.futures_margin(self.current_leverage) * 100)
        for sec, w in sorted(self.sector_budget.items(), key=lambda x: -x[1])[:5]:
            self.plot("Sector budgets", sec, w)
        n_app = sum(1 for d in self.decisions.values() if d.approved)
        self.plot("Decisions", "approved", n_app)
        self.plot("Decisions", "candidates", len(self.decisions))
        for m, w in self.scorecard.weights().items():
            self.plot("Model weights", m, w)

    def _publish_state(self):
        nav = self._nav()
        pos = {}
        for kv in self.portfolio:
            sym, h = kv.key, kv.value
            if not h.invested:
                continue
            t = sym.value
            d = self.decisions.get(t)
            st = self.protector.states.get(t)
            pos[t] = {"type": str(h.type), "quantity": float(h.quantity), "weight": h.holdings_value / nav,
                      "unrealised_pct": float(h.unrealized_profit_percent), "target": self.targets.get(t),
                      "gates": d.gates if d else None, "reasons": d.reasons if d else None,
                      "subsector": getattr(self.scores.get(t), "subsector", None),
                      "credit_tier": getattr(getattr(self.scores.get(t), "capital", None), "credit_tier", None),
                      "peak": st.peak if st else None, "option": self.option_book.get(sym, {}).get("kind")}
        report = {"time": str(self.time), "nav": nav, "state": self.state, "breaker": self.breaker_on,
                  "feed": {"ok": self.feed_ok, "reasons": self.feed.state.reasons,
                           "spy_last": self.securities[self.spy].price, "spy_time": str(self.time),
                           "resting_stops": {str(k): round(v["stop"], 2) for k, v in self.stop_tickets.items()}},
                  "sector_scores": self.sector_score, "sector_budget": self.sector_budget,
                  "etf_buckets": getattr(self, "state_etf", {}), "bucket_caps": getattr(self, "caps", {}),
                  "positions": pos, "targets": self.targets,
                  "model_scorecard": self.scorecard.report(), "model_weights": self.scorecard.weights(),
                  "option_plans": [{k: v for k, v in p.items() if k != "rv"} for p in self.option_plans]}
        try:
            self.object_store.save("metadron/state.json", json.dumps(report, default=str))
        except Exception:
            pass

    # ------------------------------------------------------------------ trader framework (sector view)
    def run_sector_view(self, zone, prices, spy_closes, esc, vix, theme_states, readings):
        readings = dict(readings)
        readings.setdefault("escalation", esc)
        if vix is not None:
            readings.setdefault("vix", vix)
        hyg, lqd = prices.get("HYG"), prices.get("LQD")
        if hyg and lqd and len(hyg) > 22 and len(lqd) > 22:
            readings.setdefault("hy_oas_change_1m", -((hyg[-1] / hyg[-22]) - (lqd[-1] / lqd[-22])) / 4.0)
        uso = prices.get("USO")
        if uso and len(uso) > 22:
            readings.setdefault("commodities_21d", uso[-1] / uso[-22] - 1)
        scen_raw = self._store_read("metadron/scenarios.json")
        scenarios = None
        if scen_raw:
            try:
                scenarios = scenarios_from_json(scen_raw)
            except Exception as exc:
                self.debug(f"scenarios.json unreadable: {exc}")
        factors, findings = factor_states(readings), diagnose(readings)
        self.last_credit_stress = any("CREDIT_STRESS" in f.tags for f in findings)   # spreads widening → themes exposed
        bias = market_bias(factors, zone)
        sb = stock_bond_regime(spy_closes, prices.get("TLT", []))
        themes = {THEME_SUBSECTOR[t.theme]: t.state for t in theme_states if t.theme in THEME_SUBSECTOR}
        views = evaluate(factors, scenarios=scenarios, findings=findings, theme_states=themes, bias=bias["bias"],
                         fundamentals=self.subsector_fundamentals())
        self.last_views = {v.subsector: {"edge": v.edge, "weight": v.weight} for v in views}
        mix = {CYCLICAL: 0.0, SECULAR: 0.0, DEFENSIVE: 0.0}
        for v in views:
            mix[v.nature] += v.weight
        self.plot("Market view", "bias score", bias["score"])
        self.plot("Market view", "stock-bond corr", sb["corr"] if sb["corr"] is not None else 0.0)
        for n, w in mix.items():
            self.plot("Equity mix", n.lower(), w)
        report = {"time": str(self.time), "zone": zone, "bias": bias, "stock_bond": sb,
                  "findings": [f.__dict__ for f in findings], "factors": factors, **summary(views, 8),
                  "gics_sector_scores": getattr(self, "sector_score", {})}
        try:
            self.object_store.save("metadron/sector_views.json", json.dumps(report, default=str))
        except Exception:
            pass
        top = ", ".join(f"{v.subsector} {v.weight:.0%}" for v in views if v.weight > 0.05)
        sig = (bias["bias"], tuple(sorted(v.subsector for v in views if v.weight > 0.05)))
        last_t = getattr(self, "last_view_time", None)
        if sig != getattr(self, "last_view_sig", None) and (last_t is None or (self.time - last_t).days >= 5):
            self.last_view_sig, self.last_view_time = sig, self.time
            self.debug(f"VIEW {bias['bias']} ({bias['score']:+.2f}) hedge={sb['hedge']} | {top}")
        return bias["bias"]
