# region imports
from AlgorithmImports import *
# endregion
"""Reporting: universe / plan / state publication, charts, live status lines, the sector view — mixed into
MetadronCorridor (split out of main.py: QuantConnect caps files at 64,000 characters)."""

import json

from corridor import R_LOW, R_HIGH
from risk_budget import daily_var, MAX_FUTURES_LEVERAGE
from regime import REGIMES
from universe import EXPECTED_MEMBERS
from profiles import THEMES as THEME_PROFILES
from rotation import THEME_SUBSECTOR
from macro_regime import factor_states, diagnose, market_bias, stock_bond_regime
from sector_rotation import evaluate, scenarios_from_json, summary
from market_map import CYCLICAL, SECULAR, DEFENSIVE
import decision as dm


class ReportMixin:   # plain holder of methods (not a base class — see main.py)
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
        ds = list(self.decisions.values())
        app = [d for d in ds if d.approved]
        why = {}
        for d in ds:
            if not d.approved and d.reasons:
                k = d.reasons[0].split(" ")[0] + " " + (d.reasons[0].split(" ") + [""])[1]
                why[k] = why.get(k, 0) + 1
        live = {t: w for t, w in self.targets.items() if w > 0}
        self._stat("Regime", f"{self.state['regime']} {self.state['zone']} Rm {self.state['rm']:.1%}")
        self._stat("Decisions", f"{len(app)} approved of {len(ds)} (shortlist {len(self.selected)})")
        self._stat("Rejected for", ", ".join(f"{k} x{n}" for k, n in sorted(why.items(), key=lambda x: -x[1])[:3]) or "-")
        sl = {}
        for t, w in live.items():                          # deployed weight by sleeve (events, distressed, …)
            d = self.decisions.get(t)
            k = d.sleeve if d is not None else "ETF"
            sl[k] = sl.get(k, 0.0) + w
        self._stat("Sleeves", " ".join(f"{k[:5]} {v:.0%}" for k, v in sorted(sl.items(), key=lambda x: -x[1])))
        ed = self._earnings_days()
        self._stat("Events", f"{sum(1 for t in live if 0 <= ed.get(t, 99) <= 21)} held with earnings ≤21d, "
                             f"{len(self.hedged)} hedged, calendar {len(ed)} names")
        short = {t: w for t, w in self.targets.items() if w < 0}
        self._stat("Targets", f"long {sum(live.values()):.0%} ({len(live)}) short {-sum(short.values()):.0%} "
                              f"({len(short)}) undeployed {getattr(self, 'undeployed', 0.0):.0%}")
        pnl, by_name = {}, {}
        for kind, name, p in self.realised.values():       # realised, every symbol ever traded
            pnl[kind] = pnl.get(kind, 0.0) + p
            by_name[name] = by_name.get(name, 0.0) + p
        for kv in self.portfolio:                          # + open positions
            h = kv.value
            try:
                p = float(h.unrealized_profit)
            except Exception:
                continue
            if p == 0 or not h.invested:
                continue
            kind = "options" if h.type == SecurityType.OPTION else "stocks/ETFs"
            pnl[kind] = pnl.get(kind, 0.0) + p
            und = getattr(kv.key, "underlying", None)
            name = (und.value + " opt") if h.type == SecurityType.OPTION and und is not None else kv.key.value
            by_name[name] = by_name.get(name, 0.0) + p
        top = sorted(by_name.items(), key=lambda x: -abs(x[1]))[:3]
        self._stat("PnL by type", ", ".join(f"{k} {v:+,.0f}" for k, v in pnl.items()) or "-")
        self._stat("PnL top 3", ", ".join(f"{k} {v:+,.0f}" for k, v in top) or "-")
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
                  "option_plans": [{k: v for k, v in p.items() if k != "rv"} for p in self.option_plans],
                  "option_picks": self.option_log[-40:], "option_rejects": self.opt_rejects,
                  "pairs": [p.__dict__ for p in self.pairs[:20]], "gate_weights": dict(dm.WEIGHTS),
                  "playbook_state": getattr(self, "pb_state", None), "theme_leadership": getattr(self, "leaders", {})}
        try:
            self.object_store.save("metadron/state.json", json.dumps(report, default=str))
        except Exception:
            pass

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
        scen_raw = self._agent_read("metadron/scenarios.json")
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

    def subsector_fundamentals(self) -> dict:
        """{subsector: −1..1} = 2 × (mean gate score − 0.5) over scored names → sector_rotation input."""
        agg = {}
        for s in self.scores.values():
            agg.setdefault(s.subsector, []).append(s.gate_score)
        return {k: max(-1.0, min(1.0, 2 * (sum(v) / len(v) - 0.5))) for k, v in agg.items() if len(v) >= 3}

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
        self.debug(f"UNIVERSE {h} | EY {td.earnings_yield:.2%} 10y {self._ten_year(self.macro_inputs()):.2%} "
                   f"ERP {td.erp:+.2%} {td.regime} | "
                   + " ".join(f"{k}:{','.join(v[:5])}" for k, v in sl.by_sleeve.items()))
