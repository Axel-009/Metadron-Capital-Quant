# region imports
from AlgorithmImports import *
# endregion
"""Research desk add-ons (owner, Nov 2026): learned playbook, theme leadership, cointegration pairs, adaptive gate
weights, earnings-based event flags, owner watch lists — mixed into MetadronCorridor."""

import json
from datetime import timedelta, date as _date

import decision as dm
import playbook as pb
import statarb as sa
from sectors import SUBSECTOR_TO_GICS

PLAYBOOK_REFIT_DAYS = 21
PLAYBOOK_BARS = 252 * 14
LEADERSHIP_WEIGHT = 0.30        # added to a sub-sector's edge: leadership (−1 … +1) × 0.30
LEADERSHIP_BARS = 63
EARNINGS_GAP = 91
GATE_PRIOR = dict(dm.WEIGHTS)
GATE_BOUNDS = (0.10, 0.50)
GATE_FLOOR = {"FUNDAMENTALS": 0.25}     # the must-pass gate never falls below a quarter of the score
GATE_BLEND = 0.5                # new weight = 50% prior + 50% learned (rank-IC proportional)
GATE_MIN_PERIODS = 3


class ResearchMixin:   # plain holder of methods (not a base class — see main.py)
    # ------------------------------------------------------------------ playbook: what works in this kind of market
    def _playbook_update(self, spy_long):
        """Refit monthly on all history to date; returns the sector tilts for today's market state."""
        today = self.time.date()
        last = getattr(self, "pb_time", None)
        if last is None or (today - last).days >= PLAYBOOK_REFIT_DAYS:
            spy = self._closes(self.spy, PLAYBOOK_BARS) or spy_long
            secs = {}
            for sec, tkr in pb.SECTOR_PROXY.items():
                sym = self.sector_syms.get(tkr)
                c = self._closes(sym, PLAYBOOK_BARS) if sym is not None else []
                if len(c) > pb.YEAR + pb.HORIZON:
                    secs[sec] = c
            self.pb_table, self.pb_time = pb.learn(spy, secs), today
            self.pb_spy = spy
            try:
                self.object_store.save("metadron/playbook.json", json.dumps(
                    {"time": str(self.time), "sectors": {st: {s: {"excess": m, "n": n} for s, (m, n) in row.items()}
                                                         for st, row in self.pb_table.items()},
                     "natures": pb.nature_view(self.pb_table)}, default=str))
            except Exception:
                pass
        state = pb.market_state(spy_long if len(spy_long) > pb.YEAR else getattr(self, "pb_spy", spy_long))
        if state != getattr(self, "pb_state", None):
            self.pb_state = state
            self.debug("PLAYBOOK " + pb.describe(self.pb_table, state) + " | natures "
                       + ", ".join(f"{k} {v:+.0%}" for k, v in pb.nature_view(self.pb_table).get(state or "", {}).items()))
        self._stat("Playbook", pb.describe(self.pb_table, state, 2))
        return pb.tilts(self.pb_table, state)

    # ------------------------------------------------------------------ themes: which sub-sectors lead / break
    def _leadership(self):
        """Sub-sector leadership from every scored name's own 63-day return → added to the sub-sector edges."""
        rets = {}
        for t, s in self.scores.items():
            px = list(self.px_buf.p.get(t, []))
            if len(px) > LEADERSHIP_BARS and px[-LEADERSHIP_BARS - 1] > 0:
                r = px[-1] / px[-LEADERSHIP_BARS - 1] - 1
                if -0.9 < r < 3.0:                              # ignore split artefacts in the raw universe price
                    rets.setdefault(s.subsector, []).append(r)
        lead = pb.leadership(rets)
        self.leaders = lead
        for sub, v in lead.items():
            if sub in self.last_views:
                self.last_views[sub]["edge"] += LEADERSHIP_WEIGHT * v
        top = sorted(lead.items(), key=lambda x: -x[1])
        if top:
            line = "lead " + ", ".join(f"{k} {v:+.1f}" for k, v in top[:3]) + " | lag " \
                   + ", ".join(f"{k} {v:+.1f}" for k, v in top[-2:])
            self._stat("Themes", line)
            sig = (tuple(k for k, _ in top[:3]), tuple(k for k, _ in top[-2:]))
            last = getattr(self, "leader_time", None)
            if sig != getattr(self, "leader_sig", None) and (last is None or (self.time - last).days >= 5):
                self.leader_sig, self.leader_time = sig, self.time
                self.debug("THEMES " + line)

    # ------------------------------------------------------------------ cointegration pairs
    def _statarb(self):
        """Signed pair-leg targets {ticker: weight}; formation refit every 21 days, z-scores daily."""
        today = self.time.date()
        px, sectors = {}, {}
        for t, d in self.decisions.items():
            c = getattr(d, "px", None)
            if c and len(c) >= 120:
                px[t], sectors[t] = c, d.sector
        last = getattr(self, "pairs_time", None)
        if last is None or (today - last).days >= sa.REFIT_DAYS:
            old = {(p.y, p.x): p for p in getattr(self, "pairs", [])}
            fresh = sa.find_pairs(px, sectors)
            for p in fresh:                                     # an open pair keeps its side through a refit
                if (p.y, p.x) in old:
                    p.side = old[(p.y, p.x)].side
            self.pairs, self.pairs_time = fresh, today
            self.debug(f"STATARB {len(fresh)} cointegrated pairs: "
                       + ", ".join(f"{p.y}/{p.x} DF {p.adf:.1f} hl {p.half_life:.0f}d" for p in fresh[:6]))
        for p in getattr(self, "pairs", []):
            if p.y in px and p.x in px:
                act = sa.update(p, px[p.y][-1], px[p.x][-1])
                if act in ("OPEN", "CLOSE", "STOP"):
                    self.debug(f"STATARB {act} {p.y}/{p.x} z={p.z:+.2f} hedge {p.b:.2f} "
                               + ("long " + (p.y if p.side > 0 else p.x) if p.side else ""))
            else:
                p.side = 0
        legs = sa.targets(getattr(self, "pairs", []))
        n_open = sum(1 for p in getattr(self, "pairs", []) if p.side)
        self._stat("Pairs", f"{len(getattr(self, 'pairs', []))} cointegrated, {n_open} open, "
                            f"gross {sum(abs(w) for w in legs.values()):.1%}")
        return legs

    # ------------------------------------------------------------------ gates that learn
    def _learn_gates(self, matured):
        """Score each gate against realised 21-day returns (rank IC) and move the gate weights toward what works."""
        if matured:
            self.gate_card.settle(dict(matured))
        for t, d in self.decisions.items():
            if d.sleeve != "SHORT":
                for g, v in d.gates.items():
                    self.gate_card.record(g, t, v)
        rep = self.gate_card.report()
        if any(rep[g]["n"] < GATE_MIN_PERIODS for g in GATE_PRIOR):
            return
        ic = {g: max(rep[g]["ic_mean"] or 0.0, 0.0) for g in GATE_PRIOR}
        tot = sum(ic.values())
        if tot <= 0:
            return                                              # nothing predicts: keep the prior weights
        w = {g: (1 - GATE_BLEND) * GATE_PRIOR[g] + GATE_BLEND * ic[g] / tot for g in GATE_PRIOR}
        w = {g: min(max(v, GATE_FLOOR.get(g, GATE_BOUNDS[0])), GATE_BOUNDS[1]) for g, v in w.items()}
        s = sum(w.values())
        new = {g: v / s for g, v in w.items()}
        if any(abs(new[g] - dm.WEIGHTS[g]) > 0.02 for g in new):
            self.debug("GATES weights " + ", ".join(f"{g} {dm.WEIGHTS[g]:.0%}→{new[g]:.0%} (IC {rep[g]['ic_mean']:+.3f})"
                                                    for g in new))
        dm.WEIGHTS.update(new)
        self._stat("Gate weights", " ".join(f"{g[:4]} {v:.0%}" for g, v in dm.WEIGHTS.items()))

    # ------------------------------------------------------------------ events from the filings calendar
    def _earnings_days(self):
        """{ticker: days to the next earnings report}, estimated from the last report's filing date (+91 days,
        rolled forward) — point-in-time fundamentals, so event flags exist in backtests and without the agent."""
        today = self.time.date()
        if getattr(self, "_earn_day", None) == today:
            return self._earn_cache
        out = {}
        for t, iso in getattr(self, "earn_dates", {}).items():
            try:
                nxt = _date.fromisoformat(iso) + timedelta(days=EARNINGS_GAP)
            except Exception:
                continue
            while (nxt - today).days < -5:
                nxt += timedelta(days=EARNINGS_GAP)
            out[t] = (nxt - today).days
        self._earn_day, self._earn_cache = today, out
        return out

    def _owner_watch(self, key):
        """Owner watch lists (project parameter, comma-separated tickers), e.g. distressed_watch=AAA,BBB."""
        raw = str(self.get_parameter(key, "") or "")
        return {t.strip().upper() for t in raw.split(",") if t.strip()}
