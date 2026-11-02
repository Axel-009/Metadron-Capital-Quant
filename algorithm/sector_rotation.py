"""Sector rotation — scenario-weighted sub-sector tilts (generalises the theme-crash playbook).

    factor states (macro_regime.factor_states)            "where are we now"
  + scenarios (base / up / down what-ifs, with probabilities and factor shocks; written by the
    macro agent or the owner — see agents/market_reasoning.md)   "what could happen next"
  → expected factors and per-scenario factors
  → sub-sector score = Σ sensitivity × factor          (market_map.SUBSECTORS)
                     + question-tree implications       (macro_regime.diagnose)
                     + fundamentals                     (earnings / revenue revisions, legal fights)
                     + theme state                      (rotation.theme_state: STRESS −0.3, CRASH −0.8)
  → EDGE      = probability-weighted score
    DOWNSIDE  = score in the worst scenario
  → tilts: equity-sleeve weights by nature (cyclical / secular / defensive) set by the market bias,
    then within each nature ∝ positive edge, penalised by downside.

Everything is explainable: each score lists its top drivers, for agents, Obsidian and Jarvis.
Pure Python + math.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from market_map import SUBSECTORS, FACTORS, CYCLICAL, SECULAR, DEFENSIVE
from macro_regime import Finding, implication_adjustments

NATURE_MIX = {      # share of the equity sleeve by market bias
    "BULLISH": {CYCLICAL: 0.40, SECULAR: 0.45, DEFENSIVE: 0.15},
    "NEUTRAL": {CYCLICAL: 0.30, SECULAR: 0.40, DEFENSIVE: 0.30},
    "BEARISH": {CYCLICAL: 0.15, SECULAR: 0.30, DEFENSIVE: 0.55},
}
THEME_PENALTY = {"NORMAL": 0.0, "STRESS": -0.3, "CRASH": -0.8}
FUNDAMENTAL_WEIGHT = 0.5
DOWNSIDE_AVERSION = 0.5          # tilt weight ∝ max(edge − aversion·max(−downside, 0), 0)


@dataclass
class Scenario:
    name: str                       # e.g. "base", "AI capex cut", "escalation"
    probability: float
    shocks: Dict[str, float]        # factor → change (added to current state, then clipped)
    narrative: str = ""
    source: str = "owner"           # agent id or "owner"


def scenarios_from_json(text: str) -> List[Scenario]:
    """Parse the agent / owner scenario file (schema in agents/market_reasoning.md)."""
    data = json.loads(text)
    out = []
    for s in data.get("scenarios", []):
        shocks = {k: float(v) for k, v in s.get("shocks", {}).items() if k in FACTORS}
        out.append(Scenario(s["name"], float(s["probability"]), shocks, s.get("narrative", ""),
                            s.get("source", data.get("source", "agent"))))
    return normalise(out)


def normalise(scens: List[Scenario]) -> List[Scenario]:
    tot = sum(max(s.probability, 0.0) for s in scens)
    if tot <= 0:
        return [Scenario("base", 1.0, {})]
    for s in scens:
        s.probability = max(s.probability, 0.0) / tot
    return scens


def apply(factors: Dict[str, float], shocks: Dict[str, float]) -> Dict[str, float]:
    out = {f: factors.get(f, 0.0) for f in FACTORS}
    for f, d in shocks.items():
        out[f] = max(-1.0, min(1.0, out.get(f, 0.0) + d))
    return out


@dataclass
class SubsectorView:
    subsector: str
    nature: str
    edge: float                     # probability-weighted score
    downside: float                 # worst-scenario score
    by_scenario: Dict[str, float]
    drivers: List[str] = field(default_factory=list)
    weight: float = 0.0             # share of the equity sleeve

    def to_dict(self) -> dict:
        return {"subsector": self.subsector, "nature": self.nature, "edge": round(self.edge, 4),
                "downside": round(self.downside, 4), "weight": round(self.weight, 4),
                "by_scenario": {k: round(v, 4) for k, v in self.by_scenario.items()},
                "drivers": self.drivers}


def _base_score(sub: str, factors: Dict[str, float]) -> float:
    sens = SUBSECTORS[sub]["sens"]
    return sum(sens.get(f, 0.0) * factors.get(f, 0.0) for f in FACTORS) / max(
        sum(abs(v) for v in sens.values()), 1.0)


def evaluate(factors: Dict[str, float],
             scenarios: Optional[List[Scenario]] = None,
             findings: Optional[List[Finding]] = None,
             fundamentals: Optional[Dict[str, float]] = None,
             theme_states: Optional[Dict[str, str]] = None,
             bias: str = "NEUTRAL",
             subsectors: Optional[List[str]] = None) -> List[SubsectorView]:
    """Score and tilt the sub-sectors. fundamentals: {subsector: −1..1} (revisions, revenue, legal);
    theme_states: {subsector: NORMAL|STRESS|CRASH}."""
    scenarios = normalise(list(scenarios)) if scenarios else [Scenario("base", 1.0, {})]
    implications = implication_adjustments(findings or [])
    fundamentals = fundamentals or {}
    theme_states = theme_states or {}
    subs = subsectors or list(SUBSECTORS)

    views: List[SubsectorView] = []
    for sub in subs:
        extra = (implications.get(sub, 0.0) + FUNDAMENTAL_WEIGHT * fundamentals.get(sub, 0.0)
                 + THEME_PENALTY.get(theme_states.get(sub, "NORMAL"), 0.0))
        per = {s.name: _base_score(sub, apply(factors, s.shocks)) + extra for s in scenarios}
        edge = sum(s.probability * per[s.name] for s in scenarios)
        downside = min(per.values())
        # explain: biggest contributions under expected factors
        exp_f = {f: sum(s.probability * apply(factors, s.shocks)[f] for s in scenarios) for f in FACTORS}
        sens = SUBSECTORS[sub]["sens"]
        contrib = sorted(((sens[f] * exp_f.get(f, 0.0), f) for f in sens), key=lambda x: -abs(x[0]))
        drivers = [f"{f} {c:+.2f}" for c, f in contrib[:3] if abs(c) > 1e-3]
        if implications.get(sub):
            drivers.append(f"implications {implications[sub]:+.2f}")
        if fundamentals.get(sub):
            drivers.append(f"fundamentals {fundamentals[sub]:+.2f}")
        if theme_states.get(sub, "NORMAL") != "NORMAL":
            drivers.append(f"theme {theme_states[sub]}")
        views.append(SubsectorView(sub, SUBSECTORS[sub]["nature"], edge, downside, per, drivers))

    mix = NATURE_MIX.get(bias, NATURE_MIX["NEUTRAL"])
    for nature, share in mix.items():
        group = [v for v in views if v.nature == nature]
        raw = {v.subsector: max(v.edge - DOWNSIDE_AVERSION * max(-v.downside, 0.0), 0.0) for v in group}
        tot = sum(raw.values())
        for v in group:
            v.weight = share * raw[v.subsector] / tot if tot > 0 else 0.0
    return sorted(views, key=lambda v: -v.edge)


def summary(views: List[SubsectorView], n: int = 5) -> dict:
    """Top / bottom sub-sectors with reasons — the text agents and Jarvis report."""
    return {
        "overweight": [v.to_dict() for v in views if v.weight > 0][:n],
        "avoid": [v.to_dict() for v in sorted(views, key=lambda v: v.edge)[:n]],
    }
