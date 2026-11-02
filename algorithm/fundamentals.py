"""Fundamentals → selection: Graham-Dodd-Klarman security analysis + capital structure + gate.

Pure Python port of engine/signals/security_analysis_engine.py (top-down + bottom-up), extended with the
capital-structure monitor (architecture point 9) and the Fundamentals gate (40% weight, must pass).

TOP-DOWN (market, from the universe aggregate — no CAPE feed needed)
    earnings yield EY = Σ earnings / Σ market cap        ERP = EY − 10y yield
    max investment P/E by rate (Graham ch. 39)           speculative % = (P/E − max P/E)/P/E
    valuation regime DEEPLY_UNDERVALUED … EXTREMELY_OVERVALUED (CAPE proxied by the aggregate P/E)
    EY is also the forward-Rm valuation anchor (forward_rm.anchor) when the macro agent sends none.

BOTTOM-UP (per name)
    normalized EPS (mean of up to 10 annual EPS seen), Graham number √(22.5·EPS·tangible BVPS), P/E×P/B,
    NCAV = current assets − total liabilities, ROIC (NOPAT / invested capital) − WACC, DuPont, owner earnings,
    accruals (FCF / NI), interest coverage, debt/EBITDA, 5-method intrinsic value (Graham number, earning
    power, max-P/E × EPS, NCAV, capped DCF) → margin of safety, 8-test investment grade, composite 0–100.

CAPITAL STRUCTURE (per name)
    Altman Z (public manufacturer form), Merton distance-to-default → PD (1y), interest coverage,
    debt/EBITDA, FCF vs capex burn, credit tier A–F (Egan-Jones D/E + current ratio, downgraded by Z / PD /
    coverage), sleeve IG (A/B) · HY (C/D) · DISTRESSED (E/F).

GATE
    IG / HY names: pass if composite ≥ 50 and no hard block.
    DISTRESSED names: pass only as a recovery play — NCAV discount ≥ 0 or MoS ≥ 33%, PD ≤ 25%, FCF ≥ 0.
    Hard blocks: negative equity with negative FCF; coverage < 1.5 with PD > 10% (outside the distressed sleeve);
    missing price / shares. Quality tier A–G from composite (A ≥ 80 … G < 30).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

TAX = 0.21
ERP_ASSUMED = 0.05
DEBT_SPREAD = 0.02
RATE_TO_MAX_PE = [(0.00, 0.02, 25.0), (0.02, 0.04, 20.0), (0.04, 0.06, 16.0), (0.06, 0.08, 14.0), (0.08, 0.15, 10.0)]
COVERAGE_MIN = {"UTILITY": 3.0, "FINANCIAL": 4.0, "REAL_ESTATE": 3.0, "NATURAL_RESOURCE": 4.0}
COVERAGE_MIN_DEFAULT = 5.0
GATE_MIN = 50.0
TIERS = [(80, "A"), (70, "B"), (60, "C"), (50, "D"), (40, "E"), (30, "F")]
SLEEVE = {"A": "IG", "B": "IG", "C": "HY", "D": "HY", "E": "DISTRESSED", "F": "DISTRESSED"}


def _n(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# ------------------------------------------------------------------------------- top-down
def max_pe_for_rate(rate: float) -> float:
    for lo, hi, pe in RATE_TO_MAX_PE:
        if lo <= rate < hi:
            return pe
    return 8.0


@dataclass
class TopDown:
    earnings_yield: float = 0.05
    market_pe: float = 20.0
    treasury_10y: float = 0.042
    erp: float = 0.0
    max_pe: float = 16.0
    speculative_pct: float = 0.0
    regime: str = "FAIR_VALUE"
    n_names: int = 0


def top_down(names: List[dict], treasury_10y: float = 0.042, vix: Optional[float] = None) -> TopDown:
    """Aggregate valuation of the universe (cap-weighted earnings yield = Σ E / Σ cap)."""
    caps = earn = 0.0
    n = 0
    for d in names:
        cap = _n(d.get("market_cap"))
        if cap <= 0:
            continue
        caps += cap
        earn += _n(d.get("net_income"))
        n += 1
    ey = earn / caps if caps > 0 else 0.05
    pe = 1 / ey if ey > 0 else 99.0
    td = TopDown(earnings_yield=ey, market_pe=pe, treasury_10y=treasury_10y, n_names=n)
    td.erp = ey - treasury_10y
    td.max_pe = max_pe_for_rate(treasury_10y)
    td.speculative_pct = max(0.0, (pe - td.max_pe) / pe)
    score = (2 if pe < 12 else 1 if pe < 16 else 0 if pe < 22 else -1 if pe < 30 else -2)
    score += (2 if td.erp > 0.06 else 1 if td.erp > 0.04 else 0 if td.erp > 0.02 else -1 if td.erp > 0 else -2)
    score += -1 if td.speculative_pct > 0.30 else 0.5 if td.speculative_pct < 0.05 else 0
    if vix is not None:
        score += 0.5 if vix > 35 else -0.5 if vix < 12 else 0
    td.regime = ("DEEPLY_UNDERVALUED" if score >= 3 else "UNDERVALUED" if score >= 1 else
                 "FAIR_VALUE" if score >= -1 else "OVERVALUED" if score >= -3 else "EXTREMELY_OVERVALUED")
    return td


# ------------------------------------------------------------------------------- capital structure
def altman_z(d: dict) -> Optional[float]:
    ta = _n(d.get("total_assets"))
    tl = _n(d.get("total_liabilities"))
    if ta <= 0 or tl <= 0:
        return None
    wc = _n(d.get("current_assets")) - _n(d.get("current_liabilities"))
    return (1.2 * wc / ta + 1.4 * _n(d.get("retained_earnings")) / ta + 3.3 * _n(d.get("ebit")) / ta
            + 0.6 * _n(d.get("market_cap")) / tl + 1.0 * _n(d.get("revenue")) / ta)


def merton_pd(equity: float, debt: float, equity_vol: float, horizon: float = 1.0, drift: float = 0.0) -> Optional[float]:
    """1-year default probability from a simplified Merton model (KMV-style approximation)."""
    if equity <= 0 or equity_vol <= 0:
        return None
    if debt <= 0:
        return 0.0
    v = equity + debt
    sv = equity_vol * equity / v + 0.05 * debt / v          # debt vol ≈ 5%
    dd = (math.log(v / debt) + (drift - 0.5 * sv * sv) * horizon) / (sv * math.sqrt(horizon))
    return _norm_cdf(-dd)


@dataclass
class CapitalStructure:
    altman_z: Optional[float] = None
    merton_pd: Optional[float] = None
    interest_coverage: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    fcf_to_capex: Optional[float] = None
    credit_tier: str = "C"
    sleeve: str = "HY"
    flags: List[str] = field(default_factory=list)


def capital_structure(d: dict, equity_vol: Optional[float] = None) -> CapitalStructure:
    cs = CapitalStructure()
    debt = _n(d.get("total_debt"))
    ebit = _n(d.get("ebit"))
    ebitda = ebit + _n(d.get("depreciation"))
    intexp = abs(_n(d.get("interest_expense")))
    equity = _n(d.get("total_assets")) - _n(d.get("total_liabilities"))
    cl = _n(d.get("current_liabilities"))
    cs.interest_coverage = ebit / intexp if intexp > 0 else (99.0 if debt <= 0 or ebit > 0 else None)
    cs.debt_to_ebitda = debt / ebitda if ebitda > 0 else (0.0 if debt <= 0 else 99.0)
    cs.debt_to_equity = debt / equity if equity > 0 else 99.0
    cs.current_ratio = _n(d.get("current_assets")) / cl if cl > 0 else None
    capex = abs(_n(d.get("capex")))
    cs.fcf_to_capex = _n(d.get("fcf")) / capex if capex > 0 else None
    cs.altman_z = altman_z(d)
    if equity_vol:
        cs.merton_pd = merton_pd(_n(d.get("market_cap")), debt, equity_vol)

    de, cr = cs.debt_to_equity, cs.current_ratio if cs.current_ratio is not None else 1.0
    tier = ("A" if de < 0.5 and cr > 1.5 else "B" if de < 1.0 and cr > 1.2 else "C" if de < 2.0 and cr > 0.8
            else "D" if de < 3.0 and cr > 0.8 else "E" if de < 5.0 and cr > 0.5 else "F")
    order = "ABCDEF"
    down = 0
    if cs.altman_z is not None and cs.altman_z < 1.8:
        down += 2; cs.flags.append(f"Altman Z {cs.altman_z:.2f} distress zone")
    elif cs.altman_z is not None and cs.altman_z < 3.0:
        down += 0 if tier in "CDEF" else 1; cs.flags.append(f"Altman Z {cs.altman_z:.2f} grey zone")
    if cs.merton_pd is not None and cs.merton_pd > 0.10:
        down += 2; cs.flags.append(f"Merton PD {cs.merton_pd:.1%}")
    elif cs.merton_pd is not None and cs.merton_pd > 0.02:
        down += 1; cs.flags.append(f"Merton PD {cs.merton_pd:.1%}")
    if cs.interest_coverage is not None and cs.interest_coverage < 1.5:
        down += 2; cs.flags.append(f"coverage {cs.interest_coverage:.1f}x")
    elif cs.interest_coverage is not None and cs.interest_coverage < 3.0:
        down += 1; cs.flags.append(f"coverage {cs.interest_coverage:.1f}x")
    if cs.debt_to_ebitda is not None and cs.debt_to_ebitda > 5:
        down += 1; cs.flags.append(f"debt/EBITDA {cs.debt_to_ebitda:.1f}x")
    if cs.fcf_to_capex is not None and cs.fcf_to_capex < -0.5:
        cs.flags.append("FCF burn > 50% of capex")
    cs.credit_tier = order[min(order.index(tier) + down, 5)]
    cs.sleeve = SLEEVE[cs.credit_tier]
    return cs


# ------------------------------------------------------------------------------- bottom-up
@dataclass
class Score:
    ticker: str
    sector: str = ""
    subsector: str = ""
    price: float = 0.0
    normalized_eps: float = 0.0
    graham_number: float = 0.0
    pe_x_pb: float = 999.0
    earnings_yield: float = 0.0
    ncav_per_share: float = 0.0
    ncav_discount: float = -1.0
    roic: float = 0.0
    wacc: float = 0.0
    roic_wacc: float = 0.0
    roe: float = 0.0
    dupont: Dict[str, float] = field(default_factory=dict)
    owner_earnings: float = 0.0
    accrual_quality: float = 0.0
    eps_stability: float = 999.0
    intrinsic_value: float = 0.0
    iv_methods: Dict[str, float] = field(default_factory=dict)
    margin_of_safety: float = -1.0
    tests_passed: int = 0
    grade: str = "AVOID"
    composite: float = 0.0
    quality_tier: str = "G"
    capital: CapitalStructure = field(default_factory=CapitalStructure)
    gate_pass: bool = False
    gate_score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["capital"] = asdict(self.capital)
        return d


def analyze(ticker: str, d: dict, td: TopDown, equity_vol: Optional[float] = None,
            eps_history: Optional[List[float]] = None, sector_type: str = "") -> Score:
    s = Score(ticker=ticker, sector=str(d.get("sector", "")))
    price, shares = _n(d.get("price")), _n(d.get("shares_out"))
    if price <= 0 or shares <= 0:
        s.reasons.append("missing price / shares")
        return s
    s.price = price
    eps_ttm = _n(d.get("eps_ttm"))
    hist = [e for e in (eps_history or []) if e is not None]
    s.normalized_eps = sum(hist) / len(hist) if len(hist) >= 3 else eps_ttm
    bvps = _n(d.get("bvps"))
    tbvps = _n(d.get("tangible_bvps")) or bvps
    s.graham_number = math.sqrt(22.5 * s.normalized_eps * tbvps) if s.normalized_eps > 0 and tbvps > 0 else 0.0
    pe = price / eps_ttm if eps_ttm > 0 else 999.0
    pb = price / tbvps if tbvps > 0 else 999.0
    s.pe_x_pb = pe * pb
    s.earnings_yield = eps_ttm / price

    ca, tl = _n(d.get("current_assets")), _n(d.get("total_liabilities"))
    s.ncav_per_share = (ca - tl) / shares
    s.ncav_discount = (s.ncav_per_share - price) / s.ncav_per_share if s.ncav_per_share > 0 else -1.0

    if len(hist) >= 3:
        m = sum(hist) / len(hist)
        sd = math.sqrt(sum((x - m) ** 2 for x in hist) / len(hist))
        s.eps_stability = sd / abs(m) if m else 999.0

    ta = _n(d.get("total_assets"))
    equity = ta - tl
    debt, cash = _n(d.get("total_debt")), _n(d.get("cash"))
    ebit, ni, rev = _n(d.get("ebit")), _n(d.get("net_income")), _n(d.get("revenue"))
    fcf = _n(d.get("fcf"))
    invested = equity + debt - cash
    s.roic = _n(d.get("roic")) or (ebit * (1 - TAX) / invested if invested > 0 else 0.0)
    coe, cod = td.treasury_10y + ERP_ASSUMED, td.treasury_10y + DEBT_SPREAD
    cap_tot = max(equity, 0) + debt
    s.wacc = (coe * max(equity, 0) / cap_tot + cod * (1 - TAX) * debt / cap_tot) if cap_tot > 0 else coe
    s.roic_wacc = s.roic - s.wacc
    if equity > 0:
        s.roe = ni / equity
        if rev > 0 and ta > 0:
            s.dupont = {"margin": ni / rev, "turnover": rev / ta, "leverage": ta / equity}
    s.owner_earnings = ni + _n(d.get("depreciation")) - 0.7 * abs(_n(d.get("capex")))
    s.accrual_quality = fcf / ni if ni > 0 else (1.5 if fcf > 0 else 0.0)
    s.capital = capital_structure(d, equity_vol)

    iv = {}
    if s.graham_number > 0:
        iv["graham_number"] = s.graham_number
    if s.normalized_eps > 0 and s.wacc > 0:
        iv["earning_power"] = s.normalized_eps / s.wacc
    if s.normalized_eps > 0:
        iv["max_pe"] = s.normalized_eps * td.max_pe
    if s.ncav_per_share > 0:
        iv["ncav"] = s.ncav_per_share
    if fcf > 0:
        fps = fcf / shares
        disc, g = max(s.wacc, 0.08), 0.02
        iv["dcf"] = min(fps * (1 + g) / (disc - g), fps * 15)
    s.iv_methods = iv
    vals = sorted(iv.values())
    if vals:
        mid = len(vals) // 2
        s.intrinsic_value = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
    else:
        s.intrinsic_value = tbvps if tbvps > 0 else 0.0
    s.margin_of_safety = (s.intrinsic_value - price) / s.intrinsic_value if s.intrinsic_value > 0 else -1.0

    cov = s.capital.interest_coverage if s.capital.interest_coverage is not None else 0.0
    cr = s.capital.current_ratio if s.capital.current_ratio is not None else 0.0
    tests = [s.margin_of_safety >= 0.33, s.earnings_yield > td.treasury_10y, cr >= 2.0, s.eps_stability < 0.5,
             s.pe_x_pb <= 22.5, s.roic_wacc > 0, cov >= COVERAGE_MIN.get(sector_type, COVERAGE_MIN_DEFAULT),
             s.accrual_quality >= 0.8]
    s.tests_passed = sum(tests)
    rate = s.tests_passed / 8
    mos = s.margin_of_safety
    s.grade = ("STRONG_INVESTMENT" if mos >= 0.5 and rate >= 0.75 else "INVESTMENT" if mos >= 0.33 and rate >= 0.6
               else "BORDERLINE" if mos >= 0.15 and rate >= 0.4 else "SPECULATIVE" if mos >= 0 else "AVOID")
    s.composite = composite(s, td)
    s.quality_tier = next((t for lim, t in TIERS if s.composite >= lim), "G")
    gate(s)
    return s


def composite(s: Score, td: TopDown) -> float:
    """0–100: MoS 25 · earnings quality 20 · balance sheet 15 · ROIC 15 · coverage 10 · valuation 10 · top-down 5."""
    mos = _clip(s.margin_of_safety * 100 + 50, 0, 100)
    eq = 50 + (20 if s.eps_stability < 0.2 else 10 if s.eps_stability < 0.4 else -20 if 0.8 < s.eps_stability < 999 else 0)
    eq += 15 if s.accrual_quality >= 1 else 5 if s.accrual_quality >= 0.8 else -15 if s.accrual_quality < 0.5 else 0
    cr = s.capital.current_ratio or 0.0
    de = s.capital.debt_to_equity if s.capital.debt_to_equity is not None else 99
    bs = 50 + (20 if cr >= 2.5 else 10 if cr >= 2 else -25 if cr < 1 else 0) + (15 if s.ncav_per_share > 0 else 0)
    bs += 10 if de < 0.3 else -15 if de > 1.5 else 0
    sp = s.roic_wacc
    roic = 50 + (30 if sp > 0.10 else 20 if sp > 0.05 else 10 if sp > 0 else -25 if sp < -0.05 else 0)
    cov = s.capital.interest_coverage if s.capital.interest_coverage is not None else 0.0
    cv = 50 + (25 if cov >= 10 else 15 if cov >= 7 else 5 if cov >= 5 else -25 if cov < 2 else 0)
    if s.capital.altman_z is not None and s.capital.altman_z < 1.8:
        cv -= 15
    val = 50 + (25 if s.pe_x_pb <= 15 else 10 if s.pe_x_pb <= 22.5 else -25 if s.pe_x_pb > 45 else 0)
    val += 15 if s.earnings_yield > td.treasury_10y + 0.03 else 0
    tdv = {"DEEPLY_UNDERVALUED": 90, "UNDERVALUED": 70, "OVERVALUED": 30, "EXTREMELY_OVERVALUED": 10}.get(td.regime, 50)
    parts = [(0.25, mos), (0.20, eq), (0.15, bs), (0.15, roic), (0.10, cv), (0.10, val), (0.05, tdv)]
    return round(sum(w * _clip(x, 0, 100) for w, x in parts), 2)


def gate(s: Score) -> None:
    """Fundamentals gate (must pass). Sets gate_pass, gate_score in [0, 1] and reasons."""
    cs = s.capital
    fcf_neg = s.accrual_quality <= 0 and s.owner_earnings < 0
    equity_neg = (cs.debt_to_equity or 0) >= 99
    pd = cs.merton_pd
    cov = cs.interest_coverage
    if equity_neg and fcf_neg:
        s.reasons.append("negative equity and negative cash generation")
        s.gate_pass, s.gate_score = False, 0.0
        return
    if cs.sleeve == "DISTRESSED":
        ok = (s.ncav_discount >= 0 or s.margin_of_safety >= 0.33) and (pd is None or pd <= 0.25) and not fcf_neg
        s.gate_pass = ok
        s.gate_score = _clip(0.5 + s.margin_of_safety / 2, 0, 1) if ok else 0.0
        s.reasons.append("distressed recovery: " + ("asset cover / MoS ok" if ok else "no asset cover or PD > 25%"))
        return
    if cov is not None and cov < 1.5 and pd is not None and pd > 0.10:
        s.reasons.append(f"coverage {cov:.1f}x with PD {pd:.0%}")
        s.gate_pass, s.gate_score = False, 0.0
        return
    s.gate_score = _clip(s.composite / 100, 0, 1)
    s.gate_pass = s.composite >= GATE_MIN
    s.reasons.append(f"composite {s.composite:.0f} {'≥' if s.gate_pass else '<'} {GATE_MIN:.0f}, grade {s.grade}, "
                     f"credit {cs.credit_tier}")


def cap_structure_map(scores: Dict[str, Score]) -> Dict[str, dict]:
    """{ticker: {merton_pd, interest_coverage}} for rotation.is_solvent (LEAPS)."""
    return {t: {"merton_pd": s.capital.merton_pd, "interest_coverage": s.capital.interest_coverage}
            for t, s in scores.items()}
