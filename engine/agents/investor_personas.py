"""Investor Persona Bridge — 12 Investor Personas + 8 Core Analysis Agents.

Integrates the ai-hedgefund persona agents and core analysis agents into the
Metadron Capital engine.  Each persona is wrapped in try/except so the system
runs degraded (neutral fallback) if the upstream agents are unavailable or if
LLM calls fail.  Rule-based fallbacks capture each persona's core investment
philosophy using fundamental/technical data already available in the state.

Signal mapping to Metadron SignalType:
    bullish  -> ML_AGENT_BUY
    bearish  -> ML_AGENT_SELL
    neutral  -> HOLD

Consensus thresholds:
    >60% agreement -> STRONG signal
    >50% agreement -> MODERATE signal
    else           -> NEUTRAL
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- agent_skills integration -------------------------------------------------
try:
    from intelligence_platform.agent_skills import (
        create_skill, list_custom_skills, test_skill,
        extract_file_ids, download_file, download_all_files,
    )
    AGENT_SKILLS_AVAILABLE = True
except ImportError:
    AGENT_SKILLS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metadron SignalType — local fallback so this module never crashes
# ---------------------------------------------------------------------------
try:
    from ..execution.paper_broker import SignalType
except ImportError:
    class SignalType(str, Enum):
        ML_AGENT_BUY = "ML_AGENT_BUY"
        ML_AGENT_SELL = "ML_AGENT_SELL"
        HOLD = "HOLD"

# ---------------------------------------------------------------------------
# Upstream agent imports (all wrapped)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # engine/agents/../../.. = repo root
_INTELLIGENCE_BASE = str(_REPO_ROOT / "intelligence_platform" / "ai-hedgefund" / "src" / "agents")

_PERSONA_NAMES: List[str] = [
    "warren_buffett", "charlie_munger", "ben_graham", "peter_lynch",
    "phil_fisher", "michael_burry", "cathie_wood", "bill_ackman",
    "rakesh_jhunjhunwala", "stanley_druckenmiller", "mohnish_pabrai",
    "aswath_damodaran",
]

_CORE_AGENT_NAMES: List[str] = [
    "fundamentals", "growth_agent", "valuation", "sentiment",
    "news_sentiment", "technicals", "portfolio_manager", "risk_manager",
]

# Map module name -> expected callable name inside that module
_AGENT_FUNC_MAP: Dict[str, str] = {
    "warren_buffett": "warren_buffett_agent",
    "charlie_munger": "charlie_munger_agent",
    "ben_graham": "ben_graham_agent",
    "peter_lynch": "peter_lynch_agent",
    "phil_fisher": "phil_fisher_agent",
    "michael_burry": "michael_burry_agent",
    "cathie_wood": "cathie_wood_agent",
    "bill_ackman": "bill_ackman_agent",
    "rakesh_jhunjhunwala": "rakesh_jhunjhunwala_agent",
    "stanley_druckenmiller": "stanley_druckenmiller_agent",
    "mohnish_pabrai": "mohnish_pabrai_agent",
    "aswath_damodaran": "aswath_damodaran_agent",
    "fundamentals": "fundamentals_analyst_agent",
    "growth_agent": "growth_agent",
    "valuation": "valuation_agent",
    "sentiment": "sentiment_agent",
    "news_sentiment": "news_sentiment_agent",
    "technicals": "technical_analyst_agent",
    "portfolio_manager": "portfolio_management_agent",
    "risk_manager": "risk_management_agent",
}

# Elite personas receive 1.5x weight in voting
_ELITE_PERSONAS = frozenset({
    "warren_buffett", "charlie_munger", "ben_graham",
    "stanley_druckenmiller", "aswath_damodaran",
})


def _safe_get(d: dict, *keys, default=None):
    """Nested dict safe-get."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ============================================================================
# Rule-Based Persona Fallbacks
# ============================================================================

def _fb_warren_buffett(state: dict) -> Dict[str, dict]:
    """High ROE, low debt, competitive moat, margin of safety."""
    return _score_tickers(state, _eval_buffett)


def _eval_buffett(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    roe = metrics.get("return_on_equity")
    de = metrics.get("debt_to_equity")
    om = metrics.get("operating_margin")
    pe = metrics.get("price_to_earnings_ratio")
    if roe and roe > 0.15:
        score += 2
    if de is not None and de < 0.5:
        score += 2
    if om and om > 0.15:
        score += 1
    if pe and pe < 20:
        score += 1
    if score >= 4:
        return "bullish", min(60 + score * 5, 95), "Strong ROE, low debt, moat indicators"
    elif score <= 1:
        return "bearish", 55, "Weak fundamentals by Buffett criteria"
    return "neutral", 50, "Mixed Buffett signals"


def _fb_charlie_munger(state: dict) -> Dict[str, dict]:
    """Quality at fair price — high ROIC, durable business, reasonable valuation."""
    return _score_tickers(state, _eval_munger)


def _eval_munger(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    roic = metrics.get("return_on_invested_capital") or metrics.get("return_on_equity")
    om = metrics.get("operating_margin")
    pe = metrics.get("price_to_earnings_ratio")
    if roic and roic > 0.15:
        score += 2
    if om and om > 0.20:
        score += 2
    if pe and 10 < pe < 25:
        score += 1
    if score >= 4:
        return "bullish", 70, "High quality at fair price"
    elif score <= 1:
        return "bearish", 55, "Lacks Munger quality criteria"
    return "neutral", 50, "Mixed quality signals"


def _fb_ben_graham(state: dict) -> Dict[str, dict]:
    """Low P/E, high current ratio, net-net, margin of safety."""
    return _score_tickers(state, _eval_graham)


def _eval_graham(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    pe = metrics.get("price_to_earnings_ratio")
    pb = metrics.get("price_to_book_ratio")
    cr = metrics.get("current_ratio")
    de = metrics.get("debt_to_equity")
    if pe and pe < 15:
        score += 2
    if pb and pb < 1.5:
        score += 2
    if cr and cr > 2.0:
        score += 1
    if de is not None and de < 0.5:
        score += 1
    if score >= 4:
        return "bullish", 75, "Classic Graham net-net value"
    elif score <= 1:
        return "bearish", 55, "Fails Graham safety screens"
    return "neutral", 50, "Partial Graham value"


def _fb_peter_lynch(state: dict) -> Dict[str, dict]:
    """PEG ratio, earnings growth, story stock with numbers to back it."""
    return _score_tickers(state, _eval_lynch)


def _eval_lynch(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    pe = metrics.get("price_to_earnings_ratio")
    eg = metrics.get("earnings_growth")
    rg = metrics.get("revenue_growth")
    if pe and eg and eg > 0:
        peg = pe / (eg * 100)
        if peg < 1.0:
            score += 3
        elif peg < 1.5:
            score += 1
    if rg and rg > 0.15:
        score += 2
    if eg and eg > 0.20:
        score += 1
    if score >= 4:
        return "bullish", 70, "Low PEG with strong growth — Lynch pick"
    elif score <= 1:
        return "bearish", 50, "High PEG or weak growth"
    return "neutral", 50, "Moderate Lynch signals"


def _fb_phil_fisher(state: dict) -> Dict[str, dict]:
    """Growth quality — R&D, margins expanding, management excellence."""
    return _score_tickers(state, _eval_fisher)


def _eval_fisher(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    om = metrics.get("operating_margin")
    rg = metrics.get("revenue_growth")
    eg = metrics.get("earnings_growth")
    if om and om > 0.20:
        score += 2
    if rg and rg > 0.10:
        score += 1
    if eg and eg > 0.15:
        score += 2
    if score >= 4:
        return "bullish", 65, "Fisher growth quality — expanding margins + growth"
    elif score <= 1:
        return "bearish", 50, "Weak growth quality"
    return "neutral", 50, "Mixed Fisher signals"


def _fb_michael_burry(state: dict) -> Dict[str, dict]:
    """Deep value contrarian — oversold, low P/B, tangible asset backing."""
    return _score_tickers(state, _eval_burry)


def _eval_burry(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    pb = metrics.get("price_to_book_ratio")
    pe = metrics.get("price_to_earnings_ratio")
    de = metrics.get("debt_to_equity")
    fcf = metrics.get("free_cash_flow_per_share")
    eps = metrics.get("earnings_per_share")
    if pb and pb < 1.0:
        score += 3
    if pe and pe < 10:
        score += 2
    if de is not None and de < 1.0:
        score += 1
    if fcf and eps and fcf > eps:
        score += 1
    if score >= 4:
        return "bullish", 70, "Deep value — Burry contrarian opportunity"
    elif score <= 1:
        return "bearish", 50, "Not deep enough value for Burry"
    return "neutral", 50, "Moderate deep value"


def _fb_cathie_wood(state: dict) -> Dict[str, dict]:
    """Disruptive innovation — high revenue growth, expanding TAM."""
    return _score_tickers(state, _eval_wood)


def _eval_wood(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    rg = metrics.get("revenue_growth")
    eg = metrics.get("earnings_growth")
    ps = metrics.get("price_to_sales_ratio")
    if rg and rg > 0.25:
        score += 3
    elif rg and rg > 0.15:
        score += 1
    if eg and eg > 0.30:
        score += 2
    # Cathie is willing to pay high multiples for growth
    if ps and ps > 10 and rg and rg > 0.30:
        score += 1  # premium acceptable for hyper-growth
    if score >= 4:
        return "bullish", 65, "Disruptive growth — ARK-style conviction"
    elif score <= 1:
        return "bearish", 50, "Insufficient disruption signal"
    return "neutral", 50, "Moderate innovation signal"


def _fb_bill_ackman(state: dict) -> Dict[str, dict]:
    """Activist value — strong brand, fixable problems, high FCF."""
    return _score_tickers(state, _eval_ackman)


def _eval_ackman(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    om = metrics.get("operating_margin")
    fcf = metrics.get("free_cash_flow_per_share")
    eps = metrics.get("earnings_per_share")
    pe = metrics.get("price_to_earnings_ratio")
    roe = metrics.get("return_on_equity")
    if om and om > 0.15:
        score += 1
    if fcf and eps and eps > 0 and fcf / eps > 0.8:
        score += 2
    if pe and pe < 20:
        score += 1
    if roe and roe > 0.12:
        score += 1
    if score >= 4:
        return "bullish", 65, "Ackman activist target — strong FCF + fixable"
    elif score <= 1:
        return "bearish", 50, "Weak activist case"
    return "neutral", 50, "Moderate activist potential"


def _fb_rakesh_jhunjhunwala(state: dict) -> Dict[str, dict]:
    """India bull — growth at reasonable price, strong earnings momentum."""
    return _score_tickers(state, _eval_jhunjhunwala)


def _eval_jhunjhunwala(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    eg = metrics.get("earnings_growth")
    rg = metrics.get("revenue_growth")
    pe = metrics.get("price_to_earnings_ratio")
    roe = metrics.get("return_on_equity")
    if eg and eg > 0.15:
        score += 2
    if rg and rg > 0.12:
        score += 1
    if pe and pe < 25:
        score += 1
    if roe and roe > 0.15:
        score += 1
    if score >= 4:
        return "bullish", 65, "Jhunjhunwala GARP — strong growth + reasonable price"
    elif score <= 1:
        return "bearish", 50, "Fails growth-at-reasonable-price"
    return "neutral", 50, "Mixed GARP signals"


def _fb_stanley_druckenmiller(state: dict) -> Dict[str, dict]:
    """Macro-driven momentum — strong trend + earnings acceleration."""
    return _score_tickers(state, _eval_druckenmiller)


def _eval_druckenmiller(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    eg = metrics.get("earnings_growth")
    rg = metrics.get("revenue_growth")
    om = metrics.get("operating_margin")
    pe = metrics.get("price_to_earnings_ratio")
    if eg and eg > 0.20:
        score += 2
    if rg and rg > 0.15:
        score += 2
    if om and om > 0.15:
        score += 1
    # Druckenmiller will pay up for accelerating growth
    if pe and pe < 30 and eg and eg > 0.25:
        score += 1
    if score >= 4:
        return "bullish", 70, "Druckenmiller momentum — accelerating earnings"
    elif score <= 1:
        return "bearish", 55, "No momentum catalyst"
    return "neutral", 50, "Mixed macro-momentum"


def _fb_mohnish_pabrai(state: dict) -> Dict[str, dict]:
    """Dhandho — heads I win, tails I don't lose much. Low downside, high upside."""
    return _score_tickers(state, _eval_pabrai)


def _eval_pabrai(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    pe = metrics.get("price_to_earnings_ratio")
    pb = metrics.get("price_to_book_ratio")
    de = metrics.get("debt_to_equity")
    roe = metrics.get("return_on_equity")
    if pe and pe < 15:
        score += 2
    if pb and pb < 1.5:
        score += 1
    if de is not None and de < 0.3:
        score += 2
    if roe and roe > 0.15:
        score += 1
    if score >= 4:
        return "bullish", 70, "Pabrai dhandho — low risk, high reward asymmetry"
    elif score <= 1:
        return "bearish", 50, "Risk-reward unfavorable"
    return "neutral", 50, "Moderate dhandho potential"


def _fb_aswath_damodaran(state: dict) -> Dict[str, dict]:
    """Valuation professor — intrinsic value via DCF, compare to market price."""
    return _score_tickers(state, _eval_damodaran)


def _eval_damodaran(metrics: dict) -> Tuple[str, int, str]:
    score = 0
    pe = metrics.get("price_to_earnings_ratio")
    pb = metrics.get("price_to_book_ratio")
    ps = metrics.get("price_to_sales_ratio")
    roe = metrics.get("return_on_equity")
    eg = metrics.get("earnings_growth")
    if pe and pe < 18:
        score += 1
    if pb and pb < 3.0:
        score += 1
    if ps and ps < 5.0:
        score += 1
    if roe and roe > 0.12:
        score += 1
    if eg and eg > 0.08:
        score += 1
    if score >= 4:
        return "bullish", 65, "Damodaran DCF suggests undervaluation"
    elif score <= 1:
        return "bearish", 60, "Overvalued by intrinsic metrics"
    return "neutral", 50, "Fair value range — Damodaran neutral"


# ---------------------------------------------------------------------------
# Core analysis fallbacks
# ---------------------------------------------------------------------------

def _fb_fundamentals(state: dict) -> Dict[str, dict]:
    """Profitability + growth + health + valuation composite."""
    return _score_tickers(state, _eval_fundamentals)


def _eval_fundamentals(m: dict) -> Tuple[str, int, str]:
    bullish = 0
    bearish = 0
    if m.get("return_on_equity") and m["return_on_equity"] > 0.15:
        bullish += 1
    elif m.get("return_on_equity") and m["return_on_equity"] < 0.05:
        bearish += 1
    if m.get("debt_to_equity") is not None and m["debt_to_equity"] < 0.5:
        bullish += 1
    elif m.get("debt_to_equity") and m["debt_to_equity"] > 2.0:
        bearish += 1
    if m.get("current_ratio") and m["current_ratio"] > 1.5:
        bullish += 1
    if m.get("price_to_earnings_ratio") and m["price_to_earnings_ratio"] < 20:
        bullish += 1
    elif m.get("price_to_earnings_ratio") and m["price_to_earnings_ratio"] > 35:
        bearish += 1
    if bullish > bearish + 1:
        return "bullish", 65, "Fundamental strength across metrics"
    elif bearish > bullish:
        return "bearish", 55, "Fundamental weakness"
    return "neutral", 50, "Mixed fundamentals"


def _fb_growth_agent(state: dict) -> Dict[str, dict]:
    return _score_tickers(state, _eval_growth)


def _eval_growth(m: dict) -> Tuple[str, int, str]:
    score = 0
    if m.get("revenue_growth") and m["revenue_growth"] > 0.15:
        score += 2
    if m.get("earnings_growth") and m["earnings_growth"] > 0.15:
        score += 2
    if score >= 3:
        return "bullish", 65, "Strong growth trajectory"
    elif score == 0:
        return "bearish", 55, "Stagnant or declining growth"
    return "neutral", 50, "Moderate growth"


def _fb_valuation(state: dict) -> Dict[str, dict]:
    return _score_tickers(state, _eval_valuation)


def _eval_valuation(m: dict) -> Tuple[str, int, str]:
    score = 0
    if m.get("price_to_earnings_ratio") and m["price_to_earnings_ratio"] < 15:
        score += 2
    if m.get("price_to_book_ratio") and m["price_to_book_ratio"] < 2:
        score += 1
    if m.get("price_to_sales_ratio") and m["price_to_sales_ratio"] < 3:
        score += 1
    if score >= 3:
        return "bullish", 65, "Attractively valued"
    elif score == 0:
        return "bearish", 55, "Overvalued on multiple metrics"
    return "neutral", 50, "Fair valuation"


def _fb_sentiment(state: dict) -> Dict[str, dict]:
    """Placeholder — sentiment requires external data."""
    return _neutral_for_tickers(state, "No sentiment data in fallback mode")


def _fb_news_sentiment(state: dict) -> Dict[str, dict]:
    return _neutral_for_tickers(state, "No news sentiment data in fallback mode")


def _fb_technicals(state: dict) -> Dict[str, dict]:
    """Rule-based technical fallback — needs price data in state."""
    return _neutral_for_tickers(state, "Technical analysis requires price data")


def _fb_portfolio_manager(state: dict) -> Dict[str, dict]:
    return _neutral_for_tickers(state, "Portfolio manager fallback — pass-through")


def _fb_risk_manager(state: dict) -> Dict[str, dict]:
    return _neutral_for_tickers(state, "Risk manager fallback — conservative neutral")


# ---------------------------------------------------------------------------
# Helper: extract tickers and metrics from state, apply eval function
# ---------------------------------------------------------------------------

def _get_tickers(state: dict) -> List[str]:
    """Extract ticker list from an AgentState-like dict."""
    data = state.get("data", state)
    return data.get("tickers", [])


def _get_metrics_for_ticker(state: dict, ticker: str) -> dict:
    """Try to pull fundamental metrics from state for a ticker.

    The ai-hedgefund agents store metrics in state['data']['analyst_signals']
    and fetch live data.  For the fallback, we look for pre-populated metrics
    in state['data']['metrics'][ticker] or return empty dict.
    """
    data = state.get("data", state)
    # Try several common locations
    metrics = _safe_get(data, "metrics", ticker) or {}
    if not metrics:
        metrics = _safe_get(data, "financial_metrics", ticker) or {}
    if not metrics:
        # Try to pull from analyst_signals if fundamentals already ran
        fund_signals = _safe_get(data, "analyst_signals", "fundamentals_analyst_agent", ticker)
        if fund_signals and isinstance(fund_signals, dict):
            metrics = fund_signals.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _score_tickers(
    state: dict,
    eval_fn: Callable[[dict], Tuple[str, int, str]],
) -> Dict[str, dict]:
    """Apply eval_fn to each ticker's metrics, return {ticker: {signal, confidence, reasoning}}."""
    results = {}
    for ticker in _get_tickers(state):
        metrics = _get_metrics_for_ticker(state, ticker)
        if metrics:
            signal, confidence, reasoning = eval_fn(metrics)
        else:
            signal, confidence, reasoning = "neutral", 40, "Insufficient data for analysis"
        results[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    return results


def _neutral_for_tickers(state: dict, reason: str) -> Dict[str, dict]:
    """Return neutral signal for all tickers."""
    return {
        ticker: {"signal": "neutral", "confidence": 40, "reasoning": reason}
        for ticker in _get_tickers(state)
    }


# ---------------------------------------------------------------------------
# Fallback registry
# ---------------------------------------------------------------------------

_PERSONA_FALLBACKS: Dict[str, Callable] = {
    "warren_buffett": _fb_warren_buffett,
    "charlie_munger": _fb_charlie_munger,
    "ben_graham": _fb_ben_graham,
    "peter_lynch": _fb_peter_lynch,
    "phil_fisher": _fb_phil_fisher,
    "michael_burry": _fb_michael_burry,
    "cathie_wood": _fb_cathie_wood,
    "bill_ackman": _fb_bill_ackman,
    "rakesh_jhunjhunwala": _fb_rakesh_jhunjhunwala,
    "stanley_druckenmiller": _fb_stanley_druckenmiller,
    "mohnish_pabrai": _fb_mohnish_pabrai,
    "aswath_damodaran": _fb_aswath_damodaran,
}

_CORE_FALLBACKS: Dict[str, Callable] = {
    "fundamentals": _fb_fundamentals,
    "growth_agent": _fb_growth_agent,
    "valuation": _fb_valuation,
    "sentiment": _fb_sentiment,
    "news_sentiment": _fb_news_sentiment,
    "technicals": _fb_technicals,
    "portfolio_manager": _fb_portfolio_manager,
    "risk_manager": _fb_risk_manager,
}


# ============================================================================
# InvestorPersonaManager
# ============================================================================

class InvestorPersonaManager:
    """Manages 12 investor persona agents + 8 core analysis agents.

    On init, attempts to import each agent from the intelligence_platform copy.
    If unavailable, falls back to simplified rule-based evaluators that mimic
    each persona's investment philosophy.
    """

    def __init__(self):
        self._persona_agents: Dict[str, Optional[Callable]] = {}
        self._core_agents: Dict[str, Optional[Callable]] = {}
        self._available_personas: List[str] = []
        self._available_cores: List[str] = []
        self._load_agents()
        logger.info(
            "InvestorPersonaManager ready: %d/%d personas, %d/%d core agents live",
            len(self._available_personas), len(_PERSONA_NAMES),
            len(self._available_cores), len(_CORE_AGENT_NAMES),
        )

    # ------------------------------------------------------------------
    # Import logic
    # ------------------------------------------------------------------

    def _load_agents(self):
        """Try importing each agent; store None on failure (fallback used)."""
        import importlib
        import sys

        # Ensure intelligence_platform path is importable
        ip_root = str(_REPO_ROOT / "intelligence_platform" / "ai-hedgefund")
        if ip_root not in sys.path:
            sys.path.insert(0, ip_root)

        all_names = _PERSONA_NAMES + _CORE_AGENT_NAMES
        for name in all_names:
            func_name = _AGENT_FUNC_MAP.get(name, f"{name}_agent")
            agent_fn = None
            try:
                mod = importlib.import_module(f"src.agents.{name}")
                agent_fn = getattr(mod, func_name, None)
                if agent_fn is not None:
                    if name in _PERSONA_NAMES:
                        self._available_personas.append(name)
                    else:
                        self._available_cores.append(name)
                    logger.debug("Loaded agent: %s.%s", name, func_name)
            except Exception as exc:
                logger.debug("Agent %s unavailable (%s), will use fallback", name, exc)

            if name in _PERSONA_NAMES:
                self._persona_agents[name] = agent_fn
            else:
                self._core_agents[name] = agent_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_persona(self, persona_name: str, state: dict) -> dict:
        """Run a single persona agent.  Returns {ticker: {signal, confidence, reasoning}}.

        Falls back to rule-based analysis if the upstream agent is not available
        or if the LLM call fails.
        """
        if persona_name not in _PERSONA_NAMES:
            raise ValueError(f"Unknown persona: {persona_name}")

        agent_fn = self._persona_agents.get(persona_name)
        agent_id = _AGENT_FUNC_MAP.get(persona_name, f"{persona_name}_agent")

        # Try upstream agent first
        if agent_fn is not None:
            try:
                result = agent_fn(state, agent_id=agent_id)
                signals = _safe_get(result, "data", "analyst_signals", agent_id)
                if signals:
                    return signals
            except Exception as exc:
                logger.warning("Persona %s LLM call failed (%s), using fallback", persona_name, exc)

        # Fallback to rule-based
        fallback = _PERSONA_FALLBACKS.get(persona_name)
        if fallback:
            return fallback(state)
        return _neutral_for_tickers(state, f"No fallback for {persona_name}")

    def run_all_personas(self, state: dict) -> dict:
        """Run all 12 persona agents.  Returns {persona_name: {ticker: {signal, confidence, reasoning}}}."""
        results = {}
        for name in _PERSONA_NAMES:
            try:
                results[name] = self.run_persona(name, state)
            except Exception as exc:
                logger.error("Persona %s error: %s", name, exc)
                results[name] = _neutral_for_tickers(state, f"Error: {exc}")
        return results

    def run_core_analysis(self, state: dict) -> dict:
        """Run all 8 core analysis agents.  Returns {agent_name: {ticker: {signal, confidence, reasoning}}}."""
        results = {}
        for name in _CORE_AGENT_NAMES:
            agent_fn = self._core_agents.get(name)
            agent_id = _AGENT_FUNC_MAP.get(name, f"{name}_agent")

            if agent_fn is not None:
                try:
                    result = agent_fn(state, agent_id=agent_id)
                    signals = _safe_get(result, "data", "analyst_signals", agent_id)
                    if signals:
                        results[name] = signals
                        continue
                except Exception as exc:
                    logger.warning("Core agent %s failed (%s), using fallback", name, exc)

            fallback = _CORE_FALLBACKS.get(name)
            if fallback:
                results[name] = fallback(state)
            else:
                results[name] = _neutral_for_tickers(state, f"No fallback for {name}")
        return results

    def aggregate_signals(
        self,
        persona_results: Dict[str, Dict[str, dict]],
        analysis_results: Dict[str, Dict[str, dict]],
    ) -> Dict[str, dict]:
        """Aggregate all signals into a consensus per ticker.

        Weighted voting:
            - ELITE personas (Buffett, Munger, Graham, Druckenmiller, Damodaran) get 1.5x weight
            - All other agents get 1.0x weight
            - Confidence is used as an additional weight multiplier

        Returns::

            {
                ticker: {
                    "signal": SignalType,
                    "confidence": float (0-100),
                    "consensus": "STRONG_BULLISH" | "MODERATE_BULLISH" | ... | "NEUTRAL",
                    "bull_count": int,
                    "bear_count": int,
                    "neutral_count": int,
                    "agents": {agent_name: {signal, confidence, reasoning}},
                }
            }
        """
        # Collect all tickers across all results
        tickers: set = set()
        for agent_results in list(persona_results.values()) + list(analysis_results.values()):
            if isinstance(agent_results, dict):
                tickers.update(agent_results.keys())

        aggregated: Dict[str, dict] = {}

        for ticker in sorted(tickers):
            bull_weight = 0.0
            bear_weight = 0.0
            neutral_weight = 0.0
            bull_count = 0
            bear_count = 0
            neutral_count = 0
            agents_detail: Dict[str, dict] = {}
            total_weight = 0.0

            all_sources = []
            for name, res in persona_results.items():
                if isinstance(res, dict) and ticker in res:
                    weight = 1.5 if name in _ELITE_PERSONAS else 1.0
                    all_sources.append((name, res[ticker], weight))
            for name, res in analysis_results.items():
                if isinstance(res, dict) and ticker in res:
                    all_sources.append((name, res[ticker], 1.0))

            for agent_name, signal_data, base_weight in all_sources:
                sig = signal_data.get("signal", "neutral")
                conf = signal_data.get("confidence", 50)
                conf_weight = conf / 100.0
                effective_weight = base_weight * conf_weight

                if sig == "bullish":
                    bull_weight += effective_weight
                    bull_count += 1
                elif sig == "bearish":
                    bear_weight += effective_weight
                    bear_count += 1
                else:
                    neutral_weight += effective_weight
                    neutral_count += 1

                total_weight += effective_weight
                agents_detail[agent_name] = signal_data

            # Determine consensus
            total_votes = bull_count + bear_count + neutral_count
            if total_votes == 0:
                aggregated[ticker] = {
                    "signal": SignalType.HOLD,
                    "confidence": 0,
                    "consensus": "NO_DATA",
                    "bull_count": 0,
                    "bear_count": 0,
                    "neutral_count": 0,
                    "agents": {},
                }
                continue

            bull_pct = bull_weight / total_weight if total_weight > 0 else 0
            bear_pct = bear_weight / total_weight if total_weight > 0 else 0

            if bull_pct > 0.60:
                signal = SignalType.ML_AGENT_BUY
                consensus = "STRONG_BULLISH"
                confidence = bull_pct * 100
            elif bull_pct > 0.50:
                signal = SignalType.ML_AGENT_BUY
                consensus = "MODERATE_BULLISH"
                confidence = bull_pct * 100
            elif bear_pct > 0.60:
                signal = SignalType.ML_AGENT_SELL
                consensus = "STRONG_BEARISH"
                confidence = bear_pct * 100
            elif bear_pct > 0.50:
                signal = SignalType.ML_AGENT_SELL
                consensus = "MODERATE_BEARISH"
                confidence = bear_pct * 100
            else:
                signal = SignalType.HOLD
                consensus = "NEUTRAL"
                confidence = max(bull_pct, bear_pct, 1 - bull_pct - bear_pct) * 100

            aggregated[ticker] = {
                "signal": signal,
                "confidence": round(confidence, 1),
                "consensus": consensus,
                "bull_count": bull_count,
                "bear_count": bear_count,
                "neutral_count": neutral_count,
                "agents": agents_detail,
            }

        return aggregated

    def format_consensus_report(self, aggregated: Dict[str, dict]) -> str:
        """Format aggregated signals into an ASCII report.

        Returns a human-readable table suitable for logging or terminal display.
        """
        lines: List[str] = []
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        lines.append("")
        lines.append("=" * 82)
        lines.append(f"  INVESTOR PERSONA CONSENSUS REPORT  |  {ts}")
        lines.append("=" * 82)
        lines.append("")

        # Header
        lines.append(
            f"  {'Ticker':<8} {'Signal':<16} {'Consensus':<20} "
            f"{'Conf':>5}  {'Bull':>4} {'Bear':>4} {'Ntrl':>4}"
        )
        lines.append("  " + "-" * 74)

        for ticker in sorted(aggregated.keys()):
            data = aggregated[ticker]
            sig = data.get("signal", "HOLD")
            sig_str = sig if isinstance(sig, str) else sig.value
            cons = data.get("consensus", "NEUTRAL")
            conf = data.get("confidence", 0)
            bc = data.get("bull_count", 0)
            brc = data.get("bear_count", 0)
            nc = data.get("neutral_count", 0)

            lines.append(
                f"  {ticker:<8} {sig_str:<16} {cons:<20} "
                f"{conf:>5.1f}  {bc:>4} {brc:>4} {nc:>4}"
            )

        lines.append("  " + "-" * 74)
        lines.append("")

        # Agent breakdown per ticker
        for ticker in sorted(aggregated.keys()):
            agents = aggregated[ticker].get("agents", {})
            if not agents:
                continue
            lines.append(f"  --- {ticker} Agent Breakdown ---")
            lines.append(f"  {'Agent':<28} {'Signal':<10} {'Conf':>5}  {'Reasoning'}")
            lines.append("  " + "-" * 74)
            for agent_name in sorted(agents.keys()):
                ad = agents[agent_name]
                sig = ad.get("signal", "neutral")
                conf = ad.get("confidence", 0)
                reason = ad.get("reasoning", "")
                if isinstance(reason, dict):
                    reason = str(reason)[:40]
                else:
                    reason = str(reason)[:40]
                elite_mark = " *" if agent_name in _ELITE_PERSONAS else "  "
                lines.append(
                    f"  {agent_name:<26}{elite_mark} {sig:<10} {conf:>5}  {reason}"
                )
            lines.append("")

        lines.append("  * = ELITE persona (1.5x vote weight)")
        lines.append("  Consensus: >60% = STRONG, >50% = MODERATE, else NEUTRAL")
        lines.append("=" * 82)
        lines.append("")

        return "\n".join(lines)

    def run_persona_skill(self, persona_name: str, state: dict) -> dict:
        """Run an analysis skill scoped to a specific persona if available.

        Falls back to an empty dict when agent_skills is not installed.
        """
        if not AGENT_SKILLS_AVAILABLE:
            return {}
        try:
            return test_skill(
                "analyzing-financial-statements",
                {"persona": persona_name, "tickers": _get_tickers(state)},
            )
        except Exception as exc:
            logger.debug("Skill run failed for persona %s: %s", persona_name, exc)
            return {}

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(self, state: dict) -> Tuple[Dict[str, dict], str]:
        """Run all personas + core analysis, aggregate, and format report.

        Returns (aggregated_signals, ascii_report).
        """
        persona_results = self.run_all_personas(state)
        core_results = self.run_core_analysis(state)
        aggregated = self.aggregate_signals(persona_results, core_results)
        report = self.format_consensus_report(aggregated)
        return aggregated, report

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def persona_names(self) -> List[str]:
        return list(_PERSONA_NAMES)

    @property
    def core_agent_names(self) -> List[str]:
        return list(_CORE_AGENT_NAMES)

    @property
    def available_personas(self) -> List[str]:
        return list(self._available_personas)

    @property
    def available_core_agents(self) -> List[str]:
        return list(self._available_cores)

    @property
    def elite_personas(self) -> frozenset:
        return _ELITE_PERSONAS

    def status(self) -> dict:
        """Return a status summary of all agents."""
        return {
            "personas": {
                name: "live" if self._persona_agents.get(name) else "fallback"
                for name in _PERSONA_NAMES
            },
            "core_agents": {
                name: "live" if self._core_agents.get(name) else "fallback"
                for name in _CORE_AGENT_NAMES
            },
            "live_persona_count": len(self._available_personas),
            "live_core_count": len(self._available_cores),
            "total_agents": len(_PERSONA_NAMES) + len(_CORE_AGENT_NAMES),
        }
