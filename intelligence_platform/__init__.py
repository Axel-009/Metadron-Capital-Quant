"""Intelligence Platform -- Complete Reference Repository Collection

28 repositories + agent skills consolidated into Metadron Capital's unified intelligence platform.
All files from every source repo are included (Python, Go, Rust, C++, CUDA, TS, Java).

Layer Architecture:
    L1 Data:       Financial-Data, open-bb, hedgefund-tracker, FRB, EquityLinkedGICPooling, Quant-Developers-Resources
    L2 Signals:    Mav-Analysis, quant-trading, stock-chain, CTA-code, TradeTheEvent, wondertrader
    L3 ML:         QLIB, Stock-techincal-prediction-model, Stock-prediction, ML-Macro-Market, AI-Newton
    L4 Portfolio:  ai-hedgefund, financial-distressed-repo, sophisticated-distress-analysis, FinancialDistressPrediction
    L5 Infra:      Kserve, nividia-repo, Air-LLM, exchange-core
    L6 Agents:     Ruflo-agents, MiroFish
    Agent Skills:  Claude Skills API, Files API, agent patterns (from claude-cookbooks)
"""

from pathlib import Path

PLATFORM_ROOT = Path(__file__).parent

# Registry of all 28 repos by layer
REPOS = {
    # Layer 1: Data Ingestion
    "Financial-Data":              {"layer": 1, "role": "Market Data Pipeline"},
    "open-bb":                     {"layer": 1, "role": "Investment Research Terminal"},
    "hedgefund-tracker":           {"layer": 1, "role": "Institutional Flow Intelligence"},
    "FRB":                         {"layer": 1, "role": "Federal Reserve Bank Data"},
    "EquityLinkedGICPooling":      {"layer": 1, "role": "GIC Pooling Methodology"},
    "Quant-Developers-Resources":  {"layer": 1, "role": "Quantitative Finance Reference"},
    # Layer 2: Signal Processing
    "Mav-Analysis":                {"layer": 2, "role": "Technical Analysis Engine"},
    "quant-trading":               {"layer": 2, "role": "Quantitative Strategy Library"},
    "stock-chain":                 {"layer": 2, "role": "Time-Series Chain Analysis"},
    "CTA-code":                    {"layer": 2, "role": "CTA/Trend-Following Signals"},
    "TradeTheEvent":               {"layer": 2, "role": "Event-Driven ML"},
    "wondertrader":                {"layer": 2, "role": "HFT Quantitative Trading"},
    # Layer 3: ML/AI Models
    "QLIB":                        {"layer": 3, "role": "Quantitative ML Framework"},
    "Stock-techincal-prediction-model": {"layer": 3, "role": "Price Prediction Models"},
    "Stock-prediction":            {"layer": 3, "role": "Technical Price Prediction"},
    "ML-Macro-Market":             {"layer": 3, "role": "Macro-Market ML Bridge"},
    "AI-Newton":                   {"layer": 3, "role": "Physics-Inspired Financial Models"},
    # Layer 4: Portfolio & Risk
    "ai-hedgefund":                {"layer": 4, "role": "AI Hedge Fund Core"},
    "financial-distressed-repo":   {"layer": 4, "role": "Credit Risk / Distress Detection"},
    "sophisticated-distress-analysis": {"layer": 4, "role": "Advanced Distress Analytics"},
    "FinancialDistressPrediction": {"layer": 4, "role": "GBM Distress Prediction"},
    # Layer 5: Infrastructure
    "Kserve":                      {"layer": 5, "role": "Model Serving Platform"},
    "nividia-repo":                {"layer": 5, "role": "GPU-Accelerated Deep Learning"},
    "Air-LLM":                     {"layer": 5, "role": "Efficient LLM Inference"},
    "exchange-core":               {"layer": 5, "role": "High-Performance Exchange Engine"},
    # Layer 6: Agents
    "Ruflo-agents":                {"layer": 6, "role": "Multi-Agent Orchestration"},
    "MiroFish":                    {"layer": 6, "role": "Social Prediction Engine"},
    "get-shit-done":               {"layer": 6, "role": "GSD Meta-Prompting & Workflow Orchestration"},
}

# Agent Skills integration (Claude Skills API + Files API)
AGENT_SKILLS = {
    "skills": ["analyzing-financial-statements", "creating-financial-models"],
    "agent_patterns": ["orchestrator_workers", "evaluator_optimizer"],
    "source": "axel-009/claude-cookbooks",
}


def get_repo_path(repo_name: str) -> Path:
    """Get the absolute path to a repo directory."""
    return PLATFORM_ROOT / repo_name


def get_layer_repos(layer: int) -> dict[str, dict]:
    """Get all repos in a specific layer."""
    return {k: v for k, v in REPOS.items() if v["layer"] == layer}


def verify_repos() -> dict[str, bool]:
    """Verify all repos exist on disk."""
    return {name: (PLATFORM_ROOT / name).is_dir() for name in REPOS}


def get_agent_skills_path() -> Path:
    """Get the absolute path to the agent skills integration."""
    return PLATFORM_ROOT / "agent_skills"
