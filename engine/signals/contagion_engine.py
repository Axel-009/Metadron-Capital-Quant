"""ContagionEngine — L3 Graph Topology Contagion Engine.

Models financial contagion and systemic risk propagation through a network
of 21 nodes representing major financial institutions (GSIBs), GICS sectors,
and macro asset classes. Pure adjacency-matrix graph — no NetworkX dependency.

Node universe:
    6  GSIB banks:   JPM, GS, MS, BAC, C, WFC
    11 GICS sectors: Energy, Materials, Industrials, ConsumerDiscretionary,
                     ConsumerStaples, HealthCare, Financials, InfoTech,
                     CommServices, Utilities, RealEstate
    4  Macro assets: UST, USD, HY_CREDIT, COMMODITIES

Shock scenarios:
    BANK_RUN, CREDIT_CRUNCH, SOVEREIGN_CRISIS, SECTOR_ROTATION,
    LIQUIDITY_FREEZE, COMMODITY_SHOCK, CORRELATION_SPIKE

Usage:
    from engine.signals.contagion_engine import ContagionEngine

    ce = ContagionEngine()
    result = ce.run_scenario("BANK_RUN")
    report = ce.format_contagion_report()
    risk   = ce.get_portfolio_contagion_risk([{"ticker": "XLF", "weight": 0.3}])
"""

import logging
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("metadron.signals.contagion")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class NodeType(str, Enum):
    """Classification of each node in the contagion graph."""
    BANK = "BANK"
    SECTOR = "SECTOR"
    ASSET = "ASSET"
    SOVEREIGN = "SOVEREIGN"


class ScenarioType(str, Enum):
    """Pre-built shock scenarios for systemic stress testing."""
    BANK_RUN = "BANK_RUN"
    CREDIT_CRUNCH = "CREDIT_CRUNCH"
    SOVEREIGN_CRISIS = "SOVEREIGN_CRISIS"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    LIQUIDITY_FREEZE = "LIQUIDITY_FREEZE"
    COMMODITY_SHOCK = "COMMODITY_SHOCK"
    CORRELATION_SPIKE = "CORRELATION_SPIKE"


# ---------------------------------------------------------------------------
# ContagionNode dataclass
# ---------------------------------------------------------------------------
@dataclass
class ContagionNode:
    """Single node in the contagion graph.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. 'JPM', 'InfoTech', 'UST').
    node_type : NodeType
        Classification — BANK, SECTOR, ASSET, or SOVEREIGN.
    health : float
        Current health in [0, 1].  1 = fully healthy, 0 = failed.
    connections : list
        Indices of connected nodes in the adjacency matrix.
    shock_sensitivity : float
        Multiplier on incoming shock magnitude (>1 = amplifier, <1 = damper).
    recovery_rate : float
        Per-step autonomous recovery toward health=1.
    """
    name: str
    node_type: NodeType
    health: float = 1.0
    connections: list = field(default_factory=list)
    shock_sensitivity: float = 1.0
    recovery_rate: float = 0.02

    def apply_shock(self, magnitude: float) -> float:
        """Apply a shock to this node, reducing its health.

        Parameters
        ----------
        magnitude : float
            Raw shock magnitude before sensitivity scaling.

        Returns
        -------
        float
            Actual health lost after clamping.
        """
        effective = magnitude * self.shock_sensitivity
        old_health = self.health
        self.health = max(0.0, min(1.0, self.health - effective))
        return old_health - self.health

    def recover(self) -> None:
        """Apply one step of autonomous recovery."""
        if self.health < 1.0:
            self.health = min(1.0, self.health + self.recovery_rate * (1.0 - self.health))

    def is_stressed(self) -> bool:
        """Return True if health is below 0.7 (stressed threshold)."""
        return self.health < 0.7

    def is_critical(self) -> bool:
        """Return True if health is below 0.4 (critical threshold)."""
        return self.health < 0.4

    def stress_level(self) -> str:
        """Return categorical stress level."""
        if self.health >= 0.85:
            return "HEALTHY"
        elif self.health >= 0.7:
            return "ELEVATED"
        elif self.health >= 0.4:
            return "STRESSED"
        elif self.health >= 0.15:
            return "CRITICAL"
        else:
            return "FAILED"


# ---------------------------------------------------------------------------
# Node definitions — canonical ordering (index 0..20)
# ---------------------------------------------------------------------------
NODE_DEFS: List[Tuple[str, NodeType, float, float]] = [
    # (name, type, shock_sensitivity, recovery_rate)
    # --- 6 GSIB banks (idx 0-5) ---
    ("JPM",       NodeType.BANK,      1.00, 0.03),
    ("GS",        NodeType.BANK,      1.20, 0.02),
    ("MS",        NodeType.BANK,      1.15, 0.02),
    ("BAC",       NodeType.BANK,      1.05, 0.03),
    ("C",         NodeType.BANK,      1.10, 0.02),
    ("WFC",       NodeType.BANK,      0.95, 0.03),
    # --- 11 GICS sectors (idx 6-16) ---
    ("Energy",              NodeType.SECTOR, 1.10, 0.04),
    ("Materials",           NodeType.SECTOR, 1.00, 0.04),
    ("Industrials",         NodeType.SECTOR, 0.95, 0.04),
    ("ConsDisc",            NodeType.SECTOR, 1.05, 0.03),
    ("ConsStaples",         NodeType.SECTOR, 0.60, 0.05),
    ("HealthCare",          NodeType.SECTOR, 0.55, 0.05),
    ("Financials",          NodeType.SECTOR, 1.25, 0.03),
    ("InfoTech",            NodeType.SECTOR, 1.15, 0.04),
    ("CommServices",        NodeType.SECTOR, 0.90, 0.04),
    ("Utilities",           NodeType.SECTOR, 0.50, 0.05),
    ("RealEstate",          NodeType.SECTOR, 1.10, 0.03),
    # --- 4 macro asset classes (idx 17-20) ---
    ("UST",         NodeType.SOVEREIGN, 0.70, 0.06),
    ("USD",         NodeType.ASSET,     0.65, 0.06),
    ("HY_CREDIT",   NodeType.ASSET,     1.30, 0.02),
    ("COMMODITIES", NodeType.ASSET,     1.05, 0.04),
]

N_NODES = len(NODE_DEFS)  # 21

# Map node name -> index for fast lookup
NODE_INDEX: Dict[str, int] = {nd[0]: i for i, nd in enumerate(NODE_DEFS)}

# Sector ETF -> node name mapping for portfolio risk
TICKER_TO_NODE: Dict[str, str] = {
    # Banks
    "JPM": "JPM", "GS": "GS", "MS": "MS", "BAC": "BAC", "C": "C", "WFC": "WFC",
    # Sector ETFs
    "XLE": "Energy",       "XLB": "Materials",    "XLI": "Industrials",
    "XLY": "ConsDisc",     "XLP": "ConsStaples",  "XLV": "HealthCare",
    "XLF": "Financials",   "XLK": "InfoTech",     "XLC": "CommServices",
    "XLU": "Utilities",    "XLRE": "RealEstate",
    # Macro
    "TLT": "UST", "IEF": "UST", "SHY": "UST",
    "UUP": "USD", "DXY": "USD",
    "HYG": "HY_CREDIT", "JNK": "HY_CREDIT", "LQD": "HY_CREDIT",
    "DBC": "COMMODITIES", "GSG": "COMMODITIES", "USO": "COMMODITIES",
    "GLD": "COMMODITIES", "SLV": "COMMODITIES",
    # Additional bank / financial tickers
    "BRK-B": "Financials", "AXP": "Financials", "SCHW": "Financials",
    # Tech mega-caps -> InfoTech
    "AAPL": "InfoTech", "MSFT": "InfoTech", "NVDA": "InfoTech",
    "GOOGL": "CommServices", "META": "CommServices", "AMZN": "ConsDisc",
    "TSLA": "ConsDisc",
    # Broad
    "SPY": "Financials",  # proxy — Financials as bellwether
    "QQQ": "InfoTech",
    "IWM": "ConsDisc",
}


# ---------------------------------------------------------------------------
# Correlation / adjacency matrix builder
# ---------------------------------------------------------------------------
def _build_base_adjacency() -> np.ndarray:
    """Build the 21x21 base adjacency matrix with realistic correlation-based
    edge weights.  Weights in [0, 1] — higher = stronger contagion channel.

    The matrix is symmetric.  Self-connections are zero.

    Correlation structure based on historical cross-asset behaviour:
        - Banks are tightly interconnected (interbank market)
        - Banks have strong link to Financials sector and HY credit
        - Cyclical sectors correlate with each other
        - Defensive sectors (Staples, Health, Utilities) are loosely coupled
        - UST is inversely correlated to risk (negative weight → 0 for
          contagion purposes; UST stress *does* propagate outward though)
        - HY_CREDIT is the nexus between banks and sectors
    """
    W = np.zeros((N_NODES, N_NODES), dtype=np.float64)

    def _sym(i: int, j: int, w: float) -> None:
        W[i, j] = w
        W[j, i] = w

    # Shorthand indices
    JPM, GS, MS, BAC, C, WFC = 0, 1, 2, 3, 4, 5
    ENE, MAT, IND, CD, CS, HC = 6, 7, 8, 9, 10, 11
    FIN, IT, COM, UTL, RE = 12, 13, 14, 15, 16
    UST_, USD_, HY, COMM = 17, 18, 19, 20

    # ------------------------------------------------------------------
    # 1. Interbank connections (dense, high weight)
    # ------------------------------------------------------------------
    bank_ids = [JPM, GS, MS, BAC, C, WFC]
    for i, b1 in enumerate(bank_ids):
        for b2 in bank_ids[i + 1:]:
            _sym(b1, b2, 0.75 + 0.05 * np.random.RandomState(b1 * 7 + b2).rand())

    # ------------------------------------------------------------------
    # 2. Banks → Financials sector (very strong)
    # ------------------------------------------------------------------
    for b in bank_ids:
        _sym(b, FIN, 0.85)

    # ------------------------------------------------------------------
    # 3. Banks → HY credit (strong — credit provision channel)
    # ------------------------------------------------------------------
    for b in bank_ids:
        _sym(b, HY, 0.70)

    # ------------------------------------------------------------------
    # 4. Banks → UST (moderate — balance sheet duration)
    # ------------------------------------------------------------------
    for b in bank_ids:
        _sym(b, UST_, 0.45)

    # ------------------------------------------------------------------
    # 5. Banks → Real Estate (mortgage exposure)
    # ------------------------------------------------------------------
    for b in bank_ids:
        _sym(b, RE, 0.55)

    # ------------------------------------------------------------------
    # 6. Cyclical sector block — cross-correlations
    # ------------------------------------------------------------------
    cyclicals = [ENE, MAT, IND, CD, FIN, RE]
    for i, s1 in enumerate(cyclicals):
        for s2 in cyclicals[i + 1:]:
            base = 0.40
            # Financials-RealEstate stronger
            if (s1 == FIN and s2 == RE) or (s1 == RE and s2 == FIN):
                base = 0.65
            _sym(s1, s2, base + 0.05 * np.random.RandomState(s1 * 11 + s2).rand())

    # ------------------------------------------------------------------
    # 7. Defensive sector block — weak cross-correlations
    # ------------------------------------------------------------------
    defensives = [CS, HC, UTL]
    for i, s1 in enumerate(defensives):
        for s2 in defensives[i + 1:]:
            _sym(s1, s2, 0.20)

    # ------------------------------------------------------------------
    # 8. Growth block — Tech / Comm Services
    # ------------------------------------------------------------------
    _sym(IT, COM, 0.55)
    _sym(IT, CD, 0.40)   # e-commerce overlap
    _sym(COM, CD, 0.35)

    # ------------------------------------------------------------------
    # 9. Sector → macro asset links
    # ------------------------------------------------------------------
    # HY_CREDIT → cyclicals
    for s in cyclicals:
        if W[HY, s] == 0:
            _sym(HY, s, 0.50)

    # HY_CREDIT → defensives (weaker)
    for s in defensives:
        _sym(HY, s, 0.20)

    # HY_CREDIT → Tech / Comm
    _sym(HY, IT, 0.35)
    _sym(HY, COM, 0.30)

    # UST → rate-sensitive sectors
    _sym(UST_, RE, 0.60)
    _sym(UST_, UTL, 0.50)
    _sym(UST_, FIN, 0.50)
    _sym(UST_, IT, 0.30)  # duration in growth

    # USD → export-sensitive
    _sym(USD_, IT, 0.35)   # multinational revenue
    _sym(USD_, ENE, 0.40)  # commodity pricing
    _sym(USD_, MAT, 0.40)
    _sym(USD_, IND, 0.30)
    _sym(USD_, COMM, 0.55) # dollar-denominated commodities

    # COMMODITIES → sectors
    _sym(COMM, ENE, 0.80)  # direct
    _sym(COMM, MAT, 0.65)
    _sym(COMM, IND, 0.35)
    _sym(COMM, CD, 0.25)   # input cost
    _sym(COMM, CS, 0.30)   # food/ag

    # UST ↔ USD (flight to quality link)
    _sym(UST_, USD_, 0.50)

    # UST ↔ HY (spread relationship)
    _sym(UST_, HY, 0.55)

    # COMMODITIES ↔ HY (energy issuers in HY index)
    _sym(COMM, HY, 0.45)

    # ------------------------------------------------------------------
    # 10. Ensure diagonal is zero
    # ------------------------------------------------------------------
    np.fill_diagonal(W, 0.0)

    return W


# ---------------------------------------------------------------------------
# ContagionGraph — pure adjacency-matrix graph
# ---------------------------------------------------------------------------
class ContagionGraph:
    """Directed weighted graph for financial contagion modelling.

    Uses a dense 21x21 numpy adjacency matrix — no NetworkX dependency.
    Edge weights represent contagion channel strength in [0, 1].
    """

    def __init__(self) -> None:
        # Build nodes
        self.nodes: List[ContagionNode] = []
        for name, ntype, sens, rec in NODE_DEFS:
            self.nodes.append(ContagionNode(
                name=name,
                node_type=ntype,
                health=1.0,
                connections=[],
                shock_sensitivity=sens,
                recovery_rate=rec,
            ))

        # Build adjacency
        self.adj: np.ndarray = _build_base_adjacency()

        # Populate connection lists from adjacency
        for i in range(N_NODES):
            self.nodes[i].connections = list(np.nonzero(self.adj[i])[0])

        # Track shock history for analysis
        self._shock_log: List[Dict] = []

    # ---------------------------------------------------------------
    # Core accessors
    # ---------------------------------------------------------------
    def node_index(self, name: str) -> int:
        """Return integer index for a node name."""
        return NODE_INDEX[name]

    def get_node(self, name: str) -> ContagionNode:
        """Retrieve node by name."""
        return self.nodes[NODE_INDEX[name]]

    def edge_weight(self, src: str, dst: str) -> float:
        """Return edge weight between two named nodes."""
        return float(self.adj[NODE_INDEX[src], NODE_INDEX[dst]])

    def get_neighbors(self, idx: int) -> List[int]:
        """Return list of neighbor indices (non-zero edges)."""
        return list(np.nonzero(self.adj[idx])[0])

    def health_vector(self) -> np.ndarray:
        """Return 1-D array of all node health values."""
        return np.array([n.health for n in self.nodes])

    def set_health_vector(self, h: np.ndarray) -> None:
        """Set all node health values from a 1-D array."""
        for i, node in enumerate(self.nodes):
            node.health = float(np.clip(h[i], 0.0, 1.0))

    # ---------------------------------------------------------------
    # Shock propagation — multi-step BFS with decay
    # ---------------------------------------------------------------
    def propagate_shock(
        self,
        source_node: str,
        shock_magnitude: float,
        steps: int = 5,
        decay: float = 0.55,
        recovery_enabled: bool = True,
    ) -> Dict:
        """Propagate a shock through the graph using multi-step BFS.

        At each step the shock is transmitted along edges, attenuated by
        edge weight * decay^step.  Nodes that have already been visited
        can be re-shocked on later steps (second-round effects).

        Parameters
        ----------
        source_node : str
            Name of the node where the shock originates.
        shock_magnitude : float
            Initial shock intensity (health points lost at source).
        steps : int
            Number of propagation rounds (BFS depth).
        decay : float
            Per-step multiplicative decay on transmitted shocks.
        recovery_enabled : bool
            Whether nodes recover between steps.

        Returns
        -------
        dict
            Keys: 'source', 'initial_shock', 'steps', 'health_before',
            'health_after', 'damage_by_node', 'total_damage',
            'nodes_stressed', 'nodes_critical'.
        """
        src_idx = NODE_INDEX[source_node]
        health_before = self.health_vector().copy()

        # Apply initial shock at source
        self.nodes[src_idx].apply_shock(shock_magnitude)

        # BFS propagation
        frontier = {src_idx: shock_magnitude}
        for step in range(steps):
            next_frontier: Dict[int, float] = {}
            for node_idx, incoming_shock in frontier.items():
                neighbors = self.get_neighbors(node_idx)
                for nb in neighbors:
                    w = self.adj[node_idx, nb]
                    transmitted = incoming_shock * w * (decay ** (step + 1))
                    if transmitted > 0.005:  # threshold to stop noise
                        actual_loss = self.nodes[nb].apply_shock(transmitted)
                        if actual_loss > 0.001:
                            # Accumulate — a node can receive from multiple parents
                            next_frontier[nb] = next_frontier.get(nb, 0) + transmitted

            # Autonomous recovery between steps
            if recovery_enabled:
                for node in self.nodes:
                    node.recover()

            frontier = next_frontier
            if not frontier:
                break

        health_after = self.health_vector()
        damage = health_before - health_after

        log_entry = {
            "source": source_node,
            "initial_shock": shock_magnitude,
            "steps": steps,
            "health_before": health_before,
            "health_after": health_after,
            "damage_by_node": {
                self.nodes[i].name: float(damage[i])
                for i in range(N_NODES) if damage[i] > 0.001
            },
            "total_damage": float(np.sum(damage)),
            "nodes_stressed": [n.name for n in self.nodes if n.is_stressed()],
            "nodes_critical": [n.name for n in self.nodes if n.is_critical()],
        }
        self._shock_log.append(log_entry)
        return log_entry

    # ---------------------------------------------------------------
    # Systemic risk score
    # ---------------------------------------------------------------
    def get_systemic_risk_score(self) -> float:
        """Compute an aggregate systemic risk score in [0, 1].

        Methodology:
            - Weighted average health deficit (1 - health)
            - Bank nodes get 2x weight (TBTF)
            - HY_CREDIT and Financials get 1.5x
            - Non-linear scaling: sigmoid to compress extremes

        Returns
        -------
        float
            Systemic risk score, 0 = healthy system, 1 = total collapse.
        """
        weights = np.ones(N_NODES)
        for i, node in enumerate(self.nodes):
            if node.node_type == NodeType.BANK:
                weights[i] = 2.0
            elif node.name in ("HY_CREDIT", "Financials"):
                weights[i] = 1.5
            elif node.node_type == NodeType.SOVEREIGN:
                weights[i] = 1.8

        deficits = 1.0 - self.health_vector()
        raw = np.average(deficits, weights=weights)

        # Amplify if many nodes are stressed simultaneously (correlation penalty)
        n_stressed = sum(1 for n in self.nodes if n.is_stressed())
        correlation_bonus = 0.0
        if n_stressed >= 5:
            correlation_bonus = 0.05 * (n_stressed - 4)
        if n_stressed >= 10:
            correlation_bonus += 0.10 * (n_stressed - 9)

        score = raw + correlation_bonus

        # Sigmoid compression into [0, 1]
        score = 1.0 / (1.0 + np.exp(-8.0 * (score - 0.35)))
        return float(np.clip(score, 0.0, 1.0))

    # ---------------------------------------------------------------
    # Vulnerability ranking
    # ---------------------------------------------------------------
    def get_vulnerability_ranking(self) -> List[Dict]:
        """Rank all nodes by vulnerability (most vulnerable first).

        Vulnerability = (1 - health) * shock_sensitivity * degree_centrality

        Returns
        -------
        list of dict
            Each dict has 'name', 'node_type', 'health', 'vulnerability',
            'stress_level', 'degree'.
        """
        degrees = np.count_nonzero(self.adj, axis=1).astype(float)
        max_deg = max(degrees.max(), 1.0)
        norm_degrees = degrees / max_deg

        ranking = []
        for i, node in enumerate(self.nodes):
            vuln = (1.0 - node.health) * node.shock_sensitivity * (0.5 + 0.5 * norm_degrees[i])
            ranking.append({
                "name": node.name,
                "node_type": node.node_type.value,
                "health": round(node.health, 4),
                "vulnerability": round(vuln, 4),
                "stress_level": node.stress_level(),
                "degree": int(degrees[i]),
            })

        ranking.sort(key=lambda x: x["vulnerability"], reverse=True)
        return ranking

    # ---------------------------------------------------------------
    # Eigenvector centrality (power iteration)
    # ---------------------------------------------------------------
    def eigenvector_centrality(self, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
        """Compute eigenvector centrality via power iteration.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        np.ndarray
            Centrality vector of length N_NODES, normalised to sum=1.
        """
        A = self.adj.copy()
        x = np.ones(N_NODES) / N_NODES
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new, 1)
            if norm < 1e-12:
                break
            x_new /= norm
            if np.max(np.abs(x_new - x)) < tol:
                x = x_new
                break
            x = x_new
        return x

    # ---------------------------------------------------------------
    # Betweenness proxy (shortest-path based)
    # ---------------------------------------------------------------
    def betweenness_proxy(self) -> np.ndarray:
        """Approximate betweenness centrality using all-pairs shortest paths.

        Converts edge weights to distances (dist = 1 / weight) and runs a
        simple Floyd-Warshall.  Counts fraction of shortest paths through
        each node.

        Returns
        -------
        np.ndarray
            Betweenness score per node, normalised to [0, 1].
        """
        # Distance matrix (inverse weights; inf where no edge)
        with np.errstate(divide="ignore"):
            dist = np.where(self.adj > 0, 1.0 / self.adj, np.inf)
        np.fill_diagonal(dist, 0.0)

        # Floyd-Warshall
        n = N_NODES
        nxt = np.full((n, n), -1, dtype=int)
        for i in range(n):
            for j in range(n):
                if dist[i, j] < np.inf and i != j:
                    nxt[i, j] = j

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        nxt[i, j] = nxt[i, k]

        # Count how often each node appears on shortest paths
        counts = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Trace path
                cur = i
                while cur != j and cur != -1:
                    nxt_node = nxt[cur, j]
                    if nxt_node == -1:
                        break
                    if nxt_node != j:
                        counts[nxt_node] += 1
                    cur = nxt_node

        mx = counts.max()
        if mx > 0:
            counts /= mx
        return counts

    # ---------------------------------------------------------------
    # Reset all nodes to full health
    # ---------------------------------------------------------------
    def reset(self) -> None:
        """Reset all nodes to health=1.0 and clear shock log."""
        for node in self.nodes:
            node.health = 1.0
        self._shock_log.clear()


# ---------------------------------------------------------------------------
# ShockScenario — pre-built stress scenarios
# ---------------------------------------------------------------------------
@dataclass
class ShockScenario:
    """Definition of a contagion stress scenario.

    Attributes
    ----------
    name : str
        Scenario identifier (matches ScenarioType enum).
    description : str
        Human-readable description of the scenario.
    shocks : list of tuple
        Each tuple is (node_name, shock_magnitude).
    adj_modifier : optional callable
        Function that temporarily modifies the adjacency matrix.
    steps : int
        Number of propagation steps.
    decay : float
        Per-step decay factor.
    """
    name: str
    description: str
    shocks: List[Tuple[str, float]]
    adj_modifier: Optional[object] = None  # callable
    steps: int = 5
    decay: float = 0.55


def _build_scenarios() -> Dict[str, ShockScenario]:
    """Construct the 7 pre-built shock scenarios."""
    scenarios: Dict[str, ShockScenario] = {}

    # ---------------------------------------------------------------
    # 1. BANK_RUN — simultaneous stress on all 6 GSIBs
    # ---------------------------------------------------------------
    scenarios["BANK_RUN"] = ShockScenario(
        name="BANK_RUN",
        description=(
            "Simultaneous confidence crisis across all 6 GSIBs. "
            "Interbank lending freezes, counterparty risk spikes. "
            "Modelled on Lehman-week dynamics."
        ),
        shocks=[
            ("JPM", 0.35), ("GS", 0.45), ("MS", 0.45),
            ("BAC", 0.40), ("C", 0.40), ("WFC", 0.35),
        ],
        steps=6,
        decay=0.50,
    )

    # ---------------------------------------------------------------
    # 2. CREDIT_CRUNCH — HY spread explosion
    # ---------------------------------------------------------------
    scenarios["CREDIT_CRUNCH"] = ShockScenario(
        name="CREDIT_CRUNCH",
        description=(
            "High-yield credit spreads blow out 300+ bps. "
            "Propagates to Financials, cyclical sectors, and banks. "
            "Modelled on 2015-16 energy credit crisis."
        ),
        shocks=[
            ("HY_CREDIT", 0.55),
            ("Financials", 0.20),
            ("Energy", 0.25),
        ],
        steps=5,
        decay=0.55,
    )

    # ---------------------------------------------------------------
    # 3. SOVEREIGN_CRISIS — UST selloff + dollar spike
    # ---------------------------------------------------------------
    scenarios["SOVEREIGN_CRISIS"] = ShockScenario(
        name="SOVEREIGN_CRISIS",
        description=(
            "UST selloff drives yields higher, USD spikes on safe-haven "
            "flows, EM debt contagion. Duration-exposed sectors hit. "
            "Modelled on 2013 Taper Tantrum / 2022 gilt crisis dynamics."
        ),
        shocks=[
            ("UST", 0.40),
            ("USD", 0.30),
            ("HY_CREDIT", 0.25),
            ("RealEstate", 0.20),
        ],
        steps=5,
        decay=0.50,
    )

    # ---------------------------------------------------------------
    # 4. SECTOR_ROTATION — tech crash propagating to growth
    # ---------------------------------------------------------------
    scenarios["SECTOR_ROTATION"] = ShockScenario(
        name="SECTOR_ROTATION",
        description=(
            "Tech sector crashes 20%+, dragging CommServices and "
            "ConsDisc (AMZN/TSLA). Growth-to-value rotation. "
            "Modelled on 2022 Nasdaq drawdown."
        ),
        shocks=[
            ("InfoTech", 0.45),
            ("CommServices", 0.35),
            ("ConsDisc", 0.25),
        ],
        steps=4,
        decay=0.55,
    )

    # ---------------------------------------------------------------
    # 5. LIQUIDITY_FREEZE — reserve drain + interbank stress
    # ---------------------------------------------------------------
    scenarios["LIQUIDITY_FREEZE"] = ShockScenario(
        name="LIQUIDITY_FREEZE",
        description=(
            "Fed reverse-repo drain + bank reserve crunch. Interbank "
            "overnight rates spike, money market stress. Treasury market "
            "liquidity evaporates.  Modelled on Sep 2019 repo spike."
        ),
        shocks=[
            ("JPM", 0.30), ("GS", 0.35), ("MS", 0.30),
            ("BAC", 0.25), ("C", 0.30), ("WFC", 0.25),
            ("UST", 0.25),
            ("HY_CREDIT", 0.20),
        ],
        steps=5,
        decay=0.50,
    )

    # ---------------------------------------------------------------
    # 6. COMMODITY_SHOCK — oil/commodity spike
    # ---------------------------------------------------------------
    scenarios["COMMODITY_SHOCK"] = ShockScenario(
        name="COMMODITY_SHOCK",
        description=(
            "Oil spikes 40%+ on supply disruption. Input costs surge, "
            "inflation expectations reprice, rate hike fears mount. "
            "Modelled on 2022 Russia/Ukraine commodity shock."
        ),
        shocks=[
            ("COMMODITIES", 0.50),
            ("Energy", 0.30),
            ("Materials", 0.25),
            ("ConsDisc", 0.15),
        ],
        steps=5,
        decay=0.55,
    )

    # ---------------------------------------------------------------
    # 7. CORRELATION_SPIKE — all correlations → 1 (2020-style)
    # ---------------------------------------------------------------
    def _correlation_spike_modifier(adj: np.ndarray) -> np.ndarray:
        """Temporarily push all non-zero edges toward 1.0."""
        modified = adj.copy()
        mask = modified > 0
        modified[mask] = 0.5 * modified[mask] + 0.5  # blend toward 1.0
        return modified

    scenarios["CORRELATION_SPIKE"] = ShockScenario(
        name="CORRELATION_SPIKE",
        description=(
            "Panic-driven correlation spike — all asset correlations "
            "converge toward 1.0. Diversification fails. Everything "
            "sells off simultaneously.  Modelled on March 2020 COVID crash."
        ),
        shocks=[
            ("Financials", 0.30),
            ("InfoTech", 0.25),
            ("ConsDisc", 0.25),
            ("Energy", 0.30),
            ("HY_CREDIT", 0.35),
            ("COMMODITIES", 0.20),
        ],
        adj_modifier=_correlation_spike_modifier,
        steps=6,
        decay=0.45,
    )

    return scenarios


# Pre-build the scenarios at module level
SCENARIOS = _build_scenarios()


# ---------------------------------------------------------------------------
# ContagionEngine — main orchestrator
# ---------------------------------------------------------------------------
class ContagionEngine:
    """L3 Graph Topology Contagion Engine.

    Builds and manages a 21-node financial contagion graph, runs pre-built
    shock scenarios, and provides portfolio-level contagion risk estimates.

    Usage
    -----
    >>> engine = ContagionEngine()
    >>> result = engine.run_scenario("BANK_RUN")
    >>> print(engine.format_contagion_report())
    """

    def __init__(self) -> None:
        self.graph = ContagionGraph()
        self.scenarios = SCENARIOS
        self._last_results: Dict[str, Dict] = {}
        self._systemic_history: List[float] = []

    # ---------------------------------------------------------------
    # Run single scenario
    # ---------------------------------------------------------------
    def run_scenario(self, scenario_name: str) -> Dict:
        """Run a named shock scenario and return detailed results.

        Parameters
        ----------
        scenario_name : str
            One of: BANK_RUN, CREDIT_CRUNCH, SOVEREIGN_CRISIS,
            SECTOR_ROTATION, LIQUIDITY_FREEZE, COMMODITY_SHOCK,
            CORRELATION_SPIKE.

        Returns
        -------
        dict
            Keys: 'scenario', 'description', 'propagation_results',
            'systemic_risk_score', 'vulnerability_ranking',
            'centrality', 'summary'.
        """
        scenario_name = scenario_name.upper().replace(" ", "_")
        if scenario_name not in self.scenarios:
            available = ", ".join(sorted(self.scenarios.keys()))
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. Available: {available}"
            )

        scenario = self.scenarios[scenario_name]
        self.graph.reset()

        # Apply adjacency modifier if present
        original_adj = None
        if scenario.adj_modifier is not None:
            original_adj = self.graph.adj.copy()
            self.graph.adj = scenario.adj_modifier(self.graph.adj)
            # Re-compute connections
            for i in range(N_NODES):
                self.graph.nodes[i].connections = list(
                    np.nonzero(self.graph.adj[i])[0]
                )

        # Apply each shock and propagate
        propagation_results = []
        for node_name, magnitude in scenario.shocks:
            result = self.graph.propagate_shock(
                source_node=node_name,
                shock_magnitude=magnitude,
                steps=scenario.steps,
                decay=scenario.decay,
            )
            propagation_results.append(result)

        # Compute metrics
        systemic_score = self.graph.get_systemic_risk_score()
        self._systemic_history.append(systemic_score)
        vuln_ranking = self.graph.get_vulnerability_ranking()
        centrality = self.graph.eigenvector_centrality()

        # Restore original adjacency if modified
        if original_adj is not None:
            self.graph.adj = original_adj
            for i in range(N_NODES):
                self.graph.nodes[i].connections = list(
                    np.nonzero(self.graph.adj[i])[0]
                )

        # Build summary
        n_stressed = sum(1 for n in self.graph.nodes if n.is_stressed())
        n_critical = sum(1 for n in self.graph.nodes if n.is_critical())
        total_damage = sum(r["total_damage"] for r in propagation_results)
        most_damaged = sorted(
            [(n.name, round(1.0 - n.health, 4)) for n in self.graph.nodes],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        summary = {
            "scenario": scenario_name,
            "systemic_risk": round(systemic_score, 4),
            "risk_level": _risk_level(systemic_score),
            "nodes_stressed": n_stressed,
            "nodes_critical": n_critical,
            "total_damage": round(total_damage, 4),
            "top_5_damaged": most_damaged,
            "health_snapshot": {
                n.name: round(n.health, 4) for n in self.graph.nodes
            },
        }

        output = {
            "scenario": scenario_name,
            "description": scenario.description,
            "propagation_results": propagation_results,
            "systemic_risk_score": systemic_score,
            "vulnerability_ranking": vuln_ranking,
            "centrality": {
                self.graph.nodes[i].name: round(float(centrality[i]), 4)
                for i in range(N_NODES)
            },
            "summary": summary,
        }

        self._last_results[scenario_name] = output
        return output

    # ---------------------------------------------------------------
    # Run all scenarios
    # ---------------------------------------------------------------
    def run_all_scenarios(self) -> Dict[str, Dict]:
        """Run all 7 pre-built scenarios and return aggregated results.

        Returns
        -------
        dict
            Keyed by scenario name, each value is the result from
            run_scenario().  Also includes an 'AGGREGATE' key with
            cross-scenario statistics.
        """
        results: Dict[str, Dict] = {}
        for name in self.scenarios:
            results[name] = self.run_scenario(name)

        # Aggregate statistics
        scores = [r["systemic_risk_score"] for r in results.values()]
        worst_scenario = max(results.keys(), key=lambda k: results[k]["systemic_risk_score"])

        # Which nodes are most frequently stressed across scenarios?
        stress_counts: Dict[str, int] = {}
        for r in results.values():
            for node_name in r["summary"]["health_snapshot"]:
                health = r["summary"]["health_snapshot"][node_name]
                if health < 0.7:
                    stress_counts[node_name] = stress_counts.get(node_name, 0) + 1

        chronically_stressed = sorted(
            stress_counts.items(), key=lambda x: x[1], reverse=True
        )

        results["AGGREGATE"] = {
            "mean_systemic_risk": round(float(np.mean(scores)), 4),
            "max_systemic_risk": round(float(np.max(scores)), 4),
            "min_systemic_risk": round(float(np.min(scores)), 4),
            "std_systemic_risk": round(float(np.std(scores)), 4),
            "worst_scenario": worst_scenario,
            "chronically_stressed_nodes": chronically_stressed[:7],
            "scenarios_above_50pct": sum(1 for s in scores if s > 0.5),
        }

        return results

    # ---------------------------------------------------------------
    # Portfolio contagion risk
    # ---------------------------------------------------------------
    def get_portfolio_contagion_risk(self, positions: List[Dict]) -> Dict:
        """Map a portfolio to the contagion graph and estimate exposure.

        Each position dict should have at minimum:
            {'ticker': str, 'weight': float}
        Optionally:
            {'ticker': str, 'weight': float, 'sector': str}

        Parameters
        ----------
        positions : list of dict
            Portfolio positions.

        Returns
        -------
        dict
            Keys: 'mapped_exposure', 'unmapped_tickers', 'contagion_beta',
            'scenario_pnl_estimates', 'risk_summary'.
        """
        # Map tickers to graph nodes
        mapped: Dict[str, float] = {}
        unmapped: List[str] = []

        for pos in positions:
            ticker = pos.get("ticker", "").upper()
            weight = pos.get("weight", 0.0)
            sector = pos.get("sector", None)

            node_name = TICKER_TO_NODE.get(ticker)
            if node_name is None and sector is not None:
                # Try sector mapping
                sector_map = {
                    "Energy": "Energy", "Materials": "Materials",
                    "Industrials": "Industrials",
                    "Consumer Discretionary": "ConsDisc",
                    "Consumer Staples": "ConsStaples",
                    "Health Care": "HealthCare",
                    "Financials": "Financials",
                    "Information Technology": "InfoTech",
                    "Communication Services": "CommServices",
                    "Utilities": "Utilities",
                    "Real Estate": "RealEstate",
                }
                node_name = sector_map.get(sector)

            if node_name is not None:
                mapped[node_name] = mapped.get(node_name, 0.0) + weight
            else:
                unmapped.append(ticker)

        # Compute contagion beta — portfolio sensitivity to systemic shock
        # Run a small uniform shock and measure portfolio-weighted damage
        self.graph.reset()
        # Apply uniform 10% shock to all nodes
        for node in self.graph.nodes:
            node.apply_shock(0.10)

        # Propagate from the most central node
        centrality = self.graph.eigenvector_centrality()
        central_idx = int(np.argmax(centrality))
        self.graph.propagate_shock(
            self.graph.nodes[central_idx].name,
            shock_magnitude=0.15,
            steps=3,
            decay=0.50,
        )

        health_after = self.graph.health_vector()
        damage = 1.0 - health_after

        portfolio_damage = 0.0
        total_mapped_weight = sum(mapped.values())
        for node_name, weight in mapped.items():
            idx = NODE_INDEX[node_name]
            portfolio_damage += weight * damage[idx]

        contagion_beta = portfolio_damage / max(total_mapped_weight, 0.01)

        # Scenario P&L estimates
        scenario_pnl: Dict[str, float] = {}
        for scenario_name in self.scenarios:
            result = self.run_scenario(scenario_name)
            scenario_damage = 0.0
            for node_name, weight in mapped.items():
                health = result["summary"]["health_snapshot"].get(node_name, 1.0)
                scenario_damage += weight * (1.0 - health)
            # Convert health damage to approximate P&L impact
            # Assume 1 health point ≈ 15% drawdown
            scenario_pnl[scenario_name] = round(-scenario_damage * 0.15, 4)

        # Concentration risk
        node_weights = sorted(mapped.items(), key=lambda x: x[1], reverse=True)
        hhi = sum(w ** 2 for _, w in node_weights) / max(total_mapped_weight ** 2, 0.01)

        risk_summary = {
            "contagion_beta": round(contagion_beta, 4),
            "concentration_hhi": round(hhi, 4),
            "concentration_level": (
                "HIGH" if hhi > 0.25 else "MODERATE" if hhi > 0.10 else "LOW"
            ),
            "total_mapped_weight": round(total_mapped_weight, 4),
            "unmapped_weight": round(1.0 - total_mapped_weight, 4),
            "worst_scenario": min(scenario_pnl, key=scenario_pnl.get),
            "worst_scenario_pnl": min(scenario_pnl.values()),
            "risk_level": _risk_level(contagion_beta),
        }

        return {
            "mapped_exposure": mapped,
            "unmapped_tickers": unmapped,
            "contagion_beta": round(contagion_beta, 4),
            "scenario_pnl_estimates": scenario_pnl,
            "node_weights": node_weights,
            "risk_summary": risk_summary,
        }

    # ---------------------------------------------------------------
    # ASCII contagion report
    # ---------------------------------------------------------------
    def format_contagion_report(self) -> str:
        """Generate an ASCII report of the current systemic risk state.

        Returns
        -------
        str
            Multi-section ASCII report suitable for terminal display.
        """
        lines: List[str] = []
        W = 78  # report width

        def _bar(health: float, width: int = 20) -> str:
            filled = int(health * width)
            return "[" + "#" * filled + "." * (width - filled) + "]"

        def _hdr(title: str) -> None:
            lines.append("")
            lines.append("=" * W)
            lines.append(f"  {title}")
            lines.append("=" * W)

        def _sub(title: str) -> None:
            lines.append("")
            lines.append(f"  --- {title} ---")

        # Title
        lines.append("")
        lines.append("+" + "-" * (W - 2) + "+")
        lines.append("|" + " METADRON CAPITAL — L3 CONTAGION ENGINE REPORT ".center(W - 2) + "|")
        lines.append("+" + "-" * (W - 2) + "+")

        # Systemic Risk Score
        _hdr("SYSTEMIC RISK OVERVIEW")
        score = self.graph.get_systemic_risk_score()
        level = _risk_level(score)
        lines.append(f"  Systemic Risk Score:  {score:.4f}  [{level}]")
        lines.append(f"  Risk Gauge:           {_bar(score, 40)}")
        lines.append("")

        n_healthy = sum(1 for n in self.graph.nodes if n.stress_level() == "HEALTHY")
        n_elevated = sum(1 for n in self.graph.nodes if n.stress_level() == "ELEVATED")
        n_stressed = sum(1 for n in self.graph.nodes if n.stress_level() == "STRESSED")
        n_critical = sum(1 for n in self.graph.nodes if n.stress_level() == "CRITICAL")
        n_failed = sum(1 for n in self.graph.nodes if n.stress_level() == "FAILED")

        lines.append(f"  Healthy: {n_healthy:2d}  |  Elevated: {n_elevated:2d}  |  "
                      f"Stressed: {n_stressed:2d}  |  Critical: {n_critical:2d}  |  "
                      f"Failed: {n_failed:2d}")

        # Node Health Table
        _hdr("NODE HEALTH STATUS")
        lines.append(f"  {'Node':<18} {'Type':<10} {'Health':>7}  {'Bar':<22} {'Status':<10}")
        lines.append("  " + "-" * 70)

        for node in sorted(self.graph.nodes, key=lambda n: n.health):
            status = node.stress_level()
            flag = " *" if node.is_critical() else ""
            lines.append(
                f"  {node.name:<18} {node.node_type.value:<10} "
                f"{node.health:>6.3f}  {_bar(node.health)}  {status:<10}{flag}"
            )

        # Centrality Analysis
        _hdr("CENTRALITY ANALYSIS")
        centrality = self.graph.eigenvector_centrality()
        betweenness = self.graph.betweenness_proxy()

        lines.append(f"  {'Node':<18} {'Eigenvec':>10} {'Betweenness':>12} {'Degree':>8}")
        lines.append("  " + "-" * 50)

        # Sort by eigenvector centrality
        indices = np.argsort(-centrality)
        for idx in indices:
            node = self.graph.nodes[idx]
            deg = len(node.connections)
            lines.append(
                f"  {node.name:<18} {centrality[idx]:>10.4f} "
                f"{betweenness[idx]:>12.4f} {deg:>8d}"
            )

        # Vulnerability Ranking
        _hdr("VULNERABILITY RANKING (Top 10)")
        vuln = self.graph.get_vulnerability_ranking()
        lines.append(f"  {'Rank':<6} {'Node':<18} {'Vuln Score':>10} {'Health':>8} {'Status':<10}")
        lines.append("  " + "-" * 55)
        for i, v in enumerate(vuln[:10]):
            lines.append(
                f"  {i+1:<6} {v['name']:<18} {v['vulnerability']:>10.4f} "
                f"{v['health']:>8.4f} {v['stress_level']:<10}"
            )

        # Scenario Results (if available)
        if self._last_results:
            _hdr("SCENARIO RESULTS SUMMARY")
            lines.append(
                f"  {'Scenario':<22} {'Risk Score':>10} {'Level':<12} "
                f"{'Stressed':>8} {'Critical':>8} {'Damage':>8}"
            )
            lines.append("  " + "-" * 72)

            for name, result in sorted(self._last_results.items()):
                s = result.get("summary", {})
                if not s:
                    continue
                lines.append(
                    f"  {name:<22} {s.get('systemic_risk', 0):>10.4f} "
                    f"{s.get('risk_level', 'N/A'):<12} "
                    f"{s.get('nodes_stressed', 0):>8} "
                    f"{s.get('nodes_critical', 0):>8} "
                    f"{s.get('total_damage', 0):>8.3f}"
                )

        # Adjacency Matrix Stats
        _hdr("ADJACENCY MATRIX STATISTICS")
        adj = self.graph.adj
        n_edges = np.count_nonzero(adj) // 2  # undirected
        density = n_edges / (N_NODES * (N_NODES - 1) / 2)
        mean_w = adj[adj > 0].mean() if np.any(adj > 0) else 0.0
        max_w = adj.max()
        min_w = adj[adj > 0].min() if np.any(adj > 0) else 0.0

        lines.append(f"  Nodes:            {N_NODES}")
        lines.append(f"  Edges:            {n_edges}")
        lines.append(f"  Density:          {density:.4f}")
        lines.append(f"  Mean edge weight: {mean_w:.4f}")
        lines.append(f"  Max edge weight:  {max_w:.4f}")
        lines.append(f"  Min edge weight:  {min_w:.4f}")

        # Cluster coefficients (local)
        _sub("Local Clustering Coefficients")
        for i, node in enumerate(self.graph.nodes):
            neighbors = node.connections
            k = len(neighbors)
            if k < 2:
                cc = 0.0
            else:
                triangles = 0
                for a_idx in range(k):
                    for b_idx in range(a_idx + 1, k):
                        na, nb = neighbors[a_idx], neighbors[b_idx]
                        if adj[na, nb] > 0:
                            triangles += 1
                cc = 2.0 * triangles / (k * (k - 1))
            if cc > 0.5:
                lines.append(f"    {node.name:<18} clustering = {cc:.3f}")

        # Footer
        lines.append("")
        lines.append("+" + "-" * (W - 2) + "+")
        lines.append("|" + " END OF CONTAGION REPORT ".center(W - 2) + "|")
        lines.append("+" + "-" * (W - 2) + "+")
        lines.append("")

        return "\n".join(lines)

    # ---------------------------------------------------------------
    # Convenience: quick systemic check
    # ---------------------------------------------------------------
    def systemic_risk_score(self) -> float:
        """Return current systemic risk score."""
        return self.graph.get_systemic_risk_score()

    def reset(self) -> None:
        """Reset the entire engine state."""
        self.graph.reset()
        self._last_results.clear()
        self._systemic_history.clear()

    def node_health(self, name: str) -> float:
        """Return current health of a named node."""
        return self.graph.get_node(name).health

    def all_node_health(self) -> Dict[str, float]:
        """Return dict of all node health values."""
        return {n.name: round(n.health, 4) for n in self.graph.nodes}

    # ---------------------------------------------------------------
    # Stress test: Monte Carlo
    # ---------------------------------------------------------------
    def monte_carlo_stress(
        self,
        n_simulations: int = 500,
        rng_seed: int = 42,
    ) -> Dict:
        """Run Monte Carlo stress simulations with random shocks.

        Each simulation picks 1-4 random nodes, applies random shocks
        in [0.1, 0.6], and records the systemic risk score.

        Parameters
        ----------
        n_simulations : int
            Number of random scenarios to simulate.
        rng_seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Keys: 'scores', 'mean', 'std', 'percentiles', 'tail_risk',
            'worst_sim'.
        """
        rng = np.random.RandomState(rng_seed)
        scores = np.zeros(n_simulations)
        worst_score = 0.0
        worst_config = None

        node_names = [n.name for n in self.graph.nodes]

        for sim in range(n_simulations):
            self.graph.reset()

            n_shocks = rng.randint(1, 5)
            shock_nodes = rng.choice(len(node_names), size=n_shocks, replace=False)
            shock_mags = rng.uniform(0.10, 0.60, size=n_shocks)

            config = []
            for idx, mag in zip(shock_nodes, shock_mags):
                name = node_names[idx]
                self.graph.propagate_shock(name, mag, steps=4, decay=0.50)
                config.append((name, float(mag)))

            score = self.graph.get_systemic_risk_score()
            scores[sim] = score

            if score > worst_score:
                worst_score = score
                worst_config = config

        percentiles = {
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
        }

        # Tail risk: probability of systemic score > 0.7
        tail_risk = float(np.mean(scores > 0.7))

        return {
            "n_simulations": n_simulations,
            "scores": scores,
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "percentiles": percentiles,
            "tail_risk": round(tail_risk, 4),
            "worst_score": round(worst_score, 4),
            "worst_config": worst_config,
        }

    # ---------------------------------------------------------------
    # Cascade depth analysis
    # ---------------------------------------------------------------
    def cascade_depth_analysis(self) -> Dict[str, Dict]:
        """For each node, measure how deep a shock propagates.

        Applies a standardised 0.3 shock to each node in isolation
        and measures cascade reach and depth.

        Returns
        -------
        dict
            Keyed by node name.  Each value has 'cascade_reach' (number
            of nodes affected), 'cascade_depth' (max damage), 'total_damage'.
        """
        results: Dict[str, Dict] = {}
        shock_mag = 0.30

        for node in self.graph.nodes:
            self.graph.reset()
            prop = self.graph.propagate_shock(
                node.name, shock_mag, steps=5, decay=0.55
            )
            n_affected = len(prop["damage_by_node"])
            max_damage = max(prop["damage_by_node"].values()) if prop["damage_by_node"] else 0.0

            results[node.name] = {
                "cascade_reach": n_affected,
                "cascade_depth": round(max_damage, 4),
                "total_damage": round(prop["total_damage"], 4),
                "affected_nodes": list(prop["damage_by_node"].keys()),
            }

        self.graph.reset()
        return results

    # ---------------------------------------------------------------
    # Adjacency matrix export (for visualization)
    # ---------------------------------------------------------------
    def get_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return the raw adjacency matrix and node labels.

        Returns
        -------
        tuple
            (21x21 np.ndarray, list of 21 node name strings)
        """
        labels = [n.name for n in self.graph.nodes]
        return self.graph.adj.copy(), labels

    # ---------------------------------------------------------------
    # Spectral risk (eigenvalue-based)
    # ---------------------------------------------------------------
    def spectral_risk_indicator(self) -> Dict:
        """Compute spectral risk indicator from adjacency eigenvalues.

        The largest eigenvalue of the adjacency matrix (spectral radius)
        indicates the amplification potential of the network.  A higher
        spectral radius means shocks can amplify more through the network.

        Returns
        -------
        dict
            'spectral_radius', 'spectral_gap', 'amplification_factor',
            'eigenvalues'.
        """
        eigenvalues = np.linalg.eigvalsh(self.graph.adj)
        eigenvalues = np.sort(eigenvalues)[::-1]

        spectral_radius = float(eigenvalues[0])
        spectral_gap = float(eigenvalues[0] - eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

        # Amplification factor: how much a unit shock can amplify
        # through the network (Leontief-style)
        amplification = 1.0 / max(1.0 - spectral_radius / N_NODES, 0.01)

        return {
            "spectral_radius": round(spectral_radius, 4),
            "spectral_gap": round(spectral_gap, 4),
            "amplification_factor": round(amplification, 4),
            "top_5_eigenvalues": [round(float(e), 4) for e in eigenvalues[:5]],
            "risk_interpretation": (
                "HIGH — network amplifies shocks significantly"
                if spectral_radius > 5.0
                else "MODERATE — network has some amplification potential"
                if spectral_radius > 3.0
                else "LOW — network dampens shocks"
            ),
        }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _risk_level(score: float) -> str:
    """Convert a [0,1] risk score to a categorical level."""
    if score >= 0.80:
        return "CRITICAL"
    elif score >= 0.60:
        return "HIGH"
    elif score >= 0.40:
        return "ELEVATED"
    elif score >= 0.20:
        return "MODERATE"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ContagionEngine — self-test")
    print("-" * 50)

    engine = ContagionEngine()

    # Validate graph structure
    assert len(engine.graph.nodes) == 21, f"Expected 21 nodes, got {len(engine.graph.nodes)}"
    assert engine.graph.adj.shape == (21, 21), f"Adj shape: {engine.graph.adj.shape}"
    assert np.allclose(engine.graph.adj, engine.graph.adj.T), "Adjacency not symmetric"
    assert np.all(np.diag(engine.graph.adj) == 0), "Self-loops detected"
    print(f"  Graph: {N_NODES} nodes, "
          f"{np.count_nonzero(engine.graph.adj) // 2} edges")

    # Test single scenario
    result = engine.run_scenario("BANK_RUN")
    print(f"  BANK_RUN systemic risk: {result['systemic_risk_score']:.4f}")
    assert 0 <= result["systemic_risk_score"] <= 1

    # Test all scenarios
    all_results = engine.run_all_scenarios()
    assert "AGGREGATE" in all_results
    print(f"  All scenarios: mean risk = {all_results['AGGREGATE']['mean_systemic_risk']:.4f}")
    print(f"  Worst scenario: {all_results['AGGREGATE']['worst_scenario']}")

    # Test portfolio risk
    portfolio = [
        {"ticker": "XLF", "weight": 0.25},
        {"ticker": "XLK", "weight": 0.30},
        {"ticker": "JPM", "weight": 0.15},
        {"ticker": "TLT", "weight": 0.10},
        {"ticker": "HYG", "weight": 0.10},
        {"ticker": "XLE", "weight": 0.10},
    ]
    prisk = engine.get_portfolio_contagion_risk(portfolio)
    print(f"  Portfolio contagion beta: {prisk['contagion_beta']:.4f}")
    print(f"  Worst scenario for portfolio: {prisk['risk_summary']['worst_scenario']}")

    # Spectral analysis
    spectral = engine.spectral_risk_indicator()
    print(f"  Spectral radius: {spectral['spectral_radius']:.4f}")
    print(f"  Amplification: {spectral['amplification_factor']:.2f}x")

    # Cascade depth
    cascades = engine.cascade_depth_analysis()
    deepest = max(cascades.items(), key=lambda x: x[1]["total_damage"])
    print(f"  Deepest cascade source: {deepest[0]} "
          f"(damage={deepest[1]['total_damage']:.4f}, "
          f"reach={deepest[1]['cascade_reach']})")

    # Monte Carlo
    mc = engine.monte_carlo_stress(n_simulations=200)
    print(f"  MC stress: mean={mc['mean']:.4f}, p95={mc['percentiles']['p95']:.4f}, "
          f"tail_risk={mc['tail_risk']:.4f}")

    # Report
    engine.reset()
    engine.run_all_scenarios()
    report = engine.format_contagion_report()
    print(f"\n  Report length: {len(report)} chars, {report.count(chr(10))} lines")
    print()
    print(report)

    print("\n  All self-tests passed.")
