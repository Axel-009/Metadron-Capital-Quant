"""
Deep Learning Engine — Pure-Numpy PPO Agent with Gym-like Trading Environment.

Implements Proximal Policy Optimization (PPO) for algorithmic trading using
only numpy. Includes a full Gym-like trading environment with a 50-feature
state vector and walk-forward validation.

Signals feed into the AlphaOptimizer (universe merge + dedup + cap
enforcement) and the 4-gate decision matrix (FUNDAMENTALS 40%,
FLOW_HEADLINES 20%, MACRO_REGIME 20%, MOMENTUM 20%).
Execution routed via IBKR; paper trade log for reconciliation only.

No torch, tensorflow, or gym dependency required.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("metadron.intelligence.deep_learning")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def _tanh_deriv(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh given *output* of tanh."""
    return 1.0 - x ** 2


def _he_init(rows: int, cols: int) -> np.ndarray:
    """He (Kaiming) weight initialisation."""
    return np.random.randn(rows, cols) * np.sqrt(2.0 / rows)


# ---------------------------------------------------------------------------
# Ensemble Advisor Metrics
# ---------------------------------------------------------------------------

@dataclass
class EnsembleAdvisorMetrics:
    """Track ensemble advisor call statistics."""
    calls_made: int = 0
    calls_succeeded: int = 0
    calls_failed: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.calls_succeeded == 0:
            return 0.0
        return self.total_latency_ms / self.calls_succeeded


# ---------------------------------------------------------------------------
# TradingEnvironment
# ---------------------------------------------------------------------------

class TradingEnvironment:
    """Gym-like trading environment with a 50-feature observation space.

    Actions
    -------
    0 — hold
    1 — buy  (go / stay long)
    2 — sell (go / stay flat)

    Reward
    ------
    PnL change + Sharpe penalty + drawdown penalty.
    """

    STATE_DIM = 50

    def __init__(
        self,
        returns: np.ndarray,
        features: np.ndarray,
        initial_cash: float = 100_000,
        use_ensemble_advisor: bool = False,
        ensemble_url: str = "http://localhost:8002/ensemble",
        ensemble_bias_weight: float = 0.2,
    ) -> None:
        """Initialise the trading environment.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of period returns (e.g. daily log-returns).
        features : np.ndarray
            2-D array of shape ``(T, 50)`` — pre-computed feature matrix.
        initial_cash : float
            Starting portfolio value.
        use_ensemble_advisor : bool
            If True, query the LLM ensemble for action bias before the agent
            selects its action. Default False — does not break existing usage.
        ensemble_url : str
            URL for the ensemble endpoint.
        ensemble_bias_weight : float
            Weight of ensemble bias on action logits (0-1). Default 0.2 means
            20% ensemble bias, 80% RL agent.
        """
        self.returns = np.asarray(returns, dtype=np.float64)
        self.features = np.asarray(features, dtype=np.float64)
        assert self.features.shape[1] == self.STATE_DIM, (
            f"Feature matrix must have {self.STATE_DIM} columns, "
            f"got {self.features.shape[1]}"
        )
        self.initial_cash = float(initial_cash)

        # Ensemble advisor config
        self.use_ensemble_advisor = use_ensemble_advisor
        self.ensemble_url = ensemble_url
        self.ensemble_bias_weight = float(ensemble_bias_weight)
        self.ensemble_metrics = EnsembleAdvisorMetrics()

        # Mutable state — set properly in reset()
        self._step_idx: int = 0
        self._cash: float = self.initial_cash
        self._position: float = 0.0  # 0 or 1 (units held)
        self._entry_price: float = 0.0
        self._portfolio_values: List[float] = []
        self._realised_pnl: float = 0.0
        self._done: bool = False

    # ---- ensemble advisor ----------------------------------------------------

    def _query_ensemble_advisor(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Query the LLM ensemble for action bias.

        Returns a 3-element bias array [hold, buy, sell] or None on failure.
        Uses a 500ms timeout to avoid blocking training.
        """
        if not self.use_ensemble_advisor:
            return None

        try:
            import urllib.request

            payload = json.dumps({
                "prompt": (
                    "Given the current trading state, what action bias "
                    "(buy/sell/hold confidence) do you recommend?"
                ),
                "ml_context": {
                    "state_features": obs.tolist(),
                    "current_position": float(self._position),
                    "current_pnl": float(self._realised_pnl),
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                self.ensemble_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            start = time.time()
            self.ensemble_metrics.calls_made += 1

            with urllib.request.urlopen(req, timeout=0.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            latency_ms = (time.time() - start) * 1000
            self.ensemble_metrics.calls_succeeded += 1
            self.ensemble_metrics.total_latency_ms += latency_ms

            # Parse directional suggestion from response text
            text = data.get("text", "").lower()
            bias = np.zeros(3, dtype=np.float64)  # [hold, buy, sell]

            # Look for directional keywords
            if "buy" in text or "long" in text or "bullish" in text:
                bias[1] = 1.0  # buy bias
            elif "sell" in text or "short" in text or "bearish" in text:
                bias[2] = 1.0  # sell bias
            else:
                bias[0] = 1.0  # hold bias

            return bias

        except Exception as e:
            self.ensemble_metrics.calls_failed += 1
            logger.debug("Ensemble advisor call failed (non-blocking): %s", e)
            return None

    # ---- public interface --------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation.

        Returns
        -------
        np.ndarray
            50-feature observation vector.
        """
        self._step_idx = 0
        self._cash = self.initial_cash
        self._position = 0.0
        self._entry_price = 0.0
        self._portfolio_values = [self.initial_cash]
        self._realised_pnl = 0.0
        self._done = False
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time-step in the environment.

        Parameters
        ----------
        action : int
            0 = hold, 1 = buy, 2 = sell.

        Returns
        -------
        obs : np.ndarray
            Next 50-feature observation.
        reward : float
            Scalar reward signal.
        done : bool
            Whether the episode has ended.
        info : dict
            Auxiliary diagnostics.
        """
        if self._done:
            return self._get_obs(), 0.0, True, {}

        ret = self.returns[self._step_idx]

        # --- execute action ---
        prev_value = self._portfolio_value()

        if action == 1 and self._position == 0.0:
            # Buy — invest all cash
            self._position = 1.0
            self._entry_price = prev_value  # notional entry
        elif action == 2 and self._position == 1.0:
            # Sell — liquidate
            pnl = self._cash * (np.exp(ret) - 1.0) * self._position
            self._realised_pnl += self._cash * (prev_value / self._entry_price - 1.0) if self._entry_price > 0 else 0.0
            self._position = 0.0
            self._entry_price = 0.0

        # --- update portfolio ---
        if self._position == 1.0:
            self._cash *= np.exp(ret)

        current_value = self._portfolio_value()
        self._portfolio_values.append(current_value)

        # --- reward ---
        pnl_change = current_value - prev_value
        reward = self._compute_reward(pnl_change)

        # --- advance ---
        self._step_idx += 1
        if self._step_idx >= len(self.returns):
            self._done = True

        obs = self._get_obs()
        info = {
            "portfolio_value": current_value,
            "position": self._position,
            "step": self._step_idx,
        }

        # --- ensemble advisor bias (returned in info for agent to use) ---
        if self.use_ensemble_advisor:
            ensemble_bias = self._query_ensemble_advisor(obs)
            if ensemble_bias is not None:
                info["ensemble_bias"] = ensemble_bias
                info["ensemble_bias_weight"] = self.ensemble_bias_weight

        return obs, reward, self._done, info

    def get_ensemble_bias_logits(self, logits: np.ndarray, info: dict) -> np.ndarray:
        """Apply ensemble advisor bias to action logits.

        Call this from the agent's action selection to blend ensemble
        suggestions as a soft bias. The RL agent still decides.

        Parameters
        ----------
        logits : np.ndarray
            Raw action logits from the actor network (3-dim).
        info : dict
            Info dict from step() which may contain ensemble_bias.

        Returns
        -------
        np.ndarray
            Adjusted logits with ensemble bias applied.
        """
        if "ensemble_bias" not in info:
            return logits

        bias = info["ensemble_bias"]
        weight = info.get("ensemble_bias_weight", self.ensemble_bias_weight)

        # Soft blend: (1 - weight) * original_logits + weight * bias
        adjusted = (1.0 - weight) * logits + weight * bias
        return adjusted

    # ---- internals ---------------------------------------------------------

    def _portfolio_value(self) -> float:
        return self._cash

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_idx, len(self.features) - 1)
        obs = self.features[idx].copy()

        # Inject live portfolio state into reserved slots (indices 15-18)
        total = self._portfolio_value()
        if total > 0:
            obs[15] = (self._cash if self._position == 0.0 else 0.0) / total  # cash_pct
            obs[16] = self._position  # position_pct
            unrealised = (total / self._entry_price - 1.0) if self._entry_price > 0 else 0.0
            obs[17] = unrealised
            obs[18] = self._realised_pnl / self.initial_cash

        return obs

    def _compute_reward(self, pnl_change: float) -> float:
        """Reward = normalised PnL change + Sharpe penalty + drawdown penalty."""
        reward = pnl_change / self.initial_cash * 100.0  # basis-points scale

        # Sharpe penalty — penalise high variance of recent returns
        if len(self._portfolio_values) > 2:
            vals = np.array(self._portfolio_values[-min(21, len(self._portfolio_values)):])
            rets = np.diff(vals) / vals[:-1]
            if rets.std() > 0:
                sharpe = rets.mean() / rets.std()
                reward += 0.1 * sharpe  # small bonus for smoother equity

        # Drawdown penalty
        peak = max(self._portfolio_values)
        dd = (peak - self._portfolio_values[-1]) / peak if peak > 0 else 0.0
        reward -= 2.0 * dd

        return float(reward)


# ---------------------------------------------------------------------------
# Feature builder (standalone helper used by DeepLearningEngine)
# ---------------------------------------------------------------------------

def build_feature_matrix(returns: np.ndarray, pad_to: int = 50) -> np.ndarray:
    """Build a ``(T, 50)`` feature matrix from a 1-D return series.

    Feature groups
    --------------
    0-4   : Price returns (1d, 5d, 10d, 21d, 63d)
    5-9   : Rolling volatility (same windows)
    10-12 : Momentum proxies (RSI-like, MACD-like, CCI-like)
    13-14 : Microstructure proxies (spread, imbalance)
    15-18 : Portfolio state (filled live by environment)
    19-22 : Statistical (skew, kurtosis, autocorrelation, Hurst proxy)
    23-25 : Risk (VaR 95, max drawdown, current drawdown)
    26-29 : Time encoding (day_sin, day_cos, month_sin, month_cos)
    30-49 : Zero-padded reserve
    """
    T = len(returns)
    feats = np.zeros((T, pad_to), dtype=np.float64)

    cumret = np.cumsum(returns)

    def _rolling_sum(arr, w):
        out = np.zeros(T, dtype=np.float64)
        cs = np.concatenate([[0.0], np.cumsum(arr)])
        for i in range(T):
            start = max(0, i - w + 1)
            out[i] = (cs[i + 1] - cs[start]) / (i - start + 1)
        return out

    def _rolling_std(arr, w):
        out = np.zeros_like(arr)
        for i in range(T):
            start = max(0, i - w + 1)
            out[i] = arr[start:i + 1].std() if i - start > 0 else 1e-8
        return out

    # -- Price returns (windows 1, 5, 10, 21, 63) --
    windows = [1, 5, 10, 21, 63]
    for j, w in enumerate(windows):
        feats[:, j] = _rolling_sum(returns, w)

    # -- Volatility --
    for j, w in enumerate(windows):
        feats[:, 5 + j] = _rolling_std(returns, w)

    # -- Momentum proxies --
    # RSI-like: fraction of positive returns over 14-day window
    for i in range(T):
        start = max(0, i - 13)
        window = returns[start:i + 1]
        up = (window > 0).sum()
        total = len(window)
        feats[i, 10] = (up / total) * 2.0 - 1.0  # scaled to [-1, 1]

    # MACD-like: fast EMA proxy − slow EMA proxy
    alpha_fast, alpha_slow = 2.0 / 13.0, 2.0 / 27.0
    ema_fast = np.zeros(T)
    ema_slow = np.zeros(T)
    ema_fast[0] = returns[0]
    ema_slow[0] = returns[0]
    for i in range(1, T):
        ema_fast[i] = alpha_fast * returns[i] + (1 - alpha_fast) * ema_fast[i - 1]
        ema_slow[i] = alpha_slow * returns[i] + (1 - alpha_slow) * ema_slow[i - 1]
    feats[:, 11] = ema_fast - ema_slow

    # CCI-like proxy
    mean20 = _rolling_sum(returns, 20) * 20  # undo the division
    mad20 = np.zeros(T)
    for i in range(T):
        start = max(0, i - 19)
        seg = returns[start:i + 1]
        mad20[i] = np.mean(np.abs(seg - seg.mean())) + 1e-10
    feats[:, 12] = (returns - mean20 / max(1, 20)) / mad20

    # -- Microstructure proxies --
    feats[:, 13] = np.abs(returns)  # spread proxy (absolute return)
    feats[:, 14] = np.sign(returns) * np.sqrt(np.abs(returns) + 1e-10)  # imbalance proxy

    # -- Portfolio state (15-18): zeros here, filled by env --

    # -- Statistical features --
    stat_window = 63
    for i in range(T):
        start = max(0, i - stat_window + 1)
        seg = returns[start:i + 1]
        n = len(seg)
        if n > 3:
            m = seg.mean()
            s = seg.std() + 1e-10
            skew = np.mean(((seg - m) / s) ** 3)
            kurt = np.mean(((seg - m) / s) ** 4) - 3.0
            # autocorrelation lag-1
            if n > 1:
                ac = np.corrcoef(seg[:-1], seg[1:])[0, 1] if seg[:-1].std() > 0 else 0.0
            else:
                ac = 0.0
            # Hurst proxy via rescaled range
            cs = np.cumsum(seg - m)
            R = cs.max() - cs.min()
            S = s
            hurst = np.log(R / S + 1e-10) / np.log(n + 1e-10) if R > 0 else 0.5
        else:
            skew, kurt, ac, hurst = 0.0, 0.0, 0.0, 0.5

        feats[i, 19] = skew
        feats[i, 20] = kurt
        feats[i, 21] = ac
        feats[i, 22] = hurst

    # -- Risk features --
    for i in range(T):
        start = max(0, i - stat_window + 1)
        seg = returns[start:i + 1]
        # VaR 95
        feats[i, 23] = np.percentile(seg, 5) if len(seg) > 1 else 0.0
        # max drawdown of cumulative returns in window
        cs = np.cumsum(seg)
        running_max = np.maximum.accumulate(cs)
        dd = running_max - cs
        feats[i, 24] = dd.max() if len(dd) > 0 else 0.0
        # current drawdown
        feats[i, 25] = dd[-1] if len(dd) > 0 else 0.0

    # -- Time features --
    for i in range(T):
        day_of_week = i % 5
        month = (i // 21) % 12
        feats[i, 26] = np.sin(2 * np.pi * day_of_week / 5.0)
        feats[i, 27] = np.cos(2 * np.pi * day_of_week / 5.0)
        feats[i, 28] = np.sin(2 * np.pi * month / 12.0)
        feats[i, 29] = np.cos(2 * np.pi * month / 12.0)

    # Columns 30-49 remain zero (padding reserve)

    # Normalise non-portfolio, non-time columns to z-scores
    for c in list(range(0, 15)) + list(range(19, 26)):
        col = feats[:, c]
        std = col.std()
        if std > 1e-10:
            feats[:, c] = (col - col.mean()) / std

    return feats


# ---------------------------------------------------------------------------
# PPOAgent (pure numpy)
# ---------------------------------------------------------------------------

class PPOAgent:
    """Proximal Policy Optimization agent implemented in pure numpy.

    Architecture
    ------------
    Actor  : state_dim -> hidden -> hidden -> action_dim (softmax)
    Critic : state_dim -> hidden -> hidden -> 1
    Activations: tanh.
    """

    def __init__(
        self,
        state_dim: int = 50,
        action_dim: int = 3,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

        # Actor weights
        self.actor_w1 = _he_init(state_dim, hidden_dim)
        self.actor_b1 = np.zeros(hidden_dim)
        self.actor_w2 = _he_init(hidden_dim, hidden_dim)
        self.actor_b2 = np.zeros(hidden_dim)
        self.actor_w3 = _he_init(hidden_dim, action_dim)
        self.actor_b3 = np.zeros(action_dim)

        # Critic weights
        self.critic_w1 = _he_init(state_dim, hidden_dim)
        self.critic_b1 = np.zeros(hidden_dim)
        self.critic_w2 = _he_init(hidden_dim, hidden_dim)
        self.critic_b2 = np.zeros(hidden_dim)
        self.critic_w3 = _he_init(hidden_dim, 1)
        self.critic_b3 = np.zeros(1)

        logger.info(
            "PPOAgent initialised: state_dim=%d, action_dim=%d, hidden=%d, lr=%.1e",
            state_dim, action_dim, hidden_dim, lr,
        )

    # ---- forward passes ----------------------------------------------------

    def _actor_forward(self, state: np.ndarray) -> np.ndarray:
        """Return action probabilities (softmax output)."""
        h1 = _tanh(state @ self.actor_w1 + self.actor_b1)
        h2 = _tanh(h1 @ self.actor_w2 + self.actor_b2)
        logits = h2 @ self.actor_w3 + self.actor_b3
        return _softmax(logits)

    def _actor_logits(self, state: np.ndarray) -> np.ndarray:
        """Return raw action logits (pre-softmax)."""
        h1 = _tanh(state @ self.actor_w1 + self.actor_b1)
        h2 = _tanh(h1 @ self.actor_w2 + self.actor_b2)
        return h2 @ self.actor_w3 + self.actor_b3

    def _critic_forward(self, state: np.ndarray) -> float:
        """Return scalar value estimate."""
        h1 = _tanh(state @ self.critic_w1 + self.critic_b1)
        h2 = _tanh(h1 @ self.critic_w2 + self.critic_b2)
        value = (h2 @ self.critic_w3 + self.critic_b3)[0]
        return float(value)

    # ---- action selection --------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        info: Optional[dict] = None,
        env: Optional["TradingEnvironment"] = None,
    ) -> Tuple[int, float]:
        """Sample an action from the policy.

        Parameters
        ----------
        state : np.ndarray
            50-d observation vector.
        info : dict, optional
            Info dict from env.step() that may contain ensemble_bias.
        env : TradingEnvironment, optional
            Environment instance for applying ensemble bias.

        Returns
        -------
        action : int
            Chosen action index (0, 1, or 2).
        log_prob : float
            Log-probability of the chosen action under the current policy.
        """
        logits = self._actor_logits(state)

        # Apply ensemble bias if available
        if info is not None and env is not None and "ensemble_bias" in info:
            logits = env.get_ensemble_bias_logits(logits, info)

        probs = _softmax(logits)
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()
        action = np.random.choice(self.action_dim, p=probs)
        log_prob = float(np.log(probs[action]))
        return int(action), log_prob

    # ---- GAE ---------------------------------------------------------------

    @staticmethod
    def _compute_advantages(
        rewards: np.ndarray,
        values: np.ndarray,
        gamma: float,
        lam: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation (GAE-lambda).

        Parameters
        ----------
        rewards : np.ndarray
            1-D reward sequence of length T.
        values : np.ndarray
            1-D value estimates of length T (+ optional bootstrap).
        gamma : float
            Discount factor.
        lam : float
            GAE smoothing parameter.

        Returns
        -------
        advantages : np.ndarray
            GAE advantage estimates, length T.
        returns_ : np.ndarray
            Discounted returns (advantages + values), length T.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float64)
        gae = 0.0
        # Append a terminal value of 0
        values_ext = np.append(values, 0.0)
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        returns_ = advantages + values[:T]
        return advantages, returns_

    # ---- PPO update --------------------------------------------------------

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
    ) -> dict:
        """Run PPO update on a collected trajectory.

        Parameters
        ----------
        states : np.ndarray   — (T, state_dim)
        actions : np.ndarray  — (T,) int
        rewards : np.ndarray  — (T,)
        log_probs : np.ndarray — (T,) old log-probs
        values : np.ndarray   — (T,) old value estimates

        Returns
        -------
        dict with ``policy_loss``, ``value_loss``, ``entropy``.
        """
        T = len(rewards)
        advantages, returns_ = self._compute_advantages(rewards, values, self.gamma)

        # Normalise advantages
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs):
            # Shuffle
            indices = np.random.permutation(T)
            batch_size = max(T // 4, 1)

            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                idx = indices[start:end]
                B = len(idx)

                # ----- Forward passes for batch -----
                actor_grads = self._zero_actor_grads()
                critic_grads = self._zero_critic_grads()
                batch_ploss = 0.0
                batch_vloss = 0.0
                batch_ent = 0.0

                for b_i in idx:
                    s = states[b_i]
                    a = int(actions[b_i])
                    old_lp = log_probs[b_i]
                    adv = advantages[b_i]
                    ret = returns_[b_i]

                    # --- Actor forward & backward ---
                    z1 = s @ self.actor_w1 + self.actor_b1
                    h1 = _tanh(z1)
                    z2 = h1 @ self.actor_w2 + self.actor_b2
                    h2 = _tanh(z2)
                    logits = h2 @ self.actor_w3 + self.actor_b3
                    probs = _softmax(logits)
                    probs = np.clip(probs, 1e-8, None)
                    probs /= probs.sum()

                    new_lp = np.log(probs[a])
                    ratio = np.exp(new_lp - old_lp)
                    surr1 = ratio * adv
                    surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                    policy_loss = -min(surr1, surr2)

                    entropy = -np.sum(probs * np.log(probs))
                    batch_ploss += policy_loss
                    batch_ent += entropy

                    # Gradient of policy loss w.r.t. logits (simplified)
                    # d(-log pi(a)) / d(logits) ≈ probs - one_hot(a)
                    d_logits = probs.copy()
                    d_logits[a] -= 1.0
                    # Scale by clipped ratio signal
                    if surr1 <= surr2:
                        scale = -adv
                    else:
                        # Clipped — no gradient w.r.t. logits when clipped
                        scale = 0.0
                    d_logits *= scale
                    # Add entropy bonus gradient
                    ent_coeff = 0.01
                    d_logits += ent_coeff * (np.log(probs) + 1.0)

                    # Backprop through actor
                    dw3 = np.outer(h2, d_logits)
                    db3 = d_logits
                    dh2 = d_logits @ self.actor_w3.T
                    dh2 *= _tanh_deriv(h2)
                    dw2 = np.outer(h1, dh2)
                    db2 = dh2
                    dh1 = dh2 @ self.actor_w2.T
                    dh1 *= _tanh_deriv(h1)
                    dw1 = np.outer(s, dh1)
                    db1 = dh1

                    actor_grads["w1"] += dw1 / B
                    actor_grads["b1"] += db1 / B
                    actor_grads["w2"] += dw2 / B
                    actor_grads["b2"] += db2 / B
                    actor_grads["w3"] += dw3 / B
                    actor_grads["b3"] += db3 / B

                    # --- Critic forward & backward ---
                    cz1 = s @ self.critic_w1 + self.critic_b1
                    ch1 = _tanh(cz1)
                    cz2 = ch1 @ self.critic_w2 + self.critic_b2
                    ch2 = _tanh(cz2)
                    v_pred = (ch2 @ self.critic_w3 + self.critic_b3)[0]

                    vloss = (v_pred - ret) ** 2
                    batch_vloss += vloss

                    # dL/dv = 2*(v - ret)
                    dv = 2.0 * (v_pred - ret)
                    dcw3 = ch2.reshape(-1, 1) * dv
                    dcb3 = np.array([dv])
                    dch2 = (self.critic_w3.flatten() * dv)
                    dch2 *= _tanh_deriv(ch2)
                    dcw2 = np.outer(ch1, dch2)
                    dcb2 = dch2
                    dch1 = dch2 @ self.critic_w2.T
                    dch1 *= _tanh_deriv(ch1)
                    dcw1 = np.outer(s, dch1)
                    dcb1 = dch1

                    critic_grads["w1"] += dcw1 / B
                    critic_grads["b1"] += dcb1 / B
                    critic_grads["w2"] += dcw2 / B
                    critic_grads["b2"] += dcb2 / B
                    critic_grads["w3"] += dcw3 / B
                    critic_grads["b3"] += dcb3 / B

                # --- Apply gradients (SGD) ---
                self.actor_w1 -= self.lr * actor_grads["w1"]
                self.actor_b1 -= self.lr * actor_grads["b1"]
                self.actor_w2 -= self.lr * actor_grads["w2"]
                self.actor_b2 -= self.lr * actor_grads["b2"]
                self.actor_w3 -= self.lr * actor_grads["w3"]
                self.actor_b3 -= self.lr * actor_grads["b3"]

                self.critic_w1 -= self.lr * critic_grads["w1"]
                self.critic_b1 -= self.lr * critic_grads["b1"]
                self.critic_w2 -= self.lr * critic_grads["w2"]
                self.critic_b2 -= self.lr * critic_grads["b2"]
                self.critic_w3 -= self.lr * critic_grads["w3"]
                self.critic_b3 -= self.lr * critic_grads["b3"]

                total_policy_loss += batch_ploss / B
                total_value_loss += batch_vloss / B
                total_entropy += batch_ent / B

        n_updates = max(self.epochs * max(T // max(T // 4, 1), 1), 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    # ---- gradient helpers --------------------------------------------------

    def _zero_actor_grads(self) -> dict:
        return {
            "w1": np.zeros_like(self.actor_w1),
            "b1": np.zeros_like(self.actor_b1),
            "w2": np.zeros_like(self.actor_w2),
            "b2": np.zeros_like(self.actor_b2),
            "w3": np.zeros_like(self.actor_w3),
            "b3": np.zeros_like(self.actor_b3),
        }

    def _zero_critic_grads(self) -> dict:
        return {
            "w1": np.zeros_like(self.critic_w1),
            "b1": np.zeros_like(self.critic_b1),
            "w2": np.zeros_like(self.critic_w2),
            "b2": np.zeros_like(self.critic_b2),
            "w3": np.zeros_like(self.critic_w3),
            "b3": np.zeros_like(self.critic_b3),
        }


# ---------------------------------------------------------------------------
# DeepLearningEngine — high-level orchestrator
# ---------------------------------------------------------------------------

class DeepLearningEngine:
    """High-level engine that wraps PPOAgent + TradingEnvironment.

    Provides train / predict / walk-forward validation entry-points.
    """

    def __init__(self, state_dim: int = 50, lookback: int = 252) -> None:
        """Initialise the engine.

        Parameters
        ----------
        state_dim : int
            Observation dimensionality (default 50).
        lookback : int
            Default lookback window for feature computation.
        """
        self.state_dim = state_dim
        self.lookback = lookback
        self.agent: Optional[PPOAgent] = None
        self._training_history: List[dict] = []
        logger.info("DeepLearningEngine initialised (state_dim=%d, lookback=%d)", state_dim, lookback)

    # ---- public API --------------------------------------------------------

    def train(self, returns: pd.Series, n_episodes: int = 100) -> dict:
        """Train the PPO agent on historical returns.

        Parameters
        ----------
        returns : pd.Series
            Daily return series.
        n_episodes : int
            Number of training episodes.

        Returns
        -------
        dict
            Training summary with ``episode_rewards``, ``policy_losses``,
            ``value_losses``.
        """
        ret_arr = returns.values.astype(np.float64)
        features = build_feature_matrix(ret_arr, pad_to=self.state_dim)

        self.agent = PPOAgent(state_dim=self.state_dim)
        env = TradingEnvironment(ret_arr, features)

        episode_rewards: List[float] = []
        policy_losses: List[float] = []
        value_losses: List[float] = []

        for ep in range(n_episodes):
            obs = env.reset()
            states, actions, rewards, log_probs, values = [], [], [], [], []
            done = False
            total_reward = 0.0

            while not done:
                action, lp = self.agent.select_action(obs)
                value = self.agent._critic_forward(obs)

                states.append(obs)
                actions.append(action)
                log_probs.append(lp)
                values.append(value)

                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                total_reward += reward

            # PPO update
            metrics = self.agent.update(
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(log_probs),
                np.array(values),
            )

            episode_rewards.append(total_reward)
            policy_losses.append(metrics["policy_loss"])
            value_losses.append(metrics["value_loss"])

            if (ep + 1) % max(1, n_episodes // 10) == 0:
                logger.info(
                    "Episode %d/%d — reward=%.2f  ploss=%.4f  vloss=%.4f",
                    ep + 1, n_episodes, total_reward,
                    metrics["policy_loss"], metrics["value_loss"],
                )

        summary = {
            "episode_rewards": episode_rewards,
            "policy_losses": policy_losses,
            "value_losses": value_losses,
            "final_reward": episode_rewards[-1] if episode_rewards else 0.0,
            "mean_reward_last10": float(np.mean(episode_rewards[-10:])),
        }
        self._training_history.append(summary)
        logger.info("Training complete — mean reward (last 10): %.2f", summary["mean_reward_last10"])
        return summary

    def predict(self, returns: pd.Series) -> dict:
        """Generate a trading signal from the trained agent.

        Parameters
        ----------
        returns : pd.Series
            Recent return series (at least ``lookback`` observations).

        Returns
        -------
        dict
            ``action`` (0/1/2), ``confidence``, ``expected_return``.
        """
        if self.agent is None:
            raise RuntimeError("Agent has not been trained. Call train() first.")

        ret_arr = returns.values.astype(np.float64)
        features = build_feature_matrix(ret_arr, pad_to=self.state_dim)
        state = features[-1]

        probs = self.agent._actor_forward(state)
        action = int(np.argmax(probs))
        confidence = float(probs[action])
        value = self.agent._critic_forward(state)

        action_map = {0: "hold", 1: "buy", 2: "sell"}
        logger.info(
            "Prediction: %s (confidence=%.2f, value=%.4f)",
            action_map[action], confidence, value,
        )

        return {
            "action": action,
            "action_label": action_map[action],
            "confidence": confidence,
            "expected_return": value,
            "probabilities": {action_map[i]: float(probs[i]) for i in range(3)},
        }

    def walk_forward(
        self,
        returns: pd.Series,
        train_window: int = 252,
        test_window: int = 63,
    ) -> List[dict]:
        """Walk-forward validation.

        Slides a training window across the series, trains on each window,
        then evaluates on the subsequent ``test_window`` period.

        Parameters
        ----------
        returns : pd.Series
            Full return series.
        train_window : int
            Number of observations for each training fold.
        test_window : int
            Number of observations for each test fold.

        Returns
        -------
        list[dict]
            One dict per fold with ``fold``, ``train_reward``,
            ``test_reward``, ``test_actions``, ``test_portfolio_value``.
        """
        n = len(returns)
        results: List[dict] = []
        fold = 0

        start = 0
        while start + train_window + test_window <= n:
            fold += 1
            train_end = start + train_window
            test_end = train_end + test_window

            train_ret = returns.iloc[start:train_end]
            test_ret = returns.iloc[train_end:test_end]

            logger.info(
                "Walk-forward fold %d: train [%d:%d], test [%d:%d]",
                fold, start, train_end, train_end, test_end,
            )

            # Train
            train_summary = self.train(train_ret, n_episodes=50)

            # Evaluate on test set
            test_arr = test_ret.values.astype(np.float64)
            test_features = build_feature_matrix(test_arr, pad_to=self.state_dim)
            test_env = TradingEnvironment(test_arr, test_features)

            obs = test_env.reset()
            done = False
            test_reward = 0.0
            test_actions = []

            while not done:
                action, _ = self.agent.select_action(obs)
                obs, reward, done, info = test_env.step(action)
                test_reward += reward
                test_actions.append(action)

            fold_result = {
                "fold": fold,
                "train_start": start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "train_reward": train_summary["final_reward"],
                "test_reward": test_reward,
                "test_actions": test_actions,
                "test_portfolio_value": test_env._portfolio_value(),
            }
            results.append(fold_result)
            logger.info(
                "Fold %d complete — train_reward=%.2f, test_reward=%.2f, "
                "test_portfolio=%.2f",
                fold, fold_result["train_reward"], test_reward,
                fold_result["test_portfolio_value"],
            )

            start += test_window  # slide forward

        logger.info("Walk-forward complete — %d folds evaluated.", len(results))
        return results

    def get_performance(self) -> dict:
        """Return aggregated training performance metrics.

        Returns
        -------
        dict
            ``n_trainings``, ``last_training``, ``avg_final_reward``, etc.
        """
        if not self._training_history:
            return {"n_trainings": 0, "status": "untrained"}

        final_rewards = [h["final_reward"] for h in self._training_history]
        mean_rewards = [h["mean_reward_last10"] for h in self._training_history]

        return {
            "n_trainings": len(self._training_history),
            "avg_final_reward": float(np.mean(final_rewards)),
            "best_final_reward": float(np.max(final_rewards)),
            "avg_mean_reward_last10": float(np.mean(mean_rewards)),
            "last_training": self._training_history[-1],
        }
