"""
Avellaneda-Stoikov Market Making Strategy
==========================================

Reference: Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit
order book." Quantitative Finance, 8(3), 217-224.

Key equations
-------------

Reservation price (mid adjusted for inventory risk):
    r(s, q, t) = s - q * γ * σ² * (T - t)

where:
    s   = current mid price
    q   = current inventory (signed, in units)
    γ   = risk aversion coefficient
    σ²  = variance of price process
    T-t = remaining time in trading session

Optimal spread:
    δ* = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/κ)

where:
    κ   = order arrival rate parameter (from Poisson model)

Optimal bid/ask quotes:
    bid = r - δ*/2
    ask = r + δ*/2

Intuition
---------
The reservation price skews away from inventory. If long, we lower both quotes
to attract selling flow (reduce inventory). The spread compensates for volatility
risk over the remaining session and the expected value of order arrival.

Assumptions
-----------
1. Price follows arithmetic Brownian motion (not GBM — this matters at high freq).
2. Order arrivals follow a Poisson process with intensity λ(δ) = A*exp(-κ*δ).
3. Utility is CARA (constant absolute risk aversion).
4. No market impact from our own quotes (small MM assumption).
5. Session has finite horizon T — we implement a rolling window approximation
   for continuous operation.

Limitations
-----------
- Arithmetic BM means prices can go negative (fine at short horizons).
- κ and A need calibration from live data; we estimate from simulation history.
- Does not account for adverse selection directly (see InventoryAwareStrategy).
- Optimal only in the Poisson arrival / CARA utility sense.

Implementation notes
--------------------
We add several practical extensions:
1. Hard inventory limits with emergency quote withdrawal.
2. Volatility regime detection (EWMA-based).
3. Dynamic κ estimation from fill history.
4. Maximum quote size and position sizing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ASParams:
    """Avellaneda-Stoikov model parameters."""

    gamma: float = 0.1          # risk aversion [0.01 = low, 1.0 = high]
    sigma: float = 0.02         # per-step price vol (recalibrated dynamically)
    kappa: float = 1.5          # order arrival decay parameter
    A: float = 140.0            # order arrival base intensity
    T: float = 1.0              # session horizon (normalised; we use rolling)
    dt: float = 1.0 / 390.0    # time step (1 minute out of 390-min session)
    max_inventory: int = 50     # hard inventory limit in units
    min_spread: float = 0.001   # minimum quoted spread
    max_spread: float = 10.0    # maximum quoted spread (don't clamp formula results)
    order_size: int = 1         # default quote size
    ewma_alpha: float = 0.94    # EWMA decay for vol estimation


@dataclass
class MMState:
    """Mutable market-maker state."""

    inventory: int = 0
    cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fills: int = 0
    bid_fills: int = 0
    ask_fills: int = 0
    adverse_selection_cost: float = 0.0
    spread_capture: float = 0.0
    active_bid_id: Optional[str] = None
    active_ask_id: Optional[str] = None
    last_mid: float = 0.0
    ewma_var: float = 0.0           # running variance estimate
    session_time: float = 0.0       # current t in [0, T]
    price_history: List[float] = field(default_factory=list)
    fill_history: List[dict] = field(default_factory=list)


class AvellanedaStoikov:
    """
    Production-grade Avellaneda-Stoikov market maker.

    The core idea: optimal quotes are derived by solving a Hamilton-Jacobi-Bellman
    equation for a CARA investor who wants to maximise terminal wealth while
    managing inventory risk over a finite horizon.

    Usage
    -----
    >>> mm = AvellanedaStoikov(params=ASParams(gamma=0.1))
    >>> bid, ask = mm.compute_quotes(mid=100.0, inventory=5, t=0.5)
    """

    def __init__(self, params: Optional[ASParams] = None) -> None:
        self.params = params or ASParams()
        self.state = MMState()
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Core model
    # ------------------------------------------------------------------

    def reservation_price(self, mid: float, inventory: int, t: float) -> float:
        """
        Skewed mid-price that internalises inventory carrying cost.

        r = s - q * γ * σ² * (T - t)

        When inventory is positive (long), reservation price < mid, meaning
        we are willing to sell slightly below fair value to reduce risk.
        """
        p = self.params
        time_remaining = max(p.T - t, p.dt)  # avoid division by zero at T
        return mid - inventory * p.gamma * (p.sigma ** 2) * time_remaining

    def optimal_spread(self, t: float) -> float:
        """
        δ* = γ σ² (T-t) + (2/γ) * ln(1 + γ/κ)

        First term: risk premium for holding inventory over remaining session.
        Second term: compensation for order-flow uncertainty (informed vs noise).
        """
        p = self.params
        time_remaining = max(p.T - t, p.dt)
        risk_term = p.gamma * (p.sigma ** 2) * time_remaining
        info_term = (2.0 / p.gamma) * math.log(1.0 + p.gamma / p.kappa)
        raw_spread = risk_term + info_term
        return float(np.clip(raw_spread, p.min_spread, p.max_spread))

    def compute_quotes(
        self, mid: float, inventory: int, t: float
    ) -> Tuple[float, float]:
        """
        Returns (bid_price, ask_price).

        Quotes are centred around reservation price, not mid.
        This is the key insight: we shift the entire quote away from
        our inventory direction.
        """
        r = self.reservation_price(mid, inventory, t)
        half_spread = self.optimal_spread(t) / 2.0
        bid = r - half_spread
        ask = r + half_spread
        return round(bid, 4), round(ask, 4)

    def fill_probability(self, quote_price: float, mid: float, side: str) -> float:
        """
        Approximate fill probability from Poisson arrival model.

        P(fill) ≈ exp(-κ * δ)   where δ = |quote - mid|

        This is derived from the intensity function:
            λ(δ) = A * exp(-κ * δ)
        integrated over the remaining session.
        """
        delta = abs(quote_price - mid)
        return math.exp(-self.params.kappa * delta)

    def expected_value_per_quote(
        self, bid: float, ask: float, mid: float, t: float
    ) -> Dict[str, float]:
        """
        Decompose expected value of a round-trip quote.

        EV_bid = P(bid fill) * (mid - bid)  — spread capture
        EV_ask = P(ask fill) * (ask - mid)  — spread capture
        EV_round_trip = EV_bid + EV_ask (ignoring correlation)

        Note: this ignores adverse selection — see adverse_selection_adjustment.
        """
        p_bid = self.fill_probability(bid, mid, "bid")
        p_ask = self.fill_probability(ask, mid, "ask")
        ev_bid = p_bid * (mid - bid)
        ev_ask = p_ask * (ask - mid)
        return {
            "ev_bid": ev_bid,
            "ev_ask": ev_ask,
            "ev_round_trip": ev_bid + ev_ask,
            "p_bid_fill": p_bid,
            "p_ask_fill": p_ask,
        }

    # ------------------------------------------------------------------
    # Dynamic recalibration
    # ------------------------------------------------------------------

    def update_volatility(self, new_mid: float) -> None:
        """
        EWMA variance update:  σ²_t = α * σ²_{t-1} + (1-α) * r_t²

        We use log returns to be scale-invariant, but for short horizons
        the difference is negligible.
        """
        alpha = self.params.ewma_alpha
        if self.state.last_mid > 0:
            ret = (new_mid - self.state.last_mid) / self.state.last_mid
            self.state.ewma_var = (
                alpha * self.state.ewma_var + (1 - alpha) * ret ** 2
            )
            self.params.sigma = math.sqrt(max(self.state.ewma_var, 1e-8))
        self.state.last_mid = new_mid

    def recalibrate_kappa(self, window: int = 100) -> None:
        """
        MLE estimate of κ from recent fill history.

        Under the Poisson model: κ_hat = 1 / mean(δ_fills)
        where δ_fills are the half-spreads at which fills occurred.
        """
        fills = self.state.fill_history[-window:]
        if len(fills) < 10:
            return
        deltas = [f["delta"] for f in fills if f["delta"] > 0]
        if deltas:
            self.params.kappa = 1.0 / (np.mean(deltas) + 1e-9)

    # ------------------------------------------------------------------
    # State updates (called by simulation loop)
    # ------------------------------------------------------------------

    def on_fill(
        self,
        side: str,
        fill_price: float,
        fill_qty: int,
        mid_at_fill: float,
        is_informed: bool = False,
    ) -> None:
        """Update state after a fill."""
        delta = abs(fill_price - mid_at_fill)
        self.state.fill_history.append({"side": side, "delta": delta, "mid": mid_at_fill})

        if side == "bid":
            self.state.inventory += fill_qty
            self.state.cash -= fill_price * fill_qty
            self.state.bid_fills += 1
        else:
            self.state.inventory -= fill_qty
            self.state.cash += fill_price * fill_qty
            self.state.ask_fills += 1

        self.state.total_fills += 1
        spread_captured = delta
        self.state.spread_capture += spread_captured * fill_qty

        # Adverse selection: if informed trader hits us, we lose the
        # subsequent price move.  We track this post-hoc.
        if is_informed:
            self.state.adverse_selection_cost += delta * fill_qty

    def mark_to_market(self, mid: float) -> float:
        """Total PnL = cash + inventory * mid."""
        self.state.unrealized_pnl = self.state.inventory * mid
        total_pnl = self.state.cash + self.state.unrealized_pnl
        return total_pnl

    def should_trade(self) -> bool:
        """Risk controls: halt quoting if inventory exceeds limit."""
        return abs(self.state.inventory) < self.params.max_inventory

    def emergency_unwind_side(self) -> Optional[str]:
        """
        If inventory is extreme, return which side to lean quotes toward
        for faster inventory reduction.
        """
        p = self.params
        if self.state.inventory > p.max_inventory * 0.8:
            return "sell"   # lean ask closer to mid
        if self.state.inventory < -p.max_inventory * 0.8:
            return "buy"    # lean bid closer to mid
        return None

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------

    def inventory_risk(self, mid: float, t: float) -> float:
        """
        Instantaneous inventory risk = q² * γ * σ² * (T - t)

        This is the value of the penalty term in the HJB equation,
        representing the certainty-equivalent cost of holding inventory.
        """
        p = self.params
        time_remaining = max(p.T - t, p.dt)
        return (self.state.inventory ** 2) * p.gamma * (p.sigma ** 2) * time_remaining

    def sharpe_ratio(self) -> float:
        """Approximate annualised Sharpe from fill history."""
        if len(self.state.fill_history) < 2:
            return 0.0
        pnls = [f.get("pnl", 0) for f in self.state.fill_history if "pnl" in f]
        if len(pnls) < 2:
            return 0.0
        arr = np.array(pnls)
        if arr.std() < 1e-12:
            return 0.0
        return float(arr.mean() / arr.std() * math.sqrt(252 * 390))

    def summary(self, mid: float) -> Dict:
        s = self.state
        return {
            "inventory": s.inventory,
            "cash": s.cash,
            "unrealized_pnl": s.inventory * mid,
            "total_pnl": s.cash + s.inventory * mid,
            "spread_capture": s.spread_capture,
            "adverse_selection_cost": s.adverse_selection_cost,
            "net_alpha": s.spread_capture - s.adverse_selection_cost,
            "total_fills": s.total_fills,
            "bid_fills": s.bid_fills,
            "ask_fills": s.ask_fills,
            "fill_imbalance": (s.bid_fills - s.ask_fills) / max(s.total_fills, 1),
            "current_sigma": self.params.sigma,
            "current_kappa": self.params.kappa,
        }
