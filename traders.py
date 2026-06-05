"""
Market Participant Models
=========================

We model three classes of participants beyond the market maker:

1. InformedTrader
   Has private signal about future price direction.
   Trades aggressively when signal strength is high.
   Main source of adverse selection for the MM.

2. NoiseTrade
   Random walk in order submission; no private information.
   Provides the "noise" that makes market making profitable.
   Size and frequency calibrated to match typical retail flow.

3. LiquidityTaker
   Institutional participant with scheduling constraints.
   Uses VWAP-style execution; prefers market impact minimisation.
   Represents institutional order flow.

The mix of these participants determines the adverse-selection environment.
Key parameter: informed_fraction = fraction of flow that is informed.

References
----------
- Kyle (1985): continuous-time insider trading model.
- Glosten-Milgrom (1985): sequential trade model with informed/uninformed traders.
- Easley-O'Hara (1987): time and the process of security price adjustment.
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.core.matching_engine import Order, OrderType, Side


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Participant(ABC):
    """Abstract base for all market participants."""

    def __init__(self, participant_id: str, rng: np.random.Generator) -> None:
        self.participant_id = participant_id
        self.rng = rng
        self.cash: float = 0.0
        self.inventory: int = 0
        self.total_orders: int = 0

    @abstractmethod
    def generate_orders(
        self, timestamp: float, mid: float, spread: float, book_state: dict
    ) -> List[Order]:
        """Generate orders for this time step."""
        ...

    def _new_order(
        self,
        side: Side,
        order_type: OrderType,
        price: float,
        quantity: int,
        timestamp: float,
    ) -> Order:
        self.total_orders += 1
        return Order(
            order_id=str(uuid.uuid4()),
            participant_id=self.participant_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Informed trader (Kyle/GM model)
# ---------------------------------------------------------------------------

@dataclass
class InformedTraderParams:
    signal_decay: float = 0.95          # AR(1) coefficient on private signal
    signal_vol: float = 0.005           # std dev of signal innovation
    aggression: float = 0.6             # fraction of signal monetised per step
    max_position: int = 200             # position limit
    order_size_base: int = 5            # base order size
    order_size_scale: float = 2.0       # size scaling with signal strength


class InformedTrader(Participant):
    """
    Trades on private information (e.g. fundamental value signal).

    Model:
        v_t = v_{t-1} + ε_t,     ε_t ~ N(0, σ_v²)
        signal_t = ρ * signal_{t-1} + η_t,  η_t ~ N(0, σ_η²)

    The informed trader knows v_t (or an estimate of it) and trades
    proportional to (v_t - mid_t) to converge to fundamental value.

    This is what makes the adverse selection problem hard: the MM cannot
    distinguish informed from noise flow ex ante.
    """

    def __init__(
        self,
        participant_id: str,
        rng: np.random.Generator,
        params: Optional[InformedTraderParams] = None,
    ) -> None:
        super().__init__(participant_id, rng)
        self.params = params or InformedTraderParams()
        self._signal: float = 0.0          # private signal relative to mid
        self._fundamental: float = 0.0     # running estimate of true value

    def update_signal(self, mid: float) -> None:
        p = self.params
        self._signal = p.signal_decay * self._signal + self.rng.normal(0, p.signal_vol)
        self._fundamental = mid + self._signal

    def generate_orders(
        self, timestamp: float, mid: float, spread: float, book_state: dict
    ) -> List[Order]:
        self.update_signal(mid)
        p = self.params

        # Only trade when signal is strong enough to overcome spread cost
        half_spread = spread / 2.0
        if abs(self._signal) <= half_spread * 1.1:
            return []

        # Direction: buy if fundamental > mid, sell otherwise
        side = Side.BID if self._signal > 0 else Side.ASK

        # Check position limits
        if side == Side.BID and self.inventory >= p.max_position:
            return []
        if side == Side.ASK and self.inventory <= -p.max_position:
            return []

        # Size proportional to signal strength
        signal_strength = abs(self._signal) / p.signal_vol
        size = max(1, int(p.order_size_base * min(signal_strength * p.order_size_scale, 10)))

        # Use market orders (informed traders want guaranteed execution)
        order = self._new_order(
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=size,
            timestamp=timestamp,
        )
        return [order]

    @property
    def is_informed(self) -> bool:
        return True

    @property
    def signal(self) -> float:
        return self._signal


# ---------------------------------------------------------------------------
# Noise trader
# ---------------------------------------------------------------------------

@dataclass
class NoiseTraderParams:
    arrival_rate: float = 0.3       # Poisson intensity per step
    order_size_mean: float = 2.0    # mean order size
    order_size_std: float = 1.0
    limit_fraction: float = 0.4     # fraction of orders that are limit
    spread_factor: float = 0.5      # how deep inside spread to place limits


class NoiseTrader(Participant):
    """
    Uninformed participant submitting random orders.

    Models retail/noise flow; no private information.
    The MM profits from noise flow (collect spread) but suffers from
    informed flow (adverse selection). The ratio of these determines
    whether market making is positive EV.

    Arrival process: Poisson(λ) per simulation step.
    Order type: mix of market and limit orders.
    """

    def __init__(
        self,
        participant_id: str,
        rng: np.random.Generator,
        params: Optional[NoiseTraderParams] = None,
    ) -> None:
        super().__init__(participant_id, rng)
        self.params = params or NoiseTraderParams()

    def generate_orders(
        self, timestamp: float, mid: float, spread: float, book_state: dict
    ) -> List[Order]:
        p = self.params

        # Poisson arrival
        n_orders = self.rng.poisson(p.arrival_rate)
        if n_orders == 0:
            return []

        orders = []
        for _ in range(n_orders):
            side = Side.BID if self.rng.random() < 0.5 else Side.ASK
            size = max(1, int(self.rng.normal(p.order_size_mean, p.order_size_std)))

            if self.rng.random() < p.limit_fraction:
                # Limit order: place inside spread with some offset
                offset = self.rng.exponential(max(spread * p.spread_factor, 1e-6))
                if side == Side.BID:
                    price = round(mid - offset, 4)
                else:
                    price = round(mid + offset, 4)
                otype = OrderType.LIMIT
            else:
                price = 0.0
                otype = OrderType.MARKET

            orders.append(
                self._new_order(
                    side=side,
                    order_type=otype,
                    price=price,
                    quantity=size,
                    timestamp=timestamp,
                )
            )

        return orders

    @property
    def is_informed(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Institutional liquidity taker (VWAP-style)
# ---------------------------------------------------------------------------

@dataclass
class LiquidityTakerParams:
    total_order_size: int = 500         # total institutional order to execute
    time_horizon: int = 100             # steps over which to execute
    urgency: float = 0.5                # 0 = passive, 1 = aggressive
    max_participation_rate: float = 0.2 # max fraction of volume per step


class LiquidityTaker(Participant):
    """
    Institutional participant with a parent order to execute.

    Uses a simplified VWAP schedule with urgency-based aggression.
    At high urgency, uses market orders; at low urgency, uses limit orders
    slightly passive relative to mid.

    This participant represents the "large order" that drives predictable
    price impact and is detectable via order flow imbalance signals.
    """

    def __init__(
        self,
        participant_id: str,
        rng: np.random.Generator,
        side: Side,
        params: Optional[LiquidityTakerParams] = None,
    ) -> None:
        super().__init__(participant_id, rng)
        self.params = params or LiquidityTakerParams()
        self.target_side = side
        self.remaining: int = params.total_order_size if params else 500
        self._step: int = 0

    @property
    def is_done(self) -> bool:
        return self.remaining <= 0

    def generate_orders(
        self, timestamp: float, mid: float, spread: float, book_state: dict
    ) -> List[Order]:
        if self.is_done:
            return []

        p = self.params
        self._step += 1

        # VWAP: uniform schedule with some randomness
        steps_remaining = max(p.time_horizon - self._step, 1)
        target_qty = max(1, int(self.remaining / steps_remaining))
        # Add noise to avoid footprint predictability
        noise = self.rng.integers(-max(1, target_qty // 4), max(2, target_qty // 4))
        qty = max(1, min(target_qty + noise, self.remaining))

        # Choose order type based on urgency
        if self.rng.random() < p.urgency:
            price = 0.0
            otype = OrderType.MARKET
        else:
            # Passive limit order
            if self.target_side == Side.BID:
                price = round(mid - spread * 0.1, 4)
            else:
                price = round(mid + spread * 0.1, 4)
            otype = OrderType.LIMIT

        self.remaining -= qty
        return [
            self._new_order(
                side=self.target_side,
                order_type=otype,
                price=price,
                quantity=qty,
                timestamp=timestamp,
            )
        ]

    @property
    def is_informed(self) -> bool:
        return False   # has directional flow but not private information per se


# ---------------------------------------------------------------------------
# Participant factory
# ---------------------------------------------------------------------------

def create_participant_pool(
    n_noise: int = 10,
    n_informed: int = 2,
    n_liquidity: int = 1,
    informed_params: Optional[InformedTraderParams] = None,
    noise_params: Optional[NoiseTraderParams] = None,
    seed: int = 42,
) -> List[Participant]:
    """Create a heterogeneous pool of market participants."""
    rng = np.random.default_rng(seed)
    participants: List[Participant] = []

    for i in range(n_noise):
        participants.append(
            NoiseTrader(f"NOISE_{i:03d}", np.random.default_rng(rng.integers(1e9)), noise_params)
        )

    for i in range(n_informed):
        participants.append(
            InformedTrader(
                f"INFORMED_{i:03d}", np.random.default_rng(rng.integers(1e9)), informed_params
            )
        )

    for i in range(n_liquidity):
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        participants.append(
            LiquidityTaker(
                f"LIQTAKER_{i:03d}",
                np.random.default_rng(rng.integers(1e9)),
                side=side,
            )
        )

    return participants
