"""
Market Simulation Engine
========================

Drives the full simulation loop:
  1. Advance price process (Heston-like with regime switching)
  2. Generate orders from all participants
  3. Submit MM quotes
  4. Process through matching engine
  5. Collect analytics

Price Process
-------------
We use a regime-switching Ornstein-Uhlenbeck process to model realistic
price dynamics with calm and volatile periods.

In regime k:
    dS = μ_k * dt + σ_k * dW + J * dN

where:
    μ_k  = drift in regime k (small for short horizons)
    σ_k  = vol in regime k (low ~ 0.5%, high ~ 2%)
    J    = jump size ~ N(0, σ_J²)
    N    = Poisson jump process, intensity λ_J

Regime transitions follow a Markov chain with transition matrix P.

This is a deliberate simplification. Real price dynamics are harder to model
(long memory, microstructure noise, etc.), but this captures the key features
we need: volatility clustering and occasional jumps.

Note on latency
---------------
We implement a simple latency model where the MM observes the book state
with a delay of `latency_steps`. In practice latency is microseconds, but
at the scale of our simulation (1 step ≈ 1 second), we use step-counts.

At low latency: MM sees current book, computes quotes, submits.
At high latency: MM quotes on stale data → adverse selection increases.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.matching_engine import MatchingEngine, Order, OrderType, Side
from src.participants.traders import (
    InformedTrader,
    LiquidityTaker,
    NoiseTrader,
    Participant,
    create_participant_pool,
)
from src.strategies.avellaneda_stoikov import ASParams, AvellanedaStoikov


# ---------------------------------------------------------------------------
# Price process
# ---------------------------------------------------------------------------

@dataclass
class RegimeParams:
    name: str
    mu: float       # drift per step
    sigma: float    # vol per step
    jump_intensity: float   # Poisson intensity per step
    jump_sigma: float       # std of log-jump


@dataclass
class SimConfig:
    n_steps: int = 5000
    initial_price: float = 100.0
    tick_size: float = 0.01
    latency_steps: int = 0          # MM latency in steps
    informed_fraction: float = 0.2  # fraction of participants that are informed
    n_noise_traders: int = 15
    n_informed_traders: int = 3
    regimes: Optional[List[RegimeParams]] = None
    regime_transition: Optional[np.ndarray] = None   # transition matrix
    mm_params: Optional[ASParams] = None
    seed: int = 42

    def __post_init__(self):
        if self.regimes is None:
            self.regimes = [
                RegimeParams("low_vol",  mu=0.0, sigma=0.002, jump_intensity=0.005, jump_sigma=0.003),
                RegimeParams("high_vol", mu=0.0, sigma=0.008, jump_intensity=0.02,  jump_sigma=0.01),
            ]
        if self.regime_transition is None:
            # Low-vol is persistent; high-vol reverts faster
            self.regime_transition = np.array([[0.98, 0.02], [0.10, 0.90]])


class PriceProcess:
    """
    Regime-switching GBM with jumps.
    Regime transitions: discrete-time Markov chain.
    """

    def __init__(self, config: SimConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.price: float = config.initial_price
        self.regime: int = 0
        self._regimes = config.regimes
        self._P = config.regime_transition

    def step(self) -> Tuple[float, int]:
        """Advance one step; returns (new_price, current_regime)."""
        cfg = self._regimes[self.regime]
        # Diffusion
        ret = cfg.mu + cfg.sigma * self.rng.normal()
        # Jump
        if self.rng.random() < cfg.jump_intensity:
            ret += self.rng.normal(0, cfg.jump_sigma)
        self.price = max(0.01, self.price * (1 + ret))

        # Regime transition
        probs = self._P[self.regime]
        self.regime = int(self.rng.choice(len(probs), p=probs))

        return self.price, self.regime


# ---------------------------------------------------------------------------
# Simulation state snapshot
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    step: int
    timestamp: float
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    regime: int
    ofi: float
    mm_inventory: int
    mm_cash: float
    mm_total_pnl: float
    mm_spread_capture: float
    mm_adverse_selection: float
    bid_depth_1: int
    ask_depth_1: int
    total_trades: int
    regime_name: str


# ---------------------------------------------------------------------------
# Main simulation engine
# ---------------------------------------------------------------------------

class MarketSimulator:
    """
    Full simulation loop integrating all components.

    Architecture
    ------------
    Engine
     ├─ PriceProcess (fundamental value)
     ├─ MatchingEngine (order book)
     ├─ [Participants]  (noise, informed, institutional)
     └─ MarketMaker (Avellaneda-Stoikov)

    Each step:
      1. Advance fundamental price
      2. Stale-quote check (latency model)
      3. Participants submit orders
      4. MM computes and submits quotes
      5. All orders processed by matching engine
      6. MM receives fill notifications
      7. Analytics recorded
    """

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.engine = MatchingEngine()
        self.price_process = PriceProcess(self.config, np.random.default_rng(self.config.seed + 1))
        self.mm = AvellanedaStoikov(self.config.mm_params)
        self.participants: List[Participant] = create_participant_pool(
            n_noise=self.config.n_noise_traders,
            n_informed=self.config.n_informed_traders,
            seed=self.config.seed + 2,
        )
        self.records: List[StepRecord] = []
        self._step = 0
        self._book_history: Deque[dict] = deque(maxlen=self.config.latency_steps + 1)

        # Seed the book with initial quotes
        self._bootstrap_book()

    def _bootstrap_book(self) -> None:
        """Place initial quotes to seed the book."""
        p0 = self.config.initial_price
        ts = self.config.mm_params.dt if self.config.mm_params else 0.0
        for offset in [0.05, 0.10, 0.20, 0.50]:
            for side, price in [(Side.BID, p0 - offset), (Side.ASK, p0 + offset)]:
                order = Order(
                    order_id=f"BOOT_{side.name}_{offset}",
                    participant_id="BOOTSTRAP",
                    side=side,
                    order_type=OrderType.LIMIT,
                    price=round(price, 4),
                    quantity=10,
                    timestamp=0.0,
                )
                self.engine.submit(order)

    def run(self) -> pd.DataFrame:
        """Run the full simulation; return DataFrame of step records."""
        cfg = self.config
        mm_params = cfg.mm_params or ASParams()

        for step in range(cfg.n_steps):
            self._step = step
            timestamp = step * mm_params.dt

            # 1. Advance fundamental price
            new_price, regime = self.price_process.step()

            # 2. Get (possibly stale) book state for MM
            if len(self._book_history) >= cfg.latency_steps + 1:
                stale_snap = self._book_history[0]
            else:
                stale_snap = self.engine.snapshot()

            # 3. Participants submit orders
            mid = self.engine.mid_price or new_price
            spread = self.engine.spread or 0.02
            book_state = self.engine.snapshot()

            for participant in self.participants:
                orders = participant.generate_orders(timestamp, mid, spread, book_state)
                for o in orders:
                    self.engine.submit(o)

            # 4. MM quotes (using potentially stale data)
            mm_mid = stale_snap.get("mid_price", mid) if stale_snap else mid
            if self.mm.should_trade():
                t_normalised = (step % 390) / 390.0  # reset each "session"
                current_mid = (self.engine.mid_price or new_price)
                bid, ask = self.mm.compute_quotes(current_mid, self.mm.state.inventory, t_normalised)
                self.mm.update_volatility(new_price)

                # Cancel old quotes
                if self.mm.state.active_bid_id:
                    self.engine.cancel(self.mm.state.active_bid_id)
                if self.mm.state.active_ask_id:
                    self.engine.cancel(self.mm.state.active_ask_id)

                # Submit new quotes
                bid_order = Order(
                    order_id=f"MM_BID_{step}",
                    participant_id="MM",
                    side=Side.BID,
                    order_type=OrderType.LIMIT,
                    price=bid,
                    quantity=mm_params.order_size,
                    timestamp=timestamp,
                )
                ask_order = Order(
                    order_id=f"MM_ASK_{step}",
                    participant_id="MM",
                    side=Side.ASK,
                    order_type=OrderType.LIMIT,
                    price=ask,
                    quantity=mm_params.order_size,
                    timestamp=timestamp,
                )
                self.engine.submit(bid_order)
                self.engine.submit(ask_order)
                self.mm.state.active_bid_id = f"MM_BID_{step}"
                self.mm.state.active_ask_id = f"MM_ASK_{step}"

            # 5. Process MM fills
            for fr in self.engine.fill_reports:
                if fr.participant_id == "MM":
                    side_str = "bid" if fr.fill_price <= (mid or 0) else "ask"
                    # Check if filled by informed trader
                    trade = next(
                        (t for t in self.engine.trades if
                         t.maker_order_id == fr.order_id or t.taker_order_id == fr.order_id),
                        None,
                    )
                    is_informed = False
                    if trade:
                        counterparty = (
                            trade.taker_participant_id
                            if trade.maker_participant_id == "MM"
                            else trade.maker_participant_id
                        )
                        is_informed = counterparty.startswith("INFORMED")
                    self.mm.on_fill(side_str, fr.fill_price, fr.fill_qty, mid, is_informed)
            self.engine.fill_reports.clear()

            # 6. Recalibrate MM params periodically
            if step % 100 == 0 and step > 0:
                self.mm.recalibrate_kappa()

            # 7. Record analytics
            snap = self.engine.snapshot()
            regime_name = cfg.regimes[regime].name if cfg.regimes else "unknown"
            bid_depth = snap["bid_depth"]
            ask_depth = snap["ask_depth"]
            total_pnl = self.mm.mark_to_market(snap.get("mid_price") or new_price)

            record = StepRecord(
                step=step,
                timestamp=timestamp,
                mid_price=snap.get("mid_price") or new_price,
                best_bid=snap.get("best_bid") or 0.0,
                best_ask=snap.get("best_ask") or 0.0,
                spread=snap.get("spread") or 0.0,
                regime=regime,
                ofi=snap.get("ofi") or 0.0,
                mm_inventory=self.mm.state.inventory,
                mm_cash=self.mm.state.cash,
                mm_total_pnl=total_pnl,
                mm_spread_capture=self.mm.state.spread_capture,
                mm_adverse_selection=self.mm.state.adverse_selection_cost,
                bid_depth_1=bid_depth[0][1] if bid_depth else 0,
                ask_depth_1=ask_depth[0][1] if ask_depth else 0,
                total_trades=snap.get("total_trades") or 0,
                regime_name=regime_name,
            )
            self.records.append(record)
            self._book_history.append(snap)

        return self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.records])

    def run_experiments(self, param_grid: List[Dict]) -> pd.DataFrame:
        """
        Run multiple simulations with different parameters.
        Returns a combined DataFrame with a 'run_id' column.
        """
        all_results = []
        for i, params in enumerate(param_grid):
            cfg = SimConfig(**params)
            sim = MarketSimulator(cfg)
            df = sim.run()
            df["run_id"] = i
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    df[f"param_{k}"] = v
            all_results.append(df)
        return pd.concat(all_results, ignore_index=True)
