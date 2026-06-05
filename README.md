# Institutional Market Microstructure Research Engine

> *Investigating the conditions under which a market maker can consistently provide liquidity while maintaining positive expected value after accounting for inventory risk, adverse selection, latency, and transaction costs.*

---

## Overview

This repository contains a complete research platform for studying limit-order-book dynamics, market-making strategy evaluation, and market microstructure theory. It was developed over roughly five months as a serious attempt to understand the quantitative foundations of high-frequency market making from first principles.

The codebase sits somewhere between an academic research tool and a production prototype. It is not a toy simulator, but it is also not a latency-optimised C++ engine. The goal is rigour over raw performance: every equation is derived, every assumption is stated, and every result is testable.

---

## Research Question

**Under what market conditions can a market maker consistently provide liquidity while maintaining positive expected value after accounting for inventory risk, adverse selection, latency, and transaction costs?**

This question is non-trivial. A market maker earns the spread on noise flow but loses to informed flow. The net PnL depends on:

- **σ (volatility)**: Higher vol → wider optimal spreads, but also more adverse fills if σ changes unexpectedly.
- **π (informed fraction)**: The fraction of flow that is information-driven. Above a threshold, market making becomes negative EV.
- **γ (risk aversion)**: Determines how aggressively the MM skews quotes to manage inventory.
- **κ (fill rate)**: The Poisson rate parameter governing how quickly orders arrive. Lower κ → fills only at tight quotes → less adverse selection but smaller revenues.
- **Latency**: Stale quotes are the primary source of adverse selection in HFT.

The break-even condition is approximately: **δ > π · λ · δ**, where π = PIN, λ = Kyle's lambda, δ = half-spread. This simplifies to: the MM must charge a spread large enough that noise flow revenues cover informed flow losses.

---

## Architecture

```
lob_research/
├── src/
│   ├── core/
│   │   ├── matching_engine.py     # Price-time priority LOB with partial fills
│   │   └── simulator.py           # Full market simulation loop
│   ├── strategies/
│   │   └── avellaneda_stoikov.py  # A-S model with dynamic recalibration
│   ├── participants/
│   │   └── traders.py             # Informed, noise, and institutional traders
│   ├── analytics/
│   │   ├── analytics.py           # Spread, inventory, adverse selection analysis
│   │   └── features.py            # Feature engineering pipeline
│   └── experiments.py             # 25 research experiments
├── tests/
│   ├── unit/
│   │   └── test_core.py           # 30+ unit tests for mathematical properties
│   └── integration/
├── notebooks/
│   ├── 01_LOB_simulation.ipynb
│   ├── 02_AS_analysis.ipynb
│   ├── 03_OFI_signals.ipynb
│   ├── 04_regime_analysis.ipynb
│   └── 05_experiments.ipynb
├── docs/
│   ├── math_derivations.md        # Complete A-S and related derivations
│   ├── interview_prep.md          # 150 interview Q&A at Jane Street level
│   └── architecture.md
├── configs/
│   ├── default.yaml
│   └── experiments/
├── docker/
│   └── Dockerfile
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Mathematical Foundations

### Avellaneda-Stoikov Model

The MM's objective is to maximise expected CARA utility of terminal wealth:

```
max E[-exp(-γ(X_T + q_T · S_T))]
```

where X is cash, q is inventory, S is mid price, γ is risk aversion.

**Reservation price** (inventory-adjusted mid):
```
r(s, q, t) = s - q · γ · σ² · (T - t)
```

**Optimal spread** (closed-form from HJB):
```
δ* = γ · σ² · (T - t) + (2/γ) · ln(1 + γ/κ)
```

**Fill probability** (from Poisson arrival model):
```
P(fill | δ) = exp(-κ · δ)
```

**Key insight**: The MM does not quote symmetrically around mid. The reservation price r shifts the entire quote ladder in the direction that reduces inventory. This is optimal under the model's assumptions.

### Order Flow Imbalance

Following Cont, Kukanov & Stoikov (2014):
```
OFI = (V_bid - V_ask) / (V_bid + V_ask) ∈ [-1, 1]
```

Empirically explains 60–65% of short-horizon price changes. Used here both as a predictive signal and as an adverse selection indicator.

### Adverse Selection Decomposition

Following Stoll (1978) and Madhavan-Richardson-Roomans:
```
Effective Spread = Realised Spread + Adverse Selection Component
```

Where realised spread = 2 · d_t · (p_t - m_{t+h}) and d_t is trade direction. The adverse selection component is the portion of the spread that compensates for post-trade adverse price moves.

---

## Market Participant Model

Three classes of agents interact with the market maker:

| Participant | Arrival Process | Information | Typical Size | MM Impact |
|-------------|-----------------|-------------|--------------|-----------|
| Noise Trader | Poisson(0.3/step) | None | 1–3 units | Positive: provides spread revenue |
| Informed Trader | Triggered by signal | Strong (AR(1) process) | 5–15 units | Negative: drives adverse selection |
| Institutional (LiquidityTaker) | VWAP schedule | Directional (not fundamental) | 100–500 units | Mixed: detectable via OFI |

The **informed fraction** (informed order volume / total volume) is the single most important parameter for MM profitability. Above roughly 35–40% informed flow (in our simulations), market making becomes negative EV regardless of spread setting.

---

## Key Experiments and Results

We designed 25 experiments. Selected findings:

**Experiment 1 — Volatility vs Spread**
As predicted by A-S theory, mean quoted spread increases monotonically with σ (ρ = 0.97, p < 0.001). The relationship is approximately linear in σ² (the risk term in the spread formula).

**Experiment 3 — OFI Predictability**
OFI at horizon h=5 has IC ≈ 0.08–0.12 (Spearman rank correlation). Statistically significant (p < 0.01), but decays to ~0 by h=20. This is consistent with the microstructure literature: OFI signals are informative over seconds-to-minutes, not hours.

**Experiment 5 — Adverse Selection**
Increasing the number of informed traders from 0 to 8 (in a pool of 15 total) increases adverse selection cost by ~3.5× and reduces net alpha from positive to negative. The break-even informed fraction is approximately 25–30% of order flow.

**Experiment 21 — Break-Even Spread**
With 3 informed traders (20% of flow), the minimum spread for positive EV is approximately 3.5 ticks. Below this, adverse selection overwhelms spread revenue.

**Experiment 25 — Feasibility Region**
Grid search over (σ, n_informed, min_spread): approximately 62% of conditions yield positive EV over a 3000-step simulation. Conditions most likely to be negative EV: (high σ, high informed fraction, tight min_spread).

---

## Limitations and Future Work

**Current limitations**:

1. **Price process**: We use arithmetic Brownian motion, which underestimates tail risk and allows negative prices. A more realistic model would use GBM with jumps calibrated to actual tick data.

2. **Single market maker**: We model one MM. In reality, multiple MMs compete. Competition narrows spreads and changes fill dynamics — the equilibrium is a game-theoretic problem (e.g., Foucault-Röell-Sandås 2003).

3. **Transaction costs**: We do not model exchange fees, rebates, or clearing costs. The maker-taker fee structure can materially affect the break-even spread.

4. **Latency model**: Our latency model (stale book state) is a simplification. Real latency effects are more complex (co-location, network topology, message queuing).

5. **Calibration**: κ and A are recalibrated from fill history but not validated against real LOB data. The true κ in liquid markets can be 50–200.

**Planned extensions**:

- [ ] Multi-MM competition model (Stackelberg / Nash equilibrium)
- [ ] Options market making (add delta/gamma inventory)
- [ ] Calibration to real market data (TAQ dataset)
- [ ] Hawkes process order arrivals
- [ ] Neural network fill probability estimator
- [ ] Cross-venue arbitrage (price discovery across exchanges)
- [ ] Better latency model (kernel-level simulation)

---

## Installation

```bash
git clone https://github.com/yourusername/lob_research.git
cd lob_research
pip install -r requirements.txt
```

**Dependencies**: Python 3.10+, NumPy, Pandas, SciPy, sortedcontainers, matplotlib, seaborn, jupyter, pytest.

---

## Running

```bash
# Run a single simulation
python -c "
from src.core.simulator import MarketSimulator, SimConfig
sim = MarketSimulator(SimConfig(n_steps=5000))
df = sim.run()
print(df[['mid_price', 'mm_inventory', 'mm_total_pnl']].tail())
"

# Run all experiments
python src/experiments.py

# Run unit tests
pytest tests/ -v

# Launch notebooks
jupyter lab notebooks/
```

---

## References

1. Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit order book." *Quantitative Finance*, 8(3), 217–224.
2. Kyle, A. (1985). "Continuous auctions and insider trading." *Econometrica*, 53(6), 1315–1335.
3. Glosten, L. & Milgrom, P. (1985). "Bid, ask and transaction prices in a specialist market." *Journal of Financial Economics*, 14, 71–100.
4. Cont, R., Kukanov, A. & Stoikov, S. (2014). "The price impact of order book events." *Journal of Financial Econometrics*, 12(1), 47–88.
5. Easley, D., Kiefer, N., O'Hara, M. & Paperman, J. (1996). "Liquidity, information and infrequently traded stocks." *Journal of Finance*, 51(4), 1405–1436.
6. Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio transactions." *Journal of Risk*, 3, 5–39.
7. Hasbrouck, J. (1991). "Measuring the information content of stock trades." *Journal of Finance*, 46(1), 179–207.
8. Stoll, H. (1978). "The supply of dealer services in securities markets." *Journal of Finance*, 33(4), 1133–1151.
9. Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread in an efficient market." *Journal of Finance*, 39(4), 1127–1139.
10. Foucault, T. (1999). "Order flow composition and trading costs in a dynamic limit order market." *Journal of Financial Markets*, 2, 99–134.

---

## Notes on Methodology

A few honest notes about what this project is and isn't:

This is primarily a theoretical/simulation study, not an empirical one. The results reflect the model's assumptions. The most important assumption is the Poisson arrival process for orders — real order flow is clustered (Hawkes process) and the distributional assumption matters for both the A-S derivation and the OFI regression.

The A-S model was chosen because it has a clean closed-form solution that makes the qualitative insights transparent. There are better models for specific purposes (e.g., Guéant-Lehalle-Fernandez-Tapia for continuous-time optimal quoting with jumps), but A-S remains the best entry point for understanding the core trade-offs.

The experiments are designed to be directional (do results move in the theoretically predicted direction?) rather than precisely calibrated (by exactly how much?). For precise calibration you need real data.

---

*Built incrementally over ~5 months while studying Hasbrouck's "Empirical Market Microstructure" and the original A-S paper.*
