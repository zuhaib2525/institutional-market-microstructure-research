# Mathematical Derivations

## 1. Avellaneda-Stoikov Model: Full Derivation

### 1.1 Problem Setup

We consider a market maker operating over a finite horizon [0, T]. The underlying asset price follows **arithmetic Brownian motion** (not GBM — this simplification is intentional at short horizons where log vs arithmetic is negligible but arithmetic gives cleaner analytics):

```
dS_t = σ dW_t
```

where W_t is a standard Brownian motion and σ is the per-unit-time volatility.

**Why arithmetic BM?** Over a 1-second horizon, the difference between arithmetic and geometric BM is O(σ²dt), which is ~10⁻⁸ for σ=1% and dt=1 second. The analytics are cleaner and the results identical at this scale.

The MM's state is described by:
- S_t: mid price
- X_t: cash position
- q_t: inventory (signed, in units of asset)

The MM simultaneously quotes a bid price b_t and ask price a_t. When these are hit:
- **Ask fill**: Sell 1 unit at a_t. Cash increases by a_t, inventory decreases by 1.
- **Bid fill**: Buy 1 unit at b_t. Cash decreases by b_t, inventory increases by 1.

### 1.2 Order Arrival Model

Order arrivals are modelled as Poisson processes whose intensities depend on the spread:

```
λ_a(δ_a) = A · exp(-κ · δ_a)   (ask fills, δ_a = a_t - S_t > 0)
λ_b(δ_b) = A · exp(-κ · δ_b)   (bid fills, δ_b = S_t - b_t > 0)
```

**Intuition**: The further your quote from mid, the less likely it is to be hit. The exponential form is the maximum entropy distribution consistent with a rate-of-decay constraint.

**Parameters**:
- A: Base arrival intensity (orders per unit time at zero spread)
- κ: Sensitivity of arrival rate to distance from mid

### 1.3 The MM's Optimisation Problem

The MM maximises expected CARA (Constant Absolute Risk Aversion) utility of terminal wealth:

```
max_{b_t, a_t} E[-exp(-γ · W_T)]
```

where W_T = X_T + q_T · S_T is terminal wealth and γ > 0 is risk aversion.

**Why CARA?** Two reasons:
1. It leads to a tractable HJB equation.
2. It penalises variance: E[-exp(-γW)] ≈ -exp(-γ(E[W] - γ/2 · Var[W])) for small γ.
   So maximising E[-exp(-γW)] ≈ maximising E[W] - γ/2 · Var[W], which is a mean-variance objective.

### 1.4 Hamilton-Jacobi-Bellman Equation

Define the value function:
```
u(x, q, s, t) = max_{b,a} E[-exp(-γ · W_T) | X_t=x, q_t=q, S_t=s]
```

By dynamic programming (Bellman principle), u satisfies the HJB PDE:

```
∂u/∂t + ½σ²∂²u/∂s² 
  + max_{δ_a} λ_a(δ_a)[u(x + s + δ_a, q-1, s, t) - u(x,q,s,t)]
  + max_{δ_b} λ_b(δ_b)[u(x - s + δ_b, q+1, s, t) - u(x,q,s,t)] = 0
```

with terminal condition: u(x, q, s, T) = -exp(-γ(x + qs)).

### 1.5 The Ansatz

We guess a solution of the form:
```
u(x, q, s, t) = -exp(-γ(x + qs - θ(q, t)))
```

where θ(q, t) is a function to be determined (it represents the "certainty equivalent" inventory penalty).

**Substituting into HJB** (after algebra):

```
∂θ/∂t + ½σ²γq² + max_{δ_a} A·exp(-κδ_a)[exp(γ(δ_a + θ(q,t) - θ(q-1,t))) - 1]/γ
       + max_{δ_b} A·exp(-κδ_b)[exp(γ(δ_b + θ(q,t) - θ(q+1,t))) - 1]/γ = 0
```

### 1.6 Optimising Over Spreads

For the ask side, maximise over δ_a:
```
f(δ_a) = A·exp(-κδ_a) · [exp(γ(δ_a + Δθ_a)) - 1]
```

where Δθ_a = θ(q,t) - θ(q-1,t).

Taking d/dδ_a = 0:
```
-κ·f + A·exp(-κδ_a)·γ·exp(γ(δ_a + Δθ_a)) = 0
→ δ_a* = (1/γ)·ln(1 + γ/κ) - Δθ_a
```

Similarly for the bid side:
```
δ_b* = (1/γ)·ln(1 + γ/κ) + Δθ_b
```

where Δθ_b = θ(q+1,t) - θ(q,t).

### 1.7 The PDE for θ

Substituting optimal spreads back into the HJB gives an ODE for θ(q,t):

```
∂θ/∂t + ½σ²γq² + (A/κ)·exp(-(κ/γ)·ln(1+γ/κ))·[exp(γΔθ_a) + exp(-γΔθ_b)] = 0
```

For the approximation where γ is small (risk aversion is not too extreme), using:
```
θ(q, t) ≈ ½γσ²(T-t)q²
```

This is exact when Δθ terms are small. Substituting:

```
Δθ_a = θ(q,t) - θ(q-1,t) ≈ ½γσ²(T-t)(q² - (q-1)²) ≈ γσ²(T-t)(q - ½)
```

### 1.8 Reservation Price and Optimal Spread

The optimal ask quote is:
```
a* = S_t + δ_a* = S_t + (1/γ)ln(1+γ/κ) - Δθ_a
```

The bid quote is:
```
b* = S_t - δ_b* = S_t - (1/γ)ln(1+γ/κ) - Δθ_b
```

The **midpoint** of these quotes is the **reservation price**:
```
r = (a* + b*)/2 = S_t - q·γ·σ²·(T-t)
```

The **optimal spread** is:
```
δ* = a* - b* = (2/γ)·ln(1+γ/κ) + γ·σ²·(T-t)
```

These are the key results (equations 3.4 and 3.5 in the original paper).

### 1.9 Limiting Cases

**As T → t (session end)**: δ* → (2/γ)ln(1+γ/κ). The spread narrows to a constant floor.

**As γ → 0 (risk neutral)**: First term → 0, second term → 2/κ. Risk-neutral MM quotes a spread of 2/κ regardless of time or vol.

**As σ → 0**: δ* → (2/γ)ln(1+γ/κ). No inventory risk, so spread only reflects order flow uncertainty.

**As κ → ∞ (infinitely responsive order flow)**: δ* → 0. Infinitely elastic demand means any positive spread would lose all fills to a MM quoting at mid.

---

## 2. Inventory Risk Quantification

### 2.1 The Cost of Holding Inventory

The instantaneous inventory risk (certainty-equivalent cost of current inventory):

```
IR(q, t) = ½ · γ · σ² · (T-t) · q²
```

This is the penalty term in the value function. Interpretation: holding q units costs the equivalent of ½γσ²(T-t)q² in certain wealth.

**Marginal risk**: Adding one more unit of inventory when q is already large:
```
∂IR/∂q = γ · σ² · (T-t) · q
```

This is linear in q — each additional unit is costlier than the previous.

### 2.2 Mean-Reversion Speed

If we model inventory as an AR(1) process:
```
q_{t+1} = ρ · q_t + η_t
```

The half-life of inventory is:
```
HL = -ln(2) / ln(ρ)
```

Under optimal A-S quoting, ρ depends on γ. Higher γ → stronger quote skew → faster inventory reversion → lower ρ → shorter half-life.

### 2.3 Optimal Inventory Skew

The amount by which quotes are shifted relative to zero-inventory quotes:
```
quote_skew = q · γ · σ² · (T-t)
```

When q = max_inventory, skew ≈ half the spread — so both quotes are on the same side of mid.

---

## 3. Fill Probability Estimation

### 3.1 Exponential Model (A-S)

Under the Poisson arrival model with intensity λ(δ) = A·exp(-κδ):

The probability of at least one fill in time interval [t, t+dt]:
```
P(fill in dt | δ) = λ(δ) dt = A·exp(-κδ) dt
```

Over a finite horizon Δt:
```
P(fill in [t, t+Δt] | δ) = 1 - exp(-λ(δ)·Δt) = 1 - exp(-A·exp(-κδ)·Δt)
```

For small A·Δt: P(fill) ≈ A·exp(-κδ)·Δt

### 3.2 Queue Position Adjustment

In a real LOB, fill probability depends on queue position, not just spread:

```
P(fill | δ, V_ahead) ≈ exp(-κ · δ) · exp(-V_ahead / V_scale)
```

where V_ahead is volume ahead in queue and V_scale is the scale of incoming volume.

More precisely: let λ_trade = expected trade volume per unit time at price level. Then:
```
P(fill at time T) ≈ 1 - exp(-λ_trade · T) · (volume of queue ahead / total queue)
```

This is the core of queue position theory — being first in queue can increase fill probability by 5–10× compared to being last.

### 3.3 MLE Estimation of κ

Given a sample of N quotes with:
- δ_i = distance from mid
- f_i = 1 if filled, 0 otherwise

Under the model P(f_i=1) = A·exp(-κδ_i) (adjusted for exposure time), the log-likelihood is:

```
ℓ(κ, A) = Σ_i [f_i · log(A·exp(-κδ_i)) + (1-f_i) · log(1 - A·exp(-κδ_i))]
```

For large samples with most fills (A·exp(-κδ_i) < 0.3), this simplifies:
```
κ_MLE ≈ 1 / mean(δ among filled orders)
```

This is the estimator we implement in `recalibrate_kappa()`.

---

## 4. Adverse Selection Models

### 4.1 Glosten-Milgrom Sequential Trade Model

At each time step, a single trader arrives. With probability α, they are **informed** and know true value V. With probability 1-α, they are **uninformed** and trade randomly.

The MM updates beliefs about V using Bayes' theorem:

**After observing a buy**:
```
E[V | buy] = E[V | informed buy] · P(informed | buy) + E[V | uninformed buy] · P(uninformed | buy)
```

Under the model: P(informed buy | buy) = α·P(buy|informed) / P(buy), and P(buy|informed) depends on whether V > ask.

**Key result**: In equilibrium, the spread is:
```
Spread ≥ 2 · α · (V_high - V_low)
```

where V_high and V_low are the values in the good-news and bad-news states.

### 4.2 Kyle's Lambda Estimation

The price impact coefficient λ from:
```
ΔP_t = λ · OFI_t + ε_t
```

is estimated by OLS. Under Kyle's model:
```
λ = σ_v / (2 · √(λ_noise · T))
```

where σ_v = fundamental value vol, λ_noise = noise trader volume rate.

Higher λ = more informed trading relative to noise. This is a direct measure of market "toxicity".

### 4.3 PIN (Probability of Informed Trading)

Full Easley-O'Hara MLE model. Let:
- α = probability of information event occurring per day
- δ = probability information is bad news (given event)
- μ = informed arrival rate per unit time
- ε_b, ε_s = uninformed buyer/seller arrival rates

Expected buys on an information day:
```
E[B | event] = μ · (1-δ) + ε_b    (good news: informed buy + noise buys)
E[B | no event] = ε_b
```

PIN = α·μ / (α·μ + ε_b + ε_s)

**MLE** requires numerical optimisation over 5 parameters {α, δ, μ, ε_b, ε_s} given daily buy/sell counts. Local optima are a known problem; initialise from multiple starting points.

---

## 5. Market Impact Models

### 5.1 Almgren-Chriss Linear Impact

Temporary impact of trading at rate v_t (shares per unit time):
```
g(v_t) = η · v_t   (linear)
```

Permanent impact of total order Q:
```
h(Q) = γ_perm · Q
```

**Total cost of liquidating X shares by time T**:
```
C = ∫_0^T [g(v_t) + ½h(v_t)] v_t dt + ½ · σ² · ∫_0^T x_t² dt · λ_risk
```

**Optimal trajectory** (with linear impact, risk aversion λ_risk):
```
x_t = X · sinh(κ(T-t)) / sinh(κT)
```

where κ = √(λ_risk · σ² / η).

### 5.2 Square Root Impact

Empirical market impact follows a square root law:
```
ΔP / σ = β · √(Q / ADV)
```

where ADV = average daily volume, β ≈ 0.5–1.0 (Barra model, Grinold-Kahn).

This is not derivable from first principles but is robustly observed empirically. The square root reflects the concavity of the supply curve: first few units are cheap, subsequent units become progressively more expensive.

---

## 6. Latency Model

### 6.1 Quote Staleness

If the MM observes the book with delay Δt, the mid price observed is:
```
S_obs = S_true + ε_Δt
```

where ε_Δt ~ N(0, σ²Δt) (BM increment over Δt).

The MM's quotes are computed from S_obs, so they are offset from fair value by ε_Δt.

**Adverse selection from latency**: If price moves by ε_Δt during the latency window, the MM's quote is stale by that amount. An informed trader observing the new price will selectively take the stale quote.

**Expected adverse selection cost per step**:
```
E[AS | latency=Δt] = σ · √Δt · P(quote is hit given staleness)
```

For large jumps relative to half-spread: E[AS] ≈ σ√Δt.
This grows as √Δt, confirming latency is costly but with diminishing marginal impact.

### 6.2 Optimal Cancel-Requote Policy

The MM should cancel and requote when the expected gain from fresher quotes exceeds the cancel cost:

```
Cancel if: |S_true - S_obs| > threshold_cancel
```

where threshold_cancel is determined by balancing cancel cost vs adverse selection from stale quote.

Under Brownian motion: the optimal cancel threshold grows as σ√(time_in_market).

---

## 7. Expected Value Analysis

### 7.1 Break-Even Condition

Net expected value per fill:
```
EV = P(noise fill) · spread/2 - P(informed fill) · E[adverse move]
   = (1-π) · δ - π · λ · δ
   = δ · (1 - π(1 + λ))
```

For EV > 0:
```
δ > 0 requires: 1 > π(1+λ)
or equivalently: π < 1/(1+λ)
```

With Kyle's λ ≈ 1 (typical liquid stock): π < 50%.
With λ = 0.5: π < 67%.
With λ = 2: π < 33%.

This is the fundamental constraint. A MM in a highly toxic flow environment (high π, high λ) cannot be profitable at any spread.

### 7.2 PnL Decomposition

Total PnL over N fills:
```
PnL = Σ spread_capture - Σ adverse_selection - Σ inventory_carry
    = N_fills · δ · (1-π) - N_fills · δ · π · λ - Σ q_t² · γ · σ² · dt
```

In our simulation, we track each component separately (mm_spread_capture, mm_adverse_selection, and the inventory risk as an implicit cost in the skewed quotes).

**Net alpha**: spread_capture - adverse_selection. Should be positive for a viable strategy.

---

*References: Avellaneda & Stoikov (2008), Kyle (1985), Glosten-Milgrom (1985), Almgren-Chriss (2001), Easley et al. (1996).*
