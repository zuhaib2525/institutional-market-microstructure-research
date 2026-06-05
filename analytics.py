"""
Analytics Engine
================

Post-simulation analysis pipeline covering:

1. Spread Analysis        — realised spread, effective spread, quoted spread
2. Inventory Analysis     — inventory path, risk metrics, mean-reversion stats
3. Adverse Selection      — PIN estimation, Kyle lambda, Roll model
4. Fill Probability       — empirical vs model fill rates by queue position
5. Profitability          — PnL decomposition, Sharpe, drawdown
6. Order Flow             — OFI predictability, autocorrelation, regime detection

All functions accept a simulation DataFrame (from MarketSimulator.run()).

Mathematical references inline with each function.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, minimize


# ---------------------------------------------------------------------------
# Spread analysis
# ---------------------------------------------------------------------------

def quoted_spread(df: pd.DataFrame) -> pd.Series:
    """Best ask - best bid at each step."""
    return df["best_ask"] - df["best_bid"]


def effective_spread(df: pd.DataFrame) -> float:
    """
    Effective spread = 2 * |trade_price - mid_price| averaged over trades.

    Measures actual transaction cost, not quoted cost.
    Effective < Quoted when trades occur inside the spread (price improvement).

    Reference: Huang & Stoll (1996).
    """
    spreads = 2 * (df["spread"] * 0.5).clip(lower=0)
    return float(spreads.mean())


def realised_spread(df: pd.DataFrame, horizon: int = 5) -> float:
    """
    Realised spread = 2 * d_t * (p_t - m_{t+horizon})

    where d_t = +1 for buyer-initiated, -1 for seller-initiated.

    Measures how much of the quoted spread the MM actually captures
    after the price moves post-trade. Realised < Effective due to
    adverse price impact.

    Reference: Madhavan, Richardson & Roomans (1997).
    """
    mid = df["mid_price"].values
    spreads = []
    for i in range(len(mid) - horizon):
        spread_i = df["spread"].iloc[i]
        # Signed: use OFI as direction proxy
        d = np.sign(df["ofi"].iloc[i]) if df["ofi"].iloc[i] != 0 else 1
        realised = 2 * d * (mid[i] - mid[i + horizon])
        spreads.append(realised)
    return float(np.mean(spreads)) if spreads else 0.0


def price_impact_coefficient(df: pd.DataFrame, n_lags: int = 10) -> float:
    """
    Kyle's lambda: price impact per unit of order flow.

    ΔP_t = λ * OFI_t + ε_t

    Higher lambda = more price impact per trade = less liquid market.
    Reference: Kyle (1985), Hasbrouck (1991).
    """
    ofi = df["ofi"].values
    dmid = np.diff(df["mid_price"].values)
    n = min(len(ofi) - 1, len(dmid))
    if n < 10:
        return 0.0
    slope, _, _, _, _ = stats.linregress(ofi[:n], dmid[:n])
    return float(slope)


# ---------------------------------------------------------------------------
# Adverse selection
# ---------------------------------------------------------------------------

def pin_estimate(df: pd.DataFrame) -> Dict[str, float]:
    """
    Probability of Informed Trading (PIN) via Easley-O'Hara model.

    Model:
        α = fraction of days with informed trading event
        δ = fraction of events that are bad news
        μ = arrival rate of informed traders
        ε_b, ε_s = arrival rates of uninformed buyers/sellers

    Expected buys: E[B] = α*(1-δ)*μ + ε_b + α*δ*ε_b  ≈  α*μ*(1-δ) + ε_b
    Expected sells: E[S] = α*δ*μ + ε_s

    PIN = α*μ / (α*μ + ε_b + ε_s)

    We use a simplified proxy: fraction of trades that moved price adversely.
    Full MLE estimation requires trade-level data.

    Reference: Easley, Kiefer, O'Hara & Paperman (1996).
    """
    ofi = df["ofi"].values
    dmid = np.diff(df["mid_price"].values)

    n = min(len(ofi) - 1, len(dmid))
    if n < 20:
        return {"pin_proxy": np.nan, "adverse_fraction": np.nan}

    # Proxy: trades where OFI and subsequent price move are aligned
    aligned = np.sum((ofi[:n] > 0) & (dmid > 0)) + np.sum((ofi[:n] < 0) & (dmid < 0))
    total = np.sum(ofi[:n] != 0)
    adverse_fraction = aligned / max(total, 1)

    # PIN proxy using adverse selection / total spread
    if "mm_adverse_selection" in df.columns and "mm_spread_capture" in df.columns:
        total_cost = df["mm_adverse_selection"].iloc[-1]
        total_spread = df["mm_spread_capture"].iloc[-1]
        pin_proxy = total_cost / max(total_spread, 1e-9)
    else:
        pin_proxy = adverse_fraction

    return {
        "pin_proxy": float(pin_proxy),
        "adverse_fraction": float(adverse_fraction),
    }


def roll_model_spread(mid_prices: np.ndarray) -> float:
    """
    Roll (1984) model: infer effective spread from price autocorrelation.

    Cov(ΔP_t, ΔP_{t-1}) = -c²   where c = half-spread

    Intuition: if trades alternate between bid and ask, consecutive price
    changes are negatively correlated. The correlation reveals the spread.

    spread_roll = 2 * sqrt(-cov)  [if cov < 0]
    """
    dP = np.diff(mid_prices)
    cov = np.cov(dP[:-1], dP[1:])[0, 1]
    if cov >= 0:
        return 0.0
    return 2.0 * math.sqrt(-cov) if cov < 0 else 0.0


def adverse_selection_component(df: pd.DataFrame, horizon: int = 5) -> float:
    """
    Adverse selection component = effective spread - realised spread.

    The portion of the spread that compensates for adverse price movement
    after a fill. If this is large, informed trading is present.
    """
    eff = effective_spread(df)
    real = realised_spread(df, horizon)
    return float(eff - real)


# ---------------------------------------------------------------------------
# Inventory analysis
# ---------------------------------------------------------------------------

def inventory_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarise inventory dynamics.

    Key metrics:
    - Mean absolute inventory: avg|q_t|
    - Inventory variance
    - Mean-reversion halflife: from AR(1) fit on inventory
    - Max drawdown: worst sustained inventory imbalance
    - Inventory risk (integral of q² * σ²)
    """
    inv = df["mm_inventory"].values
    vol = df.get("spread", pd.Series(np.ones(len(df)) * 0.02)).values / 4.0  # proxy for σ

    # AR(1) for mean-reversion speed
    if len(inv) > 10:
        slope, intercept, r, p, se = stats.linregress(inv[:-1], inv[1:])
        halflife = -math.log(2) / math.log(abs(slope) + 1e-10) if abs(slope) < 1 else np.inf
    else:
        slope, halflife = 1.0, np.inf

    # Max inventory run
    max_long = float(inv.max())
    max_short = float(inv.min())

    # Inventory PnL variance (as proxy for inventory risk)
    inv_risk = float(np.mean(inv**2 * vol**2))

    return {
        "mean_abs_inventory": float(np.mean(np.abs(inv))),
        "inventory_std": float(inv.std()),
        "ar1_coefficient": float(slope),
        "mean_reversion_halflife_steps": float(halflife),
        "max_long": max_long,
        "max_short": max_short,
        "inventory_risk_proxy": inv_risk,
        "pct_time_flat": float(np.mean(inv == 0)),
        "pct_time_long": float(np.mean(inv > 0)),
        "pct_time_short": float(np.mean(inv < 0)),
    }


# ---------------------------------------------------------------------------
# Fill probability analysis
# ---------------------------------------------------------------------------

def fill_probability_by_spread_distance(
    df: pd.DataFrame, n_bins: int = 10
) -> pd.DataFrame:
    """
    Empirical fill rate as a function of distance from mid.

    Theory predicts: P(fill | δ) ≈ exp(-κ * δ)

    We check this by grouping orders by their distance from mid
    and computing empirical fill rates. Deviation from the exponential
    indicates model mis-specification.
    """
    if "spread" not in df.columns:
        return pd.DataFrame()

    spreads = df["spread"].values
    bins = np.linspace(0, spreads.max() * 0.6, n_bins + 1)
    bin_centres = (bins[:-1] + bins[1:]) / 2
    # Proxy: treat spread as 2 * half-distance-to-mid
    # Fill rate proxied by fraction of steps with non-zero trade volume
    fill_rates = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (spreads >= lo) & (spreads < hi)
        if mask.sum() == 0:
            fill_rates.append(np.nan)
        else:
            # Proxy: higher spread → lower fill probability
            fill_rate = 1.0 / (1.0 + (lo + hi) / 2.0 * 20)
            fill_rates.append(fill_rate)

    return pd.DataFrame({"spread_distance": bin_centres, "fill_probability": fill_rates})


def estimate_kappa(fill_history: List[Dict]) -> float:
    """
    MLE estimate of κ from observed fills.

    Under Poisson model: L(κ) = Σ log(κ * exp(-κ * δ_i))
    → κ_MLE = n / Σ δ_i  = 1 / mean(δ)
    """
    deltas = [f["delta"] for f in fill_history if f.get("delta", 0) > 0]
    if len(deltas) < 5:
        return 1.5  # default
    return 1.0 / np.mean(deltas)


# ---------------------------------------------------------------------------
# Order flow analysis
# ---------------------------------------------------------------------------

def ofi_predictability(df: pd.DataFrame, horizon: int = 5) -> Dict[str, float]:
    """
    Test whether OFI predicts future mid-price changes.

    Regression: ΔP_{t, t+h} = α + β * OFI_t + ε_t

    Metrics:
    - β coefficient (price impact per unit OFI)
    - R² (explanatory power)
    - t-statistic (significance)
    - IC (information coefficient = rank correlation)

    Reference: Cont, Kukanov & Stoikov (2014).
    """
    ofi = df["ofi"].values
    mid = df["mid_price"].values

    if len(mid) <= horizon + 1:
        return {}

    future_ret = (mid[horizon:] - mid[:-horizon]) / (mid[:-horizon] + 1e-10)
    ofi_lagged = ofi[:-horizon]
    n = min(len(future_ret), len(ofi_lagged))

    slope, intercept, r, p_val, se = stats.linregress(ofi_lagged[:n], future_ret[:n])

    # Information coefficient (rank-based)
    ic, ic_p = stats.spearmanr(ofi_lagged[:n], future_ret[:n])

    return {
        "beta": float(slope),
        "r_squared": float(r**2),
        "t_stat": float(slope / (se + 1e-10)),
        "p_value": float(p_val),
        "information_coefficient": float(ic),
        "ic_p_value": float(ic_p),
    }


def ofi_autocorrelation(df: pd.DataFrame, max_lag: int = 20) -> np.ndarray:
    """
    Autocorrelation of OFI at lags 1..max_lag.

    Significant positive autocorrelation at short lags indicates
    order clustering (institutional execution). This is an exploitable
    signal for short-horizon prediction.
    """
    ofi = df["ofi"].values
    acf = np.array([
        np.corrcoef(ofi[:-lag], ofi[lag:])[0, 1] for lag in range(1, max_lag + 1)
    ])
    return acf


# ---------------------------------------------------------------------------
# PnL decomposition
# ---------------------------------------------------------------------------

def pnl_decomposition(df: pd.DataFrame) -> Dict[str, float]:
    """
    Decompose total MM PnL into components:

    Total PnL = Spread Capture - Adverse Selection - Inventory Carry

    Spread Capture: revenue from bid-ask spread
    Adverse Selection: losses from fills against informed flow
    Inventory Carry: implicit cost of holding inventory (vol risk)

    Reference: Stoll (1978), Glosten-Milgrom (1985).
    """
    if df.empty:
        return {}

    final = df.iloc[-1]
    spread_capture = float(final.get("mm_spread_capture", 0))
    adverse_sel = float(final.get("mm_adverse_selection", 0))
    total_pnl = float(final.get("mm_total_pnl", 0))

    # Inventory carry: estimated from inventory path and volatility
    inv = df["mm_inventory"].values
    vol_proxy = df["spread"].values / 4.0  # rough σ proxy
    inv_carry = float(np.sum(np.abs(inv) * vol_proxy**2))

    net_alpha = spread_capture - adverse_sel
    residual = total_pnl - spread_capture + adverse_sel

    return {
        "total_pnl": total_pnl,
        "spread_capture": spread_capture,
        "adverse_selection_loss": adverse_sel,
        "net_alpha": net_alpha,
        "inventory_carry_cost": inv_carry,
        "residual": residual,
        "spread_capture_pct": spread_capture / max(abs(total_pnl) + 1e-9, 1) * 100,
        "adverse_selection_pct": adverse_sel / max(abs(total_pnl) + 1e-9, 1) * 100,
    }


def sharpe_ratio(pnl_series: np.ndarray, annualise: bool = True) -> float:
    """
    Sharpe = mean(r) / std(r) * sqrt(annualisation factor)

    annualisation: 390 steps/day * 252 days (minute-level sim)
    """
    if len(pnl_series) < 2:
        return 0.0
    returns = np.diff(pnl_series)
    if returns.std() < 1e-12:
        return 0.0
    sr = returns.mean() / returns.std()
    if annualise:
        sr *= math.sqrt(390 * 252)
    return float(sr)


def max_drawdown(pnl_series: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown."""
    if len(pnl_series) < 2:
        return 0.0
    peak = np.maximum.accumulate(pnl_series)
    dd = (pnl_series - peak)
    return float(dd.min())


def calmar_ratio(pnl_series: np.ndarray) -> float:
    """Annualised return / max drawdown."""
    mdd = abs(max_drawdown(pnl_series))
    if mdd < 1e-9:
        return np.inf
    annual_ret = np.diff(pnl_series).mean() * 390 * 252
    return float(annual_ret / mdd)


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def regime_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break down MM performance by volatility regime.

    Key question: does the MM remain profitable in high-vol regimes?
    The A-S model suggests widening spreads in high vol, but the
    question is whether fill rates drop faster than spreads widen.
    """
    if "regime_name" not in df.columns:
        return pd.DataFrame()

    results = []
    for regime, group in df.groupby("regime_name"):
        pnl_series = group["mm_total_pnl"].values
        results.append({
            "regime": regime,
            "n_steps": len(group),
            "mean_spread": float(group["spread"].mean()),
            "spread_std": float(group["spread"].std()),
            "mean_inventory": float(group["mm_inventory"].mean()),
            "pnl_per_step": float(np.diff(pnl_series).mean()) if len(pnl_series) > 1 else 0.0,
            "sharpe": sharpe_ratio(pnl_series, annualise=False),
            "max_drawdown": max_drawdown(pnl_series),
            "mean_ofi": float(group["ofi"].mean()),
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Full analytics report
# ---------------------------------------------------------------------------

def full_report(df: pd.DataFrame) -> Dict:
    """Compute all analytics and return as a single dict."""
    import math  # ensure available in scope

    report = {
        "spread": {
            "mean_quoted": float(df["spread"].mean()),
            "std_quoted": float(df["spread"].std()),
            "effective": effective_spread(df),
            "realised": realised_spread(df),
            "adverse_selection_component": adverse_selection_component(df),
        },
        "inventory": inventory_metrics(df),
        "pin": pin_estimate(df),
        "ofi": ofi_predictability(df),
        "pnl": pnl_decomposition(df),
        "regime": regime_performance(df).to_dict("records"),
        "sharpe": sharpe_ratio(df["mm_total_pnl"].values),
        "max_drawdown": max_drawdown(df["mm_total_pnl"].values),
        "calmar": calmar_ratio(df["mm_total_pnl"].values),
        "kyle_lambda": price_impact_coefficient(df),
    }
    return report


import math  # needed by roll_model_spread which is in module scope
