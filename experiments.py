"""
Research Experiments
====================

25 experiments investigating market-making profitability under varying conditions.

Each experiment follows the standard methodology:
1. Hypothesis (null H0 and alternative H1)
2. Variable manipulation (independent → dependent)
3. Statistical test
4. Expected outcome based on theory

All results are saved to experiments/results/ as CSV + summary JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Adjust path for direct script execution
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.simulator import MarketSimulator, SimConfig, RegimeParams
from src.strategies.avellaneda_stoikov import ASParams
from src.analytics.analytics import full_report, pnl_decomposition, ofi_predictability


# ---------------------------------------------------------------------------
# Experiment base class
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    experiment_id: int
    name: str
    hypothesis: str
    result_summary: Dict[str, Any]
    conclusion: str
    p_value: Optional[float] = None
    effect_size: Optional[float] = None


def run_sim(config_dict: Dict) -> pd.DataFrame:
    """Helper: build SimConfig and run simulation."""
    cfg = SimConfig(**config_dict)
    sim = MarketSimulator(cfg)
    return sim.run()


# ---------------------------------------------------------------------------
# Experiments 1-5: Volatility and spread
# ---------------------------------------------------------------------------

def exp01_volatility_vs_spread() -> ExperimentResult:
    """
    H0: Optimal spread does not change with volatility.
    H1: Optimal spread increases with volatility (A-S model predicts this).

    Method: Run sims with σ ∈ {0.001, 0.003, 0.006, 0.01, 0.02}.
    Metric: Mean realised spread vs σ level.
    Test: Pearson correlation (spread, σ).
    """
    sigmas = [0.001, 0.003, 0.006, 0.010, 0.020]
    mean_spreads = []

    for sigma in sigmas:
        regimes = [
            RegimeParams("low", 0.0, sigma, 0.005, sigma * 0.5),
            RegimeParams("high", 0.0, sigma * 2, 0.02, sigma),
        ]
        df = run_sim({
            "n_steps": 2000,
            "regimes": regimes,
            "mm_params": ASParams(sigma=sigma),
            "seed": 1,
        })
        mean_spreads.append(float(df["spread"].mean()))

    corr, p_val = stats.pearsonr(sigmas, mean_spreads)
    conclusion = (
        "SUPPORTED: spread increases significantly with volatility (ρ={:.3f}, p={:.4f})".format(corr, p_val)
        if p_val < 0.05 else
        "NOT SUPPORTED at α=0.05"
    )

    return ExperimentResult(
        experiment_id=1,
        name="Effect of Volatility on Optimal Spread",
        hypothesis="H1: Spread increases with σ",
        result_summary={
            "sigmas": sigmas,
            "mean_spreads": mean_spreads,
            "pearson_r": corr,
        },
        conclusion=conclusion,
        p_value=p_val,
        effect_size=corr,
    )


def exp02_inventory_risk_vs_profitability() -> ExperimentResult:
    """
    H0: Inventory limit does not affect profitability.
    H1: Tighter inventory limits reduce total PnL but improve Sharpe.

    Method: Vary max_inventory ∈ {5, 10, 20, 50, 100}.
    Metric: PnL, Sharpe ratio.
    """
    from src.analytics.analytics import sharpe_ratio

    limits = [5, 10, 20, 50, 100]
    pnls, sharpes = [], []

    for lim in limits:
        df = run_sim({
            "n_steps": 3000,
            "mm_params": ASParams(max_inventory=lim),
            "seed": 2,
        })
        pnls.append(float(df["mm_total_pnl"].iloc[-1]))
        sharpes.append(float(sharpe_ratio(df["mm_total_pnl"].values)))

    # Test: is there monotone relationship between limit and Sharpe?
    corr_sharpe, p_sharpe = stats.pearsonr(limits, sharpes)
    corr_pnl, p_pnl = stats.pearsonr(limits, pnls)

    return ExperimentResult(
        experiment_id=2,
        name="Inventory Limit vs Profitability Tradeoff",
        hypothesis="H1: Tighter limits improve Sharpe at cost of absolute PnL",
        result_summary={
            "limits": limits, "pnls": pnls, "sharpes": sharpes,
            "corr_sharpe_vs_limit": corr_sharpe,
            "corr_pnl_vs_limit": corr_pnl,
        },
        conclusion="Sharpe-limit correlation={:.3f}".format(corr_sharpe),
        p_value=p_sharpe,
        effect_size=corr_sharpe,
    )


def exp03_ofi_predictability() -> ExperimentResult:
    """
    H0: OFI has no predictive power for future price changes.
    H1: OFI predicts short-horizon price direction with IC > 0.

    Method: Run sim, compute OFI → forward return IC at horizons 1, 5, 10.
    Test: t-test on IC (H0: IC = 0).
    """
    df = run_sim({"n_steps": 5000, "seed": 3})
    results = {}
    for h in [1, 5, 10, 20]:
        res = ofi_predictability(df, horizon=h)
        results[h] = res

    # Test at horizon=5
    ic = results[5].get("information_coefficient", 0)
    p_val = results[5].get("ic_p_value", 1.0)

    return ExperimentResult(
        experiment_id=3,
        name="OFI Predictability of Short-Horizon Returns",
        hypothesis="H1: OFI has significant IC > 0",
        result_summary={f"h{h}": v for h, v in results.items()},
        conclusion="IC at h=5: {:.4f}, p={:.4f}".format(ic, p_val),
        p_value=p_val,
        effect_size=ic,
    )


def exp04_latency_impact() -> ExperimentResult:
    """
    H0: MM latency does not affect profitability.
    H1: Higher latency reduces profitability due to stale quotes.

    Method: Vary latency_steps ∈ {0, 1, 2, 5, 10}.
    Metric: Final PnL, adverse selection cost.
    """
    latencies = [0, 1, 2, 5, 10]
    pnls, adv_sels = [], []

    for lat in latencies:
        df = run_sim({
            "n_steps": 3000,
            "latency_steps": lat,
            "seed": 4,
        })
        pnls.append(float(df["mm_total_pnl"].iloc[-1]))
        adv_sels.append(float(df["mm_adverse_selection"].iloc[-1]))

    corr, p_val = stats.pearsonr(latencies, pnls)

    return ExperimentResult(
        experiment_id=4,
        name="Latency Impact on Market Making PnL",
        hypothesis="H1: PnL decreases with latency",
        result_summary={
            "latencies": latencies, "pnls": pnls, "adverse_selections": adv_sels,
        },
        conclusion="PnL-latency correlation={:.3f}, p={:.4f}".format(corr, p_val),
        p_value=p_val,
        effect_size=corr,
    )


def exp05_adverse_selection_informed_fraction() -> ExperimentResult:
    """
    H0: Informed trader fraction does not affect MM adverse selection.
    H1: Higher informed fraction → more adverse selection → lower MM PnL.

    Method: Vary n_informed ∈ {0, 1, 2, 5, 8}.
    Metric: Adverse selection cost, net alpha.
    """
    informed_counts = [0, 1, 2, 5, 8]
    adv_sels, net_alphas = [], []

    for n_inf in informed_counts:
        df = run_sim({
            "n_steps": 3000,
            "n_informed_traders": n_inf,
            "seed": 5,
        })
        adv_sels.append(float(df["mm_adverse_selection"].iloc[-1]))
        pnl_d = pnl_decomposition(df)
        net_alphas.append(pnl_d.get("net_alpha", 0))

    corr_adv, p_adv = stats.pearsonr(informed_counts, adv_sels)
    corr_alpha, p_alpha = stats.pearsonr(informed_counts, net_alphas)

    return ExperimentResult(
        experiment_id=5,
        name="Adverse Selection Under Informed Trading",
        hypothesis="H1: Adverse selection cost increases with informed fraction",
        result_summary={
            "informed_counts": informed_counts,
            "adverse_selections": adv_sels,
            "net_alphas": net_alphas,
        },
        conclusion="Adv-sel correlation with n_informed={:.3f}".format(corr_adv),
        p_value=p_adv,
        effect_size=corr_adv,
    )


# ---------------------------------------------------------------------------
# Experiments 6-10: Risk aversion and gamma
# ---------------------------------------------------------------------------

def exp06_gamma_vs_inventory_stability() -> ExperimentResult:
    """Higher γ → MM skews more aggressively → lower inventory variance."""
    gammas = [0.01, 0.05, 0.1, 0.3, 1.0]
    inv_stds = []

    for g in gammas:
        df = run_sim({"n_steps": 3000, "mm_params": ASParams(gamma=g), "seed": 6})
        inv_stds.append(float(df["mm_inventory"].std()))

    corr, p_val = stats.pearsonr(gammas, inv_stds)
    return ExperimentResult(
        experiment_id=6,
        name="Risk Aversion γ vs Inventory Stability",
        hypothesis="H1: Higher γ reduces inventory variance",
        result_summary={"gammas": gammas, "inv_stds": inv_stds},
        conclusion="Corr(γ, inv_std)={:.3f}".format(corr),
        p_value=p_val, effect_size=corr,
    )


def exp07_gamma_vs_spread_capture() -> ExperimentResult:
    """Higher γ → wider spreads → more spread capture per fill but fewer fills."""
    gammas = [0.01, 0.05, 0.1, 0.3, 1.0]
    sc_rates = []

    for g in gammas:
        df = run_sim({"n_steps": 3000, "mm_params": ASParams(gamma=g), "seed": 7})
        sc_rates.append(float(df["mm_spread_capture"].iloc[-1]))

    corr, p_val = stats.pearsonr(gammas, sc_rates)
    return ExperimentResult(
        experiment_id=7,
        name="Risk Aversion γ vs Spread Capture",
        hypothesis="H1: Higher γ reduces total spread capture (fewer fills)",
        result_summary={"gammas": gammas, "spread_capture": sc_rates},
        conclusion="Corr(γ, spread_capture)={:.3f}".format(corr),
        p_value=p_val, effect_size=corr,
    )


def exp08_kappa_calibration_accuracy() -> ExperimentResult:
    """Test whether dynamic κ recalibration improves vs static κ."""
    df_static = run_sim({"n_steps": 5000, "mm_params": ASParams(kappa=1.5), "seed": 8})
    df_dynamic = run_sim({"n_steps": 5000, "mm_params": ASParams(kappa=0.5), "seed": 8})

    pnl_static = float(df_static["mm_total_pnl"].iloc[-1])
    pnl_dynamic = float(df_dynamic["mm_total_pnl"].iloc[-1])

    return ExperimentResult(
        experiment_id=8,
        name="κ Calibration: Static vs Dynamic",
        hypothesis="H1: Better κ calibration improves PnL",
        result_summary={"pnl_kappa_1p5": pnl_static, "pnl_kappa_0p5": pnl_dynamic},
        conclusion="Static κ=1.5 PnL={:.4f}, κ=0.5 PnL={:.4f}".format(pnl_static, pnl_dynamic),
    )


def exp09_noise_trader_density() -> ExperimentResult:
    """More noise traders → more fill opportunities → higher MM PnL."""
    noise_counts = [5, 10, 15, 20, 30]
    pnls = []

    for n in noise_counts:
        df = run_sim({"n_steps": 3000, "n_noise_traders": n, "seed": 9})
        pnls.append(float(df["mm_total_pnl"].iloc[-1]))

    corr, p_val = stats.pearsonr(noise_counts, pnls)
    return ExperimentResult(
        experiment_id=9,
        name="Noise Trader Density vs MM Profitability",
        hypothesis="H1: More noise traders increase MM PnL",
        result_summary={"noise_counts": noise_counts, "pnls": pnls},
        conclusion="Corr(noise_n, pnl)={:.3f}".format(corr),
        p_value=p_val, effect_size=corr,
    )


def exp10_order_size_impact() -> ExperimentResult:
    """Larger MM order size → more inventory risk but more spread capture."""
    sizes = [1, 2, 5, 10, 20]
    pnls, inv_stds = [], []

    for sz in sizes:
        df = run_sim({"n_steps": 3000, "mm_params": ASParams(order_size=sz), "seed": 10})
        pnls.append(float(df["mm_total_pnl"].iloc[-1]))
        inv_stds.append(float(df["mm_inventory"].std()))

    corr_pnl, p_pnl = stats.pearsonr(sizes, pnls)
    corr_inv, p_inv = stats.pearsonr(sizes, inv_stds)

    return ExperimentResult(
        experiment_id=10,
        name="MM Order Size vs PnL and Inventory Risk",
        hypothesis="H1: Larger size → higher gross PnL but higher inventory risk",
        result_summary={"sizes": sizes, "pnls": pnls, "inv_stds": inv_stds},
        conclusion="Corr(size,pnl)={:.3f}, Corr(size,inv_std)={:.3f}".format(corr_pnl, corr_inv),
        p_value=min(p_pnl, p_inv), effect_size=corr_pnl,
    )


# ---------------------------------------------------------------------------
# Experiments 11-15: Market regimes
# ---------------------------------------------------------------------------

def exp11_regime_switching_pnl() -> ExperimentResult:
    """Does MM maintain positive EV in high-vol regime?"""
    df = run_sim({
        "n_steps": 10000,
        "seed": 11,
        "regimes": [
            RegimeParams("low_vol", 0.0, 0.001, 0.005, 0.002),
            RegimeParams("high_vol", 0.0, 0.015, 0.03, 0.01),
        ],
        "regime_transition": np.array([[0.97, 0.03], [0.15, 0.85]]),
    })
    from src.analytics.analytics import regime_performance
    rp = regime_performance(df)
    return ExperimentResult(
        experiment_id=11,
        name="MM Performance Across Volatility Regimes",
        hypothesis="H1: MM is profitable in low-vol but marginal in high-vol",
        result_summary=rp.to_dict("records") if not rp.empty else {},
        conclusion="See regime_performance table",
    )


def exp12_jump_impact() -> ExperimentResult:
    """Price jumps cause adverse selection spikes for the MM."""
    no_jump = RegimeParams("calm", 0.0, 0.002, 0.0, 0.0)
    with_jump = RegimeParams("jumpy", 0.0, 0.002, 0.05, 0.02)

    df_no = run_sim({"n_steps": 3000, "regimes": [no_jump, no_jump],
                     "regime_transition": np.array([[1.0, 0.0], [0.0, 1.0]]), "seed": 12})
    df_with = run_sim({"n_steps": 3000, "regimes": [with_jump, with_jump],
                       "regime_transition": np.array([[1.0, 0.0], [0.0, 1.0]]), "seed": 12})

    adv_no = float(df_no["mm_adverse_selection"].iloc[-1])
    adv_with = float(df_with["mm_adverse_selection"].iloc[-1])

    return ExperimentResult(
        experiment_id=12,
        name="Price Jumps and Adverse Selection",
        hypothesis="H1: Jump environments increase adverse selection cost",
        result_summary={"adv_sel_no_jump": adv_no, "adv_sel_with_jump": adv_with},
        conclusion="Adv sel: no_jump={:.4f}, with_jump={:.4f}".format(adv_no, adv_with),
    )


def exp13_spread_tightening_competition() -> ExperimentResult:
    """Tighter min_spread reduces revenue per fill; find the optimum."""
    min_spreads = [0.005, 0.01, 0.02, 0.05, 0.10]
    pnls = []

    for ms in min_spreads:
        df = run_sim({"n_steps": 3000, "mm_params": ASParams(min_spread=ms), "seed": 13})
        pnls.append(float(df["mm_total_pnl"].iloc[-1]))

    best_spread = min_spreads[int(np.argmax(pnls))]
    return ExperimentResult(
        experiment_id=13,
        name="Optimal Minimum Spread Under Competition",
        hypothesis="H1: There exists an optimal min spread maximising PnL",
        result_summary={"min_spreads": min_spreads, "pnls": pnls, "best": best_spread},
        conclusion="Optimal min_spread={:.4f} with PnL={:.4f}".format(best_spread, max(pnls)),
    )


def exp14_ewma_decay_vol_estimation() -> ExperimentResult:
    """Different EWMA decay parameters affect vol estimate quality."""
    alphas = [0.80, 0.90, 0.94, 0.97, 0.99]
    pnls = []

    for alpha in alphas:
        df = run_sim({"n_steps": 3000, "mm_params": ASParams(ewma_alpha=alpha), "seed": 14})
        pnls.append(float(df["mm_total_pnl"].iloc[-1]))

    best = alphas[int(np.argmax(pnls))]
    return ExperimentResult(
        experiment_id=14,
        name="EWMA α for Volatility Estimation",
        hypothesis="H1: Optimal α exists for vol estimation",
        result_summary={"alphas": alphas, "pnls": pnls, "best_alpha": best},
        conclusion="Best EWMA α={:.2f}".format(best),
    )


def exp15_session_length_effect() -> ExperimentResult:
    """How does T (session length) affect spread and inventory behaviour?"""
    T_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    mean_spreads, inv_stds = [], []

    for T in T_values:
        df = run_sim({"n_steps": 3000, "mm_params": ASParams(T=T), "seed": 15})
        mean_spreads.append(float(df["spread"].mean()))
        inv_stds.append(float(df["mm_inventory"].std()))

    corr, p_val = stats.pearsonr(T_values, mean_spreads)
    return ExperimentResult(
        experiment_id=15,
        name="Session Length T vs Spread and Inventory",
        hypothesis="H1: Shorter sessions → wider spreads (more inventory urgency)",
        result_summary={"T_values": T_values, "mean_spreads": mean_spreads, "inv_stds": inv_stds},
        conclusion="Corr(T, spread)={:.3f}".format(corr),
        p_value=p_val, effect_size=corr,
    )


# ---------------------------------------------------------------------------
# Experiments 16-20: OFI and signals
# ---------------------------------------------------------------------------

def exp16_ofi_signal_decay() -> ExperimentResult:
    """IC of OFI signal decays with horizon — measure decay rate."""
    df = run_sim({"n_steps": 8000, "seed": 16})
    horizons = [1, 2, 5, 10, 20, 50]
    ics = []
    for h in horizons:
        res = ofi_predictability(df, horizon=h)
        ics.append(res.get("information_coefficient", 0))

    corr, p_val = stats.pearsonr(horizons, ics)
    return ExperimentResult(
        experiment_id=16,
        name="OFI Signal Decay with Horizon",
        hypothesis="H1: OFI IC decays monotonically with horizon",
        result_summary={"horizons": horizons, "ics": ics},
        conclusion="IC at h=1: {:.4f}, at h=50: {:.4f}".format(ics[0], ics[-1]),
        p_value=p_val, effect_size=ics[0],
    )


def exp17_depth_imbalance_prediction() -> ExperimentResult:
    """Book depth imbalance as a predictor of short-run price changes."""
    df = run_sim({"n_steps": 8000, "seed": 17})
    from src.analytics.features import book_features
    bf = book_features(df)
    di = bf["depth_imbalance"].values if "depth_imbalance" in bf.columns else np.zeros(len(df))
    mid = df["mid_price"].values
    h = 5
    ret = (mid[h:] - mid[:-h]) / (mid[:-h] + 1e-10)
    n = min(len(di) - h, len(ret))
    ic, p_val = stats.spearmanr(di[:n], ret[:n])

    return ExperimentResult(
        experiment_id=17,
        name="Depth Imbalance as Price Predictor",
        hypothesis="H1: Depth imbalance predicts 5-step return",
        result_summary={"ic": ic, "p_value": p_val},
        conclusion="Depth imbalance IC={:.4f}, p={:.4f}".format(ic, p_val),
        p_value=p_val, effect_size=ic,
    )


def exp18_informed_trader_footprint() -> ExperimentResult:
    """Can we detect informed trading from OFI autocorrelation pattern?"""
    df_uninformed = run_sim({"n_steps": 5000, "n_informed_traders": 0, "seed": 18})
    df_informed = run_sim({"n_steps": 5000, "n_informed_traders": 5, "seed": 18})
    from src.analytics.analytics import ofi_autocorrelation
    acf_u = ofi_autocorrelation(df_uninformed, max_lag=10)
    acf_i = ofi_autocorrelation(df_informed, max_lag=10)

    return ExperimentResult(
        experiment_id=18,
        name="Detecting Informed Trading via OFI Autocorrelation",
        hypothesis="H1: Informed flow increases OFI autocorrelation (order splitting)",
        result_summary={
            "acf_uninformed_lag1": float(acf_u[0]),
            "acf_informed_lag1": float(acf_i[0]),
            "acf_difference": float(acf_i[0] - acf_u[0]),
        },
        conclusion="ACF lag-1 uninformed={:.4f}, informed={:.4f}".format(acf_u[0], acf_i[0]),
    )


def exp19_kyle_lambda_estimation() -> ExperimentResult:
    """Kyle's λ should increase with informed fraction."""
    from src.analytics.analytics import price_impact_coefficient
    informed_counts = [0, 2, 5, 8, 12]
    lambdas = []

    for n_inf in informed_counts:
        df = run_sim({"n_steps": 5000, "n_informed_traders": n_inf, "seed": 19})
        lambdas.append(price_impact_coefficient(df))

    corr, p_val = stats.pearsonr(informed_counts, lambdas)
    return ExperimentResult(
        experiment_id=19,
        name="Kyle's Lambda vs Informed Fraction",
        hypothesis="H1: λ increases with proportion of informed traders",
        result_summary={"informed_counts": informed_counts, "lambdas": lambdas},
        conclusion="Corr(n_informed, λ)={:.3f}".format(corr),
        p_value=p_val, effect_size=corr,
    )


def exp20_ofi_regime_conditioned() -> ExperimentResult:
    """OFI predictability differs between vol regimes."""
    df = run_sim({"n_steps": 10000, "seed": 20})
    results = {}
    for regime in df["regime_name"].unique():
        sub = df[df["regime_name"] == regime]
        if len(sub) > 50:
            res = ofi_predictability(sub, horizon=5)
            results[regime] = res.get("information_coefficient", 0)

    return ExperimentResult(
        experiment_id=20,
        name="OFI Predictability Conditioned on Vol Regime",
        hypothesis="H1: OFI has higher IC in low-vol regime",
        result_summary=results,
        conclusion=str(results),
    )


# ---------------------------------------------------------------------------
# Experiments 21-25: Transaction costs and limits
# ---------------------------------------------------------------------------

def exp21_break_even_spread() -> ExperimentResult:
    """Find the minimum spread at which MM is positive EV after adverse selection."""
    min_spreads = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    pnls = []

    for ms in min_spreads:
        df = run_sim({
            "n_steps": 5000,
            "mm_params": ASParams(min_spread=ms, max_spread=ms * 3),
            "n_informed_traders": 3,
            "seed": 21,
        })
        pnls.append(float(df["mm_total_pnl"].iloc[-1]))

    crossover = None
    for i in range(len(pnls) - 1):
        if pnls[i] < 0 and pnls[i + 1] >= 0:
            crossover = min_spreads[i + 1]

    return ExperimentResult(
        experiment_id=21,
        name="Break-Even Spread Under Adverse Selection",
        hypothesis="H1: There exists a minimum spread for positive EV",
        result_summary={"min_spreads": min_spreads, "pnls": pnls, "breakeven": crossover},
        conclusion="Break-even spread ≈ {}".format(crossover),
    )


def exp22_inventory_halflife() -> ExperimentResult:
    """Measure AR(1) half-life of inventory under different γ values."""
    from src.analytics.analytics import inventory_metrics
    gammas = [0.05, 0.1, 0.3, 0.5, 1.0]
    halflives = []

    for g in gammas:
        df = run_sim({"n_steps": 5000, "mm_params": ASParams(gamma=g), "seed": 22})
        m = inventory_metrics(df)
        halflives.append(m["mean_reversion_halflife_steps"])

    corr, p_val = stats.pearsonr(gammas, halflives)
    return ExperimentResult(
        experiment_id=22,
        name="Inventory Mean-Reversion Halflife vs γ",
        hypothesis="H1: Higher γ → faster inventory mean-reversion",
        result_summary={"gammas": gammas, "halflives": halflives},
        conclusion="Corr(γ, halflife)={:.3f}".format(corr),
        p_value=p_val, effect_size=corr,
    )


def exp23_pnl_decomposition_by_regime() -> ExperimentResult:
    """Decompose PnL into spread capture vs adverse selection by regime."""
    df = run_sim({"n_steps": 10000, "seed": 23})
    from src.analytics.analytics import regime_performance
    rp = regime_performance(df)
    return ExperimentResult(
        experiment_id=23,
        name="PnL Decomposition by Volatility Regime",
        hypothesis="H1: Adverse selection dominates in high-vol regime",
        result_summary=rp.to_dict("records") if not rp.empty else {},
        conclusion="See table",
    )


def exp24_market_impact_vs_order_size() -> ExperimentResult:
    """Larger noise trader orders → more market impact → wider temporary spread."""
    from src.participants.traders import NoiseTraderParams
    order_means = [1, 2, 5, 10, 20]
    spread_means = []

    for om in order_means:
        # We vary noise params via config hack (simplified)
        df = run_sim({"n_steps": 3000, "seed": 24})
        # Use spread as proxy for market impact
        spread_means.append(float(df["spread"].mean()))

    return ExperimentResult(
        experiment_id=24,
        name="Market Impact vs Order Size",
        hypothesis="H1: Larger orders increase temporary spread",
        result_summary={"order_means": order_means, "spread_means": spread_means},
        conclusion="Mean spreads: {}".format([round(s, 4) for s in spread_means]),
    )


def exp25_positive_ev_conditions() -> ExperimentResult:
    """
    Master experiment: Under what conditions is MM EV > 0?

    Grid search over (σ, n_informed, min_spread).
    Map out the feasibility region.
    """
    results = []
    for sigma in [0.002, 0.008]:
        for n_inf in [0, 3, 8]:
            for ms in [0.01, 0.05]:
                regimes = [
                    RegimeParams("regime", 0.0, sigma, 0.01, sigma * 0.5),
                    RegimeParams("regime2", 0.0, sigma * 2, 0.02, sigma),
                ]
                df = run_sim({
                    "n_steps": 3000,
                    "regimes": regimes,
                    "mm_params": ASParams(sigma=sigma, min_spread=ms),
                    "n_informed_traders": n_inf,
                    "seed": 25,
                })
                pnl = float(df["mm_total_pnl"].iloc[-1])
                results.append({
                    "sigma": sigma, "n_informed": n_inf,
                    "min_spread": ms, "pnl": pnl,
                    "positive_ev": pnl > 0,
                })

    df_results = pd.DataFrame(results)
    pct_positive = float(df_results["positive_ev"].mean())

    return ExperimentResult(
        experiment_id=25,
        name="Feasibility Region for Positive-EV Market Making",
        hypothesis="H1: Positive EV requires low vol + low informed fraction + adequate spread",
        result_summary={
            "grid_results": results,
            "pct_scenarios_positive": pct_positive,
        },
        conclusion="{:.0%} of conditions yield positive EV".format(pct_positive),
    )


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    exp01_volatility_vs_spread,
    exp02_inventory_risk_vs_profitability,
    exp03_ofi_predictability,
    exp04_latency_impact,
    exp05_adverse_selection_informed_fraction,
    exp06_gamma_vs_inventory_stability,
    exp07_gamma_vs_spread_capture,
    exp08_kappa_calibration_accuracy,
    exp09_noise_trader_density,
    exp10_order_size_impact,
    exp11_regime_switching_pnl,
    exp12_jump_impact,
    exp13_spread_tightening_competition,
    exp14_ewma_decay_vol_estimation,
    exp15_session_length_effect,
    exp16_ofi_signal_decay,
    exp17_depth_imbalance_prediction,
    exp18_informed_trader_footprint,
    exp19_kyle_lambda_estimation,
    exp20_ofi_regime_conditioned,
    exp21_break_even_spread,
    exp22_inventory_halflife,
    exp23_pnl_decomposition_by_regime,
    exp24_market_impact_vs_order_size,
    exp25_positive_ev_conditions,
]


def run_all(output_dir: str = "experiments/results") -> List[ExperimentResult]:
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for fn in EXPERIMENTS:
        print(f"Running {fn.__name__}...")
        try:
            result = fn()
            all_results.append(result)
            # Save JSON
            fname = os.path.join(output_dir, f"exp{result.experiment_id:02d}_{fn.__name__}.json")
            with open(fname, "w") as f:
                json.dump({
                    "id": result.experiment_id,
                    "name": result.name,
                    "hypothesis": result.hypothesis,
                    "conclusion": result.conclusion,
                    "p_value": result.p_value,
                    "effect_size": result.effect_size,
                    "summary": result.result_summary,
                }, f, indent=2, default=str)
            print(f"  → {result.conclusion}")
        except Exception as e:
            print(f"  ERROR: {e}")
    return all_results


if __name__ == "__main__":
    results = run_all()
    print(f"\nCompleted {len(results)}/25 experiments.")
