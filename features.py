"""
Feature Engineering Pipeline
=============================

Constructs predictive features from raw simulation/market data.

Feature categories:
1. Order Flow Features     — OFI, signed volume, trade imbalance
2. Book Shape Features     — depth imbalance, slope, convexity
3. Inventory Features      — MM inventory state, risk metrics
4. Execution Features      — fill rates, queue position, latency effects
5. Regime Features         — vol regime, trend indicators
6. Cross-sectional         — relative to rolling windows

Design philosophy:
- All features are normalised to be roughly unit-scale (z-score or rank).
- Features are computed on rolling windows to avoid lookahead.
- We document the economic intuition for each feature.

Target variables:
- Forward mid-price change (directional)
- Fill probability (execution)
- Realised spread (profitability)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Order flow features
# ---------------------------------------------------------------------------

def ofi_features(df: pd.DataFrame, windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
    """
    Multi-scale Order Flow Imbalance features.

    OFI_w = (bid_vol - ask_vol) / (bid_vol + ask_vol)  over window w

    Economic intuition: OFI captures the net buy/sell pressure in the book.
    Cont, Kukanov & Stoikov (2014) show OFI explains ~65% of short-run
    price changes on top US equities.
    """
    features = pd.DataFrame(index=df.index)

    for w in windows:
        features[f"ofi_{w}"] = df["ofi"].rolling(w).mean()
        features[f"ofi_std_{w}"] = df["ofi"].rolling(w).std()
        features[f"ofi_skew_{w}"] = df["ofi"].rolling(w).skew()
        # OFI momentum: acceleration of order flow
        features[f"ofi_diff_{w}"] = features[f"ofi_{w}"].diff()

    # Signed volume (from spread direction as proxy)
    features["ofi_raw"] = df["ofi"]
    features["ofi_abs"] = df["ofi"].abs()
    features["ofi_sign"] = np.sign(df["ofi"])

    # OFI z-score (rolling)
    roll_mean = df["ofi"].rolling(100).mean()
    roll_std = df["ofi"].rolling(100).std().replace(0, 1)
    features["ofi_zscore"] = (df["ofi"] - roll_mean) / roll_std

    return features


def trade_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trade-based flow features.

    Total trades acceleration, trade clustering.
    """
    features = pd.DataFrame(index=df.index)
    if "total_trades" not in df.columns:
        return features

    trade_vol = df["total_trades"].diff().fillna(0)
    features["trade_rate"] = trade_vol
    features["trade_rate_5"] = trade_vol.rolling(5).mean()
    features["trade_rate_20"] = trade_vol.rolling(20).mean()
    features["trade_acceleration"] = trade_vol.diff()
    # Deviation from recent mean (identifies unusual activity)
    features["trade_rate_zscore"] = (
        (trade_vol - trade_vol.rolling(50).mean()) /
        (trade_vol.rolling(50).std().replace(0, 1))
    )
    return features


# ---------------------------------------------------------------------------
# Book shape features
# ---------------------------------------------------------------------------

def book_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limit order book shape features.

    Depth imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
    Stronger than simple OFI as it captures resting liquidity, not just flow.
    """
    features = pd.DataFrame(index=df.index)

    if "bid_depth_1" not in df.columns or "ask_depth_1" not in df.columns:
        return features

    bd = df["bid_depth_1"].fillna(0)
    ad = df["ask_depth_1"].fillna(0)
    total = bd + ad
    total = total.replace(0, 1)

    features["depth_imbalance"] = (bd - ad) / total
    features["total_depth"] = bd + ad
    features["depth_ratio"] = bd / (ad + 1)
    features["log_depth_ratio"] = np.log((bd + 1) / (ad + 1))

    # Depth imbalance momentum
    for w in [5, 20]:
        features[f"depth_imbalance_{w}"] = features["depth_imbalance"].rolling(w).mean()

    return features


def spread_features(df: pd.DataFrame, windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
    """
    Spread dynamics features.

    Spread expansion is a signal of:
    1. Increased adverse selection risk
    2. Reduced liquidity / increased market uncertainty
    3. Potential regime transition
    """
    features = pd.DataFrame(index=df.index)
    spread = df["spread"]

    for w in windows:
        features[f"spread_ma_{w}"] = spread.rolling(w).mean()
        features[f"spread_std_{w}"] = spread.rolling(w).std()
        features[f"spread_zscore_{w}"] = (
            (spread - spread.rolling(w).mean()) / (spread.rolling(w).std().replace(0, 1))
        )

    features["spread_raw"] = spread
    features["spread_change"] = spread.diff()
    features["spread_pct_change"] = spread.pct_change()
    # Spread regime: above/below median
    features["spread_above_median"] = (spread > spread.rolling(100).median()).astype(int)

    return features


# ---------------------------------------------------------------------------
# Price dynamics features
# ---------------------------------------------------------------------------

def price_features(df: pd.DataFrame, windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
    """
    Price momentum, volatility, and microstructure features.
    """
    features = pd.DataFrame(index=df.index)
    mid = df["mid_price"]
    ret = mid.pct_change()

    for w in windows:
        features[f"ret_{w}"] = ret.rolling(w).sum()
        features[f"vol_{w}"] = ret.rolling(w).std()
        features[f"vol_of_vol_{w}"] = features[f"vol_{w}"].rolling(w).std()

    # Signed return for momentum
    features["ret_1"] = ret
    features["ret_1_sign"] = np.sign(ret)

    # Realised volatility (squared returns)
    features["realised_var_20"] = (ret**2).rolling(20).sum()
    features["realised_var_60"] = (ret**2).rolling(60).sum()

    # Vol ratio: short-to-long term vol (vol regime indicator)
    features["vol_ratio_5_60"] = features["vol_5"] / (features["vol_60"] + 1e-8)

    # Price deviation from rolling mean (reversion signal)
    for w in [20, 60]:
        roll_mean = mid.rolling(w).mean()
        roll_std = mid.rolling(w).std().replace(0, 1)
        features[f"price_zscore_{w}"] = (mid - roll_mean) / roll_std

    return features


# ---------------------------------------------------------------------------
# Inventory features
# ---------------------------------------------------------------------------

def inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MM inventory state features for execution quality analysis.

    High inventory → MM leans quotes → predictable quote skew.
    This is a signal other sophisticated participants can exploit.
    """
    features = pd.DataFrame(index=df.index)
    inv = df["mm_inventory"]

    features["inventory"] = inv
    features["inventory_abs"] = inv.abs()
    features["inventory_sign"] = np.sign(inv)
    features["inventory_sq"] = inv**2

    # Inventory momentum (building or reducing?)
    features["inventory_change"] = inv.diff()
    features["inventory_change_5"] = inv.diff(5)

    # Inventory z-score
    roll_std = inv.rolling(50).std().replace(0, 1)
    features["inventory_zscore"] = inv / roll_std

    # Inventory risk proxy (q² * σ²)
    if "spread" in df.columns:
        vol_proxy = df["spread"] / 4.0
        features["inventory_risk"] = inv**2 * vol_proxy**2
        features["inventory_risk_20"] = features["inventory_risk"].rolling(20).mean()

    return features


# ---------------------------------------------------------------------------
# Execution quality features
# ---------------------------------------------------------------------------

def execution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features related to execution quality and fill probability.
    """
    features = pd.DataFrame(index=df.index)

    if "mm_spread_capture" in df.columns:
        sc = df["mm_spread_capture"].diff().fillna(0)
        features["spread_capture_rate"] = sc
        features["spread_capture_20"] = sc.rolling(20).mean()

    if "mm_adverse_selection" in df.columns:
        asc = df["mm_adverse_selection"].diff().fillna(0)
        features["adverse_selection_rate"] = asc
        features["adverse_selection_20"] = asc.rolling(20).mean()

    if "mm_spread_capture" in df.columns and "mm_adverse_selection" in df.columns:
        sc = df["mm_spread_capture"].diff().fillna(0)
        asc = df["mm_adverse_selection"].diff().fillna(0)
        total = sc + asc
        total = total.replace(0, 1)
        features["alpha_ratio"] = (sc - asc) / total

    return features


# ---------------------------------------------------------------------------
# Master feature pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    include_targets: bool = True,
) -> pd.DataFrame:
    """
    Build full feature matrix from simulation DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output from MarketSimulator.run()
    windows : list of int, optional
        Rolling window sizes for multi-scale features
    include_targets : bool
        Whether to include target columns (for ML training)

    Returns
    -------
    pd.DataFrame with all features (NaN rows from rolling windows dropped)
    """
    if windows is None:
        windows = [5, 20, 60]

    feature_dfs = [
        ofi_features(df, windows),
        trade_flow_features(df),
        book_features(df),
        spread_features(df, windows),
        price_features(df, windows),
        inventory_features(df),
        execution_features(df),
    ]

    result = pd.concat(feature_dfs, axis=1)

    if include_targets:
        # Short-horizon price direction (main target)
        for h in [1, 5, 10]:
            result[f"target_ret_{h}"] = df["mid_price"].pct_change(h).shift(-h)
        result["target_ret_sign_5"] = np.sign(result["target_ret_5"])
        # Fill probability target (binary: was there a MM fill this step?)
        if "mm_spread_capture" in df.columns:
            result["target_fill"] = (df["mm_spread_capture"].diff() > 0).astype(int)

    # Add regime labels
    if "regime_name" in df.columns:
        result["regime"] = df["regime_name"]
        result["regime_low_vol"] = (df["regime_name"] == "low_vol").astype(int)

    # Drop warmup rows (rolling window NaNs)
    max_window = max(windows)
    result = result.iloc[max_window:].copy()
    result = result.dropna(how="all", axis=1)

    return result


def feature_importance_ols(
    features: pd.DataFrame, target_col: str = "target_ret_5"
) -> pd.Series:
    """
    Quick OLS-based feature importance (t-statistics).
    Not for production use — just for exploratory analysis.
    """
    if target_col not in features.columns:
        return pd.Series(dtype=float)

    target = features[target_col].dropna()
    feat_cols = [c for c in features.columns if not c.startswith("target") and c != "regime"]

    X = features[feat_cols].loc[target.index].fillna(0)
    y = target

    # Standardise
    X_std = (X - X.mean()) / (X.std().replace(0, 1))

    t_stats = {}
    for col in feat_cols:
        x = X_std[col].values
        mask = np.isfinite(x) & np.isfinite(y.values)
        if mask.sum() < 20:
            t_stats[col] = 0.0
            continue
        slope, _, _, _, se = stats.linregress(x[mask], y.values[mask])
        t_stats[col] = slope / (se + 1e-10)

    return pd.Series(t_stats).sort_values(key=abs, ascending=False)


def correlation_matrix(features: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Feature correlation matrix for top_n features by variance."""
    feat_cols = [c for c in features.columns if not c.startswith("target") and c != "regime"]
    var = features[feat_cols].var().sort_values(ascending=False)
    top = var.head(top_n).index.tolist()
    return features[top].corr()
