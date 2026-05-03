# ==============================================================
# Synthetic Market Scenarios & CI Comparisons
# ==============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Optional

from analysis import summarize_returns_for_ci, build_ci_table


def generate_scenario(kind: str, n_days: int = 60, seed: int = 42, base_price: float = 25.0) -> pd.DataFrame:
    """
    Generates synthetic price series with normally distributed returns
    using a simple drift + volatility model.

    'kind' determines the parameters (μ, σ):
    - "bull": positive drift, moderate volatility
    - "bear": negative drift, slightly higher volatility
    - "volatile": ~0 drift, high volatility
    - "crisis": strong negative drift, very high volatility
    - else: "neutral": slight positive drift

    Returns:
        DataFrame with columns "Price" (built from returns) and "Return"
    """
    np.random.seed(seed)

    # Parameter selection per scenario (illustrative magnitudes only)
    if kind == "bull":
        mu, sigma = 0.004, 0.010
    elif kind == "bear":
        mu, sigma = -0.003, 0.012
    elif kind == "volatile":
        mu, sigma = 0.000, 0.030
    elif kind == "crisis":
        mu, sigma = -0.006, 0.040
    else:  # "neutral"
        mu, sigma = 0.001, 0.015

    returns = np.random.normal(mu, sigma, n_days)

    # Build prices from returns (starting value = base_price)
    prices = [base_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    return pd.DataFrame({"Price": prices[1:], "Return": returns})


def scenario_ci_table(df: pd.DataFrame, conf_levels: List[float], sample_n: int = 10) -> pd.DataFrame:
    """
    Takes the first 'sample_n' returns from a scenario, computes
    Mean/SE/df and returns a CI table for 'conf_levels'.
    (Ensures comparability across scenarios)
    """
    sample = df["Return"].dropna().iloc[:sample_n]
    m = summarize_returns_for_ci(sample)
    return build_ci_table(m, conf_levels)


def compute_scenario_ci_widths(scenarios: Dict[str, pd.DataFrame],
                               conf: float | List[float],
                               sample_n: int = 10) -> Tuple[List[str], Dict[float, List[float]]]:
    """
    Helper: extracts CI widths per scenario (for one or more confidence
    levels) for later bar chart display.
    """
    conf_list = conf if isinstance(conf, list) else [conf]
    names = list(scenarios.keys())
    widths_by_conf: Dict[float, List[float]] = {c: [] for c in conf_list}
    for name in names:
        df = scenarios[name]
        tbl = scenario_ci_table(df, conf_list, sample_n=sample_n)
        for c in conf_list:
            row = tbl.loc[tbl["Confidence Level"] == f"{int(c*100)}%"].iloc[0]
            widths_by_conf[c].append(row["Width"])
    return names, widths_by_conf


def plot_scenarios_overview_one_image(scenarios: Dict[str, pd.DataFrame],
                                      conf: float | List[float] = 0.95,
                                      sample_n: int = 10,
                                      title: str = "Scenarios Overview",
                                      rebase_to: float = 25.0):
    """
    Creates a combined figure with:
    (top) overlaid, rebased price paths for all scenarios,
    (bottom) bar chart of CI widths (for 10-day mean return).
    """
    # Prepare CI widths
    names, widths_by_conf = compute_scenario_ci_widths(scenarios, conf, sample_n=sample_n)
    conf_list = conf if isinstance(conf, list) else [conf]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2.0, 1.2]})

    # -------- Top: all price paths (common start via rebase) --------
    ax0 = axes[0]
    for name, df in scenarios.items():
        p = df["Price"].to_numpy()
        if rebase_to is not None and len(p) > 0 and p[0] != 0:
            p = p / p[0] * rebase_to  # Align starting values
        x = np.arange(len(p))       # simple x vector (day 0..n-1)
        ax0.plot(x, p, label=name)
    ax0.set_title("Price Paths – All Scenarios (synthetic & real, rebased)")
    ax0.set_xlabel("Day")
    ax0.set_ylabel("Price (rebased)")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(ncol=3, fontsize=9, frameon=False)

    # -------- Bottom: CI width comparison --------
    ax1 = axes[1]
    x = np.arange(len(names))
    bar_w = 0.8 / max(1, len(conf_list))
    if len(conf_list) == 1:
        # Single confidence level → simple bars
        c = conf_list[0]
        bars = ax1.bar(x, widths_by_conf[c], width=0.6)
        # Annotate bars with values
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=9)
        ax1.set_title(f"CI Width of 10-Day Mean Return at {int(c*100)}%")
    else:
        # Multiple confidence levels → grouped bars
        for i, c in enumerate(conf_list):
            offs = x - 0.4 + (i + 0.5) * bar_w
            ax1.bar(offs, widths_by_conf[c], width=bar_w, label=f"{int(c*100)}%")
            for (bx, h) in zip(offs, widths_by_conf[c]):
                ax1.text(bx, h, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
        ax1.legend(title="Confidence", frameon=False)
        ax1.set_title("CI Width of 10-Day Mean Return by Scenario")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Interval Width")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    conf_txt = ", ".join([f"{int(c*100)}%" for c in conf_list])
    fig.suptitle(f"{title} – Confidence: {conf_txt}", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
