# ==============================================================
# Swiss Stock Analysis – Confidence Intervals, Visualisations & Scenarios
# ==============================================================
# Overview:
# This module computes t-confidence intervals for mean daily returns,
# displays reliability tables/plots, compares CI widths, performs rolling
# 10-day window analysis, and supports synthetic scenario generation.
#
# Key statistical points:
# - Returns: simple daily percentage changes (pct_change)
# - For n=10 (small sample) we use the t-distribution (df = n-1)
# - Standard error SE(ȳ) = s / √n, with s as sample std dev (ddof=1)
# - (1-α) confidence interval: ȳ ± t_{α/2, df} * SE(ȳ)
#
# Dependencies: numpy, pandas, matplotlib, seaborn, scipy, yfinance
# - numpy: Mathematical operations and arrays
# - pandas: Data analysis and DataFrames
# - matplotlib: Plotting and charts
# - seaborn: Enhanced statistical visualisations
# - scipy: Scientific and statistical functions
# - yfinance: Downloads financial data (e.g. stock prices) from Yahoo Finance
# Notes:
# - yfinance returns a DataFrame; we set auto_adjust=True to get
#   split/dividend-adjusted close prices ("Close").
# - Plots are illustrative; axes/scales are for returns in decimal form.
# ==============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy import stats

# ----------------------- Stock tickers ----------------------------------
TICKERS: list[str] = ["UBS", "NESN.SW", "NOVN.SW"]  # UBS, Nestlé, Novartis
ticker: str = TICKERS[0]  # default / primary ticker

# ----------------------- Logging --------------------------------
# Basic logging setup: time | level | message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
# Named logger for easy filtering
log = logging.getLogger(ticker)
# ----------------------- Config ---------------------------------
# Higher matplotlib DPI for sharper images
plt.rcParams.update({"figure.dpi": 120})
# Seaborn "talk" context: larger fonts, good for presentations
sns.set_context("talk")

# ==============================================================
# (b) RELIABILITY / CRITICAL z-VALUES (large df ≈ Normal distribution)
# ==============================================================

def make_reliability_table(conf_levels: List[float]) -> pd.DataFrame:
    """
    Creates a table of critical t-values for given confidence levels.

    Background:
    - For large degrees of freedom, t approaches the normal distribution.
    - z_{α/2} = Φ^{-1}(1 − α/2), where Φ is the standard normal CDF.

    Columns:
    - Confidence Level: "95%"
    - 1 - α: 0.95
    - t_{α/2}: 1.960
    """
    rows = []
    for c in conf_levels:
        alpha = 1 - c
        z = stats.t.ppf(1 - alpha/2, df=9)
        rows.append({
            "Confidence Level": f"{int(c*100)}%",
            "1 - α": c,
            "t_{α/2}": round(z, 3)
        })
    return pd.DataFrame(rows)


def plot_reliability_table(tbl: pd.DataFrame, title="Critical t-values (df=9)"):
    """
    Displays the reliability table as a static matplotlib table.
    """
    fig, ax = plt.subplots(figsize=(7.5, 1 * len(tbl)))
    ax.axis("off")
    the_table = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        loc="center",
        cellLoc="center"
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1.2)
    ax.set_title(title, pad=10)
    plt.tight_layout()
    plt.show()

# ==============================================================
# (c)(d) STANDARD ERROR & t-CONFIDENCE INTERVALS (n=10)
# ==============================================================

@dataclass
class MeanCI:
    """
    Data object for CI calculations:
    - mean: sample mean ȳ
    - se: standard error SE(ȳ) = s/√n
    - df: degrees of freedom = n - 1
    """
    mean: float
    se: float
    df: int


def summarize_returns_for_ci(rets: pd.Series) -> MeanCI:
    """
    Aggregates a return series into the statistics needed for a t-CI.

    Steps:
    - Determine n
    - Compute mean and sample std dev (ddof=1)
      ddof=1: sample standard deviation — divides by (n − 1)
      ddof=0: population standard deviation — divides by n
    - Standard error: se = sd / sqrt(n)
    """
    data = pd.Series(rets)
    n = len(data)
    mean = data.mean()

    sd = data.std(ddof=1)  # ddof=1 → sample std dev
    se = sd / np.sqrt(n)
    return MeanCI(mean=mean, se=se, df=n-1)


def t_confidence_interval(m: MeanCI, confidence: float) -> Tuple[float, float]:
    """
    Computes the two-sided t-confidence interval:
    [ȳ − t_{α/2, df} * SE, ȳ + t_{α/2, df} * SE]
    """
    tcrit = stats.t.ppf(1 - (1-confidence)/2, df=m.df)  # critical t-value
    halfwidth = tcrit * m.se                             # half-width of interval
    return (m.mean - halfwidth, m.mean + halfwidth)


def build_ci_table(m: MeanCI, conf_levels: List[float]) -> pd.DataFrame:
    """
    Builds a table with CI bounds, width, etc. for multiple confidence levels.
    Columns:
    - Confidence Level, df, Mean (ȳ), SE(ȳ), CI_lower, CI_upper, Width
    """
    rows = []
    for c in conf_levels:
        lo, hi = t_confidence_interval(m, c)
        rows.append({
            "Confidence Level": f"{int(c*100)}%",
            "df": m.df,
            "Mean (ȳ)": m.mean,
            "SE(ȳ)": m.se,
            "CI_lower": lo,
            "CI_upper": hi,
            "Width": hi - lo
        })
    return pd.DataFrame(rows)


# Visualisations for (c)(d)
def plot_mean_se_and_ci(m: MeanCI, ci_tbl: pd.DataFrame, title_prefix=f"{ticker} (10 trading days)"):
    """
    Two figures:
    1) Point = mean, errorbar = ± SE
    2) Errorbars for different confidence levels (centres + half-widths)
    """
    log.info(m)
    # (1) Mean & SE (errorbar)
    plt.figure(figsize=(8, 4))
    plt.errorbar(["ȳ"], [m.mean], yerr=[m.se], fmt="o", capsize=5, color="tab:blue")
    plt.title(f"{title_prefix} – Mean & Standard Error")
    plt.ylabel("Daily Return")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # (2) t-confidence intervals as errorbars
    plt.figure(figsize=(7, 6))
    xs = list(ci_tbl["Confidence Level"])
    # Centre and half-width per interval for display
    centers = [(lo + hi) / 2 for lo, hi in zip(ci_tbl["CI_lower"], ci_tbl["CI_upper"])]
    half = [(hi - lo) / 2 for lo, hi in zip(ci_tbl["CI_lower"], ci_tbl["CI_upper"])]
    plt.errorbar(xs, centers, yerr=half, fmt="o", capsize=6)
    plt.axhline(0, color="black", linewidth=1)  # Reference line at 0 return
    plt.title(f"{title_prefix} – t-Confidence Intervals (df={m.df})")
    plt.ylabel("Daily Return (Mean ± Half-width)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==============================================================
# (e) CI WIDTH COMPARISON (chart)
# ==============================================================

def plot_ci_widths(ci_table: pd.DataFrame, title: str = "Confidence Interval Widths (n=10)"):
    """
    Bar chart of interval widths for different confidence levels.
    Interpretation: Higher confidence level → wider interval.
    """
    plt.figure(figsize=(7, 4))
    x = ci_table["Confidence Level"]
    y = ci_table["Width"]
    plt.bar(x, y)
    plt.ylabel("Interval Width")
    plt.xlabel("Confidence Level")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==============================================================
# EXTENSION: Rolling 10-day intervals & chunking
# ==============================================================

def rolling_ci_analysis(df: pd.DataFrame, window: int = 10, conf: float = 0.95) -> pd.DataFrame:
    """
    Rolling analysis: For each 10-day window (or 'window'), the mean
    return and t-confidence interval are computed.

    Return columns:
    - EndDate: end date of the window
    - Mean: 10-day mean return
    - Lower/Upper: CI bounds
    - Width: CI width
    """
    rets = df["Return"].dropna()
    results = []
    # Slide window over the return series
    for i in range(window, len(rets) + 1):
        sample = rets.iloc[i - window:i]
        m = summarize_returns_for_ci(sample)
        lo, hi = t_confidence_interval(m, conf)
        results.append({
            "EndDate": rets.index[i - 1],
            "Mean": m.mean,
            "Lower": lo,
            "Upper": hi,
            "Width": hi - lo
        })

    log.info(results)

    return pd.DataFrame(results)


def plot_rolling_ci(df_roll: pd.DataFrame, conf: float,
                    ylims: Tuple[float, float] | None = None,
                    ystep: float = 0.01):
    """
    Visualises a rolling 10-day mean return with CI band.
    - ylims/ystep allow uniform scales (comparable across plots)
    """
    plt.figure(figsize=(10,5))
    plt.plot(df_roll["EndDate"], df_roll["Mean"], label="10d Mean Return")
    plt.fill_between(df_roll["EndDate"], df_roll["Lower"], df_roll["Upper"],
                     alpha=0.3, label=f"{int(conf*100)}% CI")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"{ticker} Rolling 10-Day Mean Return ± {int(conf*100)}% CI")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Enforce uniform scale (optional)
    if ylims is not None:
        plt.ylim(*ylims)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_rolling_ci_all(df_dict: Dict[float, pd.DataFrame],
                        ylims: Tuple[float, float] | None = None,
                        ystep: float = 0.01):
    """
    Draws 90%, 95% and 99% rolling CIs together.
    - Wider intervals filled first (z-order), narrower on top.
    - One shared mean line suffices (identical across confidence levels).
    """
    plt.figure(figsize=(11, 6))

    # Draw order: from wide (99%) to narrow (90%)
    conf_order = sorted(df_dict.keys(), reverse=True)
    # Colour legend (freely chosen, just for distinction)
    colors = {0.90: "#4CAF50", 0.95: "#FFC107", 0.99: "#F44336"}  # green, yellow, red

    # Fill confidence bands
    for conf in conf_order:
        df = df_dict[conf]
        plt.fill_between(df["EndDate"], df["Lower"], df["Upper"],
                         color=colors.get(conf, "gray"), alpha=0.3,
                         label=f"{int(conf*100)}% CI")

    # One mean line (same for all confidence levels)
    mean_df = next(iter(df_dict.values()))
    plt.plot(mean_df["EndDate"], mean_df["Mean"], color="black",
             linewidth=1.5, label="10-Day Mean Return")

    plt.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    plt.title(f"{ticker} Rolling 10-Day Mean Returns with 90%, 95%, and 99% CIs")
    plt.xlabel("Date")
    plt.ylabel("Mean Daily Return")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Confidence Intervals", loc="upper left", frameon=False)

    if ylims is not None:
        plt.ylim(*ylims)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def chunked_ci_analysis(df: pd.DataFrame, window: int = 10, conf: float = 0.95) -> pd.DataFrame:
    """
    Chunked (non-overlapping) CI analysis:
    - The return series is split into disjoint blocks of length 'window'.
    - A CI is computed for each full block.

    Return columns:
    - Period (running number), Start, End, Mean, Lower, Upper, Width
    """
    rets = df["Return"].dropna().reset_index()
    chunks = [rets.iloc[i:i+window] for i in range(0, len(rets), window)]
    results = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) < window:
            continue  # discard incomplete trailing block
        m = summarize_returns_for_ci(chunk["Return"])
        lo, hi = t_confidence_interval(m, conf)
        results.append({
            "Period": idx + 1,
            "Start": chunk[chunk.columns[0]].iloc[0],
            "End": chunk[chunk.columns[0]].iloc[-1],
            "Mean": m.mean,
            "Lower": lo,
            "Upper": hi,
            "Width": hi - lo
        })
    return pd.DataFrame(results)

