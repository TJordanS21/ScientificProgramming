from typing import Dict

import pandas as pd

from data_loader import DataLoader, prepare_data
from analysis import (make_reliability_table, summarize_returns_for_ci, plot_reliability_table,
                      build_ci_table, plot_mean_se_and_ci, ticker, TICKERS, plot_ci_widths, log,
                      rolling_ci_analysis, plot_rolling_ci_all)
from scenarios import generate_scenario, plot_scenarios_overview_one_image

# ==============================================================
# MAIN: runs steps (a)–(e) + extensions
# ==============================================================

if __name__ == "__main__":
    # --- (a) Load & prepare: exactly 10 trading days for SE/CI
    raw = DataLoader(lookback_days=180).fetch()
    stock10 = prepare_data(raw, n_trading_days=10)

    # --- (b) Reliability table (critical t-values) + plot
    confs_table = [0.80, 0.90, 0.95, 0.98, 0.99, 0.998, 0.999]
    reliability_tbl = make_reliability_table(confs_table)
    plot_reliability_table(reliability_tbl, title="Critical t-values (df=9)")

    # --- (c)(d) SE & t-intervals for (10 days) + visuals
    stats10 = summarize_returns_for_ci(stock10["Return"])
    ci_tbl = build_ci_table(stats10, [0.90, 0.95, 0.99])
    plot_mean_se_and_ci(stats10, ci_tbl, title_prefix=f"{ticker} (10 days)")

    # --- (e) CI widths (bar plot)
    plot_ci_widths(ci_tbl, title=f"{ticker} (10 days) – Confidence Interval Widths")

    # ==============================================================
    # EXTENSION: rolling 10-day CIs over ~6 months
    # ==============================================================
    log.info("Rolling 10-day")
    # Full series without trimming → better time-series analysis
    stock_full = prepare_data(raw, n_trading_days=None)

    confs = [0.90, 0.95, 0.99]
    # One rolling DataFrame per confidence level
    roll_dfs = {c: rolling_ci_analysis(stock_full, window=10, conf=c) for c in confs}

    # All confidence intervals overlaid in one plot
    plot_rolling_ci_all(roll_dfs, ystep=0.05, ylims=(-0.15, 0.20))

    # ==============================================================
    # EXTENSION: Scenarios + real stocks
    # ==============================================================
    log.info("Scenarios")
    # Real stock data for ~60 trading days (from already-loaded 'raw')
    stocks_60 = prepare_data(raw, n_trading_days=60)
    base_price = float(stocks_60["Price"].iloc[0])

    # Synthetic market regimes (60 days each) + real stock
    scenarios: Dict[str, pd.DataFrame] = {
        f"Real {ticker}": stocks_60,
        "Bull": generate_scenario("bull", n_days=60, seed=42, base_price=base_price),
        "Bear": generate_scenario("bear", n_days=60, seed=42, base_price=base_price),
        "Volatile": generate_scenario("volatile", n_days=60, seed=42, base_price=base_price),
        "Crisis": generate_scenario("crisis", n_days=60, seed=42, base_price=base_price),
        "Neutral": generate_scenario("neutral", n_days=60, seed=42, base_price=base_price)
    }

    # Combined figure with grouped bars (90/95/99)
    plot_scenarios_overview_one_image(
        scenarios,
        conf=[0.90, 0.95, 0.99],
        sample_n=10,
        title=f"{ticker} Real Data vs. Synthetic Market Scenarios",
        rebase_to=base_price
    )
