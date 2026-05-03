from typing import Dict

import pandas as pd

from DataLoader import DataLoader, prepare_data
from analysis import (make_reliability_table, summarize_returns_for_ci, plot_reliability_table,
                      build_ci_table, plot_mean_se_and_ci, ticker, plot_ci_widths, log,
                      rolling_ci_analysis, plot_rolling_ci_all)
from scenarios import generate_scenario, plot_scenarios_overview_one_image

# ==============================================================
# MAIN: führt Schritte (a)–(e) + Erweiterungen aus
# ==============================================================

if __name__ == "__main__":
    # --- (a) Laden & vorbereiten: exakt 10 Handelstage für SE/CI
    raw = DataLoader(lookback_days=180).fetch()
    stock10 = prepare_data(raw, n_trading_days=10)

    # --- (b) Zuverlässigkeitstabelle (kritische z-Werte) + Plot
    confs_table = [0.80, 0.90, 0.95, 0.98, 0.99, 0.998, 0.999]
    reliability_tbl = make_reliability_table(confs_table)
    plot_reliability_table(reliability_tbl, title="Tabelle 6 (hohe Freiheitsgrade) – z_{α/2}")

    # --- (c)(d) SE & t-Intervalle für (10 Tage) + Visuals
    stats10 = summarize_returns_for_ci(stock10["Return"])
    ci_tbl = build_ci_table(stats10, [0.90, 0.95, 0.99])
    plot_mean_se_and_ci(stats10, ci_tbl, title_prefix=f"{ticker} (10 Tage)")

    # --- (e) CI-Breiten (Balkenplot)
    plot_ci_widths(ci_tbl, title=f"{ticker} (10 Tage) – Breite der Konfidenzintervalle")

    # ==============================================================
    # ERWEITERUNG: rollierende 10-Tages-CIs über ~6 Monate
    # ==============================================================
    log.info("Rolling 10-day")
    # Ganze Serie ohne Abschneiden → bessere Zeitreihen-Analyse
    stock_full = prepare_data(raw, n_trading_days=None)

    confs = [0.90, 0.95, 0.99]
    # Für jedes Konfidenzniveau ein eigener Rolling-DataFrame
    roll_dfs = {c: rolling_ci_analysis(stock_full, window=10, conf=c) for c in confs}

    # Alle Konfidenzintervalle in EINEM Plot überlagert
    plot_rolling_ci_all(roll_dfs, ystep=0.05, ylims=(-0.15, 0.20))

    # ==============================================================
    # ERWEITERUNG: Szenarien + reale Stocks
    # ==============================================================
    log.info("Bonus")
    # Reale Stocks-Daten für ~60 Handelstage (aus dem bereits geladenen 'raw')
    stocks_60 = prepare_data(raw, n_trading_days=60)
    base_price = float(stocks_60["Price"].iloc[0])

    # Synthetische Marktregime (jeweils 60 Tage) + reale Stocks
    scenarios: Dict[str, pd.DataFrame] = {
        f"Real {ticker}": stocks_60,
        "Bull": generate_scenario("bull", n_days=60, seed=42, base_price=base_price),
        "Bear": generate_scenario("bear", n_days=60, seed=42, base_price=base_price),
        "Volatil": generate_scenario("volatile", n_days=60, seed=42, base_price=base_price),
        "Krise": generate_scenario("crisis", n_days=60, seed=42, base_price=base_price),
        "Neutral": generate_scenario("neutral", n_days=60, seed=42, base_price=base_price)
    }

    # Eine kombinierte Abbildung mit gruppierten Balken (90/95/99)
    plot_scenarios_overview_one_image(
        scenarios,
        conf=[0.90, 0.95, 0.99],
        sample_n=10,
        title=f"{ticker} Real Data vs. Synthetic Market Scenarios",
        rebase_to=base_price
    )
