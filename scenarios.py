# ==============================================================
# BONUS: SZENARIEN (synthetische Preisreihen, CI-Vergleiche)
# ==============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Optional

from aufgabe6 import summarize_returns_for_ci, build_ci_table


def generate_scenario(kind: str, n_days: int = 60, seed: int = 42, base_price: float = 25.0) -> pd.DataFrame:
    """
    Erzeugt synthetische Kursverläufe mit normalverteilten Renditen und
    einfachem Drift+Volatilität-Model

    'kind' bestimmt die Parameter (μ, σ):
    - "bull": positiver Drift, moderate Volatilität
    - "bear": negativer Drift, etwas höhere Volatilität
    - "volatile": ~0-Drift, hohe Volatilität
    - "crisis": starker negativer Drift, sehr hohe Volatilität
    - sonst: "neutral": leichter positiver Drift

    Rückgabe:
    - DataFrame mit Spalten "Price" (aus Renditen aufgespult) und "Return"
    """
    np.random.seed(seed)

    # Parameterauswahl je Szenario (nur illustrative Grössenordnungen)
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

    # Aus Renditen Preise aufbauen (Startwert = base_price)
    prices = [base_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    return pd.DataFrame({"Price": prices[1:], "Return": returns})


def scenario_ci_table(df: pd.DataFrame, conf_levels: List[float], sample_n: int = 10) -> pd.DataFrame:
    """
    Nimmt die ersten 'sample_n' Renditen aus einem Szenario, berechnet
    Mean/SE/df und gibt eine CI-Tabelle für 'conf_levels' zurück.
    (Vergleichbarkeit zwischen Szenarien)
    """
    sample = df["Return"].dropna().iloc[:sample_n]
    m = summarize_returns_for_ci(sample)
    return build_ci_table(m, conf_levels)


def compute_scenario_ci_widths(scenarios: Dict[str, pd.DataFrame],
                               conf: float | List[float],
                               sample_n: int = 10) -> Tuple[List[str], Dict[float, List[float]]]:
    """
    Hilfsfunktion: extrahiert CI-Breiten je Szenario (für ein oder mehrere
    Konfidenzniveaus) zur späteren Balkenplot-Darstellung.
    """
    conf_list = conf if isinstance(conf, list) else [conf]
    names = list(scenarios.keys())
    widths_by_conf: Dict[float, List[float]] = {c: [] for c in conf_list}
    for name in names:
        df = scenarios[name]
        tbl = scenario_ci_table(df, conf_list, sample_n=sample_n)
        for c in conf_list:
            row = tbl.loc[tbl["Konfidenzniveau"] == f"{int(c*100)}%"].iloc[0]
            widths_by_conf[c].append(row["Breite"])
    return names, widths_by_conf


def plot_scenarios_overview_one_image(scenarios: Dict[str, pd.DataFrame],
                                      conf: float | List[float] = 0.95,
                                      sample_n: int = 10,
                                      title: str = "Scenarios Overview",
                                      rebase_to: float = 25.0):
    """
    Erzeugt eine kombinierte Abbildung mit:
    (oben) überlagerten, rebasierten Preisverläufen aller Szenarien,
    (unten) Balkendiagramm der CI-Breiten (für 10-Tages-Mittelrendite).
    """
    # CI-Breiten vorbereiten
    names, widths_by_conf = compute_scenario_ci_widths(scenarios, conf, sample_n=sample_n)
    conf_list = conf if isinstance(conf, list) else [conf]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2.0, 1.2]})

    # -------- oben: alle Preisverläufe (gemeinsamer Startwert durch Rebase) --------
    ax0 = axes[0]
    for name, df in scenarios.items():
        p = df["Price"].to_numpy()
        if rebase_to is not None and len(p) > 0 and p[0] != 0:
            p = p / p[0] * rebase_to  # Startwerte angleichen
        x = np.arange(len(p))       # einfacher x-Vektor (Tag 0..n-1)
        ax0.plot(x, p, label=name)
    ax0.set_title("Preisverläufe – alle Szenarien (synthetisch & real, rebasiert)")
    ax0.set_xlabel("Tag")
    ax0.set_ylabel("Preis (rebasiert)")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(ncol=3, fontsize=9, frameon=False)

    # -------- unten: CI-Breitenvergleich --------
    ax1 = axes[1]
    x = np.arange(len(names))
    bar_w = 0.8 / max(1, len(conf_list))
    if len(conf_list) == 1:
        # Ein einziges Konfidenzniveau → einfache Balken
        c = conf_list[0]
        bars = ax1.bar(x, widths_by_conf[c], width=0.6)
        # Balken mit Werten annotieren
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=9)
        ax1.set_title(f"CI-Breite der 10-Tages-Mittelrendite bei {int(c*100)}%")
    else:
        # Mehrere Konfidenzniveaus → gruppierte Balken
        for i, c in enumerate(conf_list):
            offs = x - 0.4 + (i + 0.5) * bar_w
            ax1.bar(offs, widths_by_conf[c], width=bar_w, label=f"{int(c*100)}%")
            for (bx, h) in zip(offs, widths_by_conf[c]):
                ax1.text(bx, h, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
        ax1.legend(title="Konfidenz", frameon=False)
        ax1.set_title("CI-Breite der 10-Tages-Mittelrendite nach Szenario")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Breite des Intervalls")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    conf_txt = ", ".join([f"{int(c*100)}%" for c in conf_list])
    fig.suptitle(f"{title} – Konfidenz: {conf_txt}", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


