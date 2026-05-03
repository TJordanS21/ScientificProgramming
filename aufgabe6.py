# ==============================================================
# Stock – Konfidenzintervalle (10 Tage) + Visualisierungen + Szenarien
# ==============================================================
# Übersicht:
# Dieses Skript lädt Kursdaten (Yahoo Finance), bereitet sie auf,
# berechnet t-Konfidenzintervalle (für Mittelwerte täglicher Renditen),
# zeigt Tabellen/Plots, vergleicht CI-Breiten, analysiert rollierende
# 10-Tages-Fenster und erzeugt synthetische Szenarien (bullish, bearish etc.).
#
# Wichtige statistische Punkte:
# - Renditen: einfache tägliche prozentuale Änderungen (pct_change)
# - Für n=10 (kleine Stichprobe) verwenden wir die t-Verteilung (df = n-1)
# - Standardfehler SE(ȳ) = s / √n, mit s als Stichproben-StdAbw (ddof=1)
# - (1-α)-Konfidenzintervall: ȳ ± t_{α/2, df} * SE(ȳ)
#
# Abhängigkeiten: numpy, pandas, matplotlib, seaborn, scipy, yfinance
# - numpy: Mathematische Operationen und Arrays.
# - pandas: Datenanalyse und Tabellen (DataFrames).
# - matplotlib: Grafische Darstellung (Plots).
# - seaborn: Erweiterte, schönere Visualisierungen.
# - scipy: Wissenschaftliche und statistische Funktionen.
# - yfinance: Lädt Finanzdaten (z.B. Aktienkurse) von Yahoo Finance.
# Hinweise:
# - yfinance liefert einen DataFrame; wir auto_adjust=True setzen, um
#   Splits/Dividenden-adj. Schlusskurse ("Close") zu bekommen.
# - Plots sind rein illustrativ; Achsen/Skalen sind für Renditen in Dezimalform.
# ==============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy import stats

# ----------------------- Gewählter Stock --------------------------------
ticker: str = "UBS"

# ----------------------- Logging --------------------------------
# Basis-Logging einrichten: Zeit | Level | Nachricht
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
# Namensspezifischer Logger, damit Ausgaben leicht filterbar sind
log = logging.getLogger(ticker)
# ----------------------- Config ---------------------------------
# Matplotlib DPI etwas höher für schärfere Bilder
plt.rcParams.update({"figure.dpi": 120})
# Seaborn-Kontext "talk": grössere Schrift, gut für Präsentationen
sns.set_context("talk")

# ==============================================================
# (b) ZUVERLÄSSIGKEIT / KRITISCHE z-WERTE (grosse df ≈ Normalverteilung)
# ==============================================================

def make_reliability_table(conf_levels: List[float]) -> pd.DataFrame:
    """
    Erzeugt eine Tabelle mit z-Werten für gegebene Konfidenzniveaus.

    Hintergrund:
    - Für grosse Freiheitsgrade nähert sich t der Normalverteilung an.
    - z_{α/2} = Φ^{-1}(1 − α/2), mit Φ als Standardnormal-CDF.

    Beispielspalten:
    - Konfidenzniveau: "95%"
    - 1 - α: 0.95
    - z_{α/2}: 1.960
    """
    rows = []
    for c in conf_levels:
        alpha = 1 - c

        # z = stats.t.ppf(1 - alpha/2, df=100)
        z = stats.t.ppf(1 - alpha/2, df=9)

        rows.append({
            "Konfidenzniveau": f"{int(c*100)}%",
            "1 - α": c,
            "z_{α/2}": round(z, 3)
        })
    return pd.DataFrame(rows)


def plot_reliability_table(tbl: pd.DataFrame, title="Tabelle 6 – hohe Freiheitsgrade (≈ z)"):
    """
    Zeigt die obige Tabelle als statische Matplotlib-Tabelle an.
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
# (c)(d) STANDARDFEHLER & t-KONF.-INTERVALLE (n=10)
# ==============================================================

@dataclass
class MeanCI:
    """
    Datenobjekt für Berechnungen:
    - mean: Stichprobenmittel ȳ
    - se: Standardfehler SE(ȳ) = s/√n
    - df: Freiheitsgrade = n - 1
    """
    mean: float
    se: float
    df: int


def summarize_returns_for_ci(rets: pd.Series) -> MeanCI:
    """
    Aggregiert eine Renditeserie zu den für t-CI benötigten Kennzahlen.

    Schritte:
    - n bestimmen
    - Mittelwert und Stichproben-StdAbw (ddof=1)
      ddof=1: Berechnet die Stichprobenstandardabweichung – das heisst, es wird durch (n − 1) geteilt.
      ddof=0: Berechnet die Standardabweichung der Grundgesamtheit – es wird durch n geteilt.
    - Standardfehler: se = sd / sqrt(n)
    """
    data = pd.Series(rets)
    n = len(data)
    mean = data.mean()

    sd = data.std(ddof=1)  # ddof=1 → Stichproben-StdAbw
    se = sd / np.sqrt(n)
    return MeanCI(mean=mean, se=se, df=n-1)


def t_confidence_interval(m: MeanCI, confidence: float) -> Tuple[float, float]:
    """
    Berechnet das zweiseitige t-Konfidenzintervall:
    [ȳ − t_{α/2, df} * SE, ȳ + t_{α/2, df} * SE]
    """
    tcrit = stats.t.ppf(1 - (1-confidence)/2, df=m.df)  # kritischer t-Wert
    halfwidth = tcrit * m.se                             # Halbbreite des Intervalls
    return (m.mean - halfwidth, m.mean + halfwidth)


def build_ci_table(m: MeanCI, conf_levels: List[float]) -> pd.DataFrame:
    """
    Baut eine Tabelle mit CI-Grenzen, Breite etc. für mehrere Konfidenzniveaus.
    Spalten:
    - Konfidenzniveau, df, Mittelwert (ȳ), SE(ȳ), CI_unten, CI_oben, Breite
    """
    rows = []
    for c in conf_levels:
        lo, hi = t_confidence_interval(m, c)
        rows.append({
            "Konfidenzniveau": f"{int(c*100)}%",
            "df": m.df,
            "Mittelwert (ȳ)": m.mean,
            "SE(ȳ)": m.se,
            "CI_unten": lo,
            "CI_oben": hi,
            "Breite": hi - lo
        })
    return pd.DataFrame(rows)


# Visualisierungen für (c)(d)
def plot_mean_se_and_ci(m: MeanCI, ci_tbl: pd.DataFrame, title_prefix= f"{ticker} (10 Handelstage)"):
    """
    Zwei Abbildungen:
    1) Balken = Mittelwert, Errorbar = ± SE
    2) Errorbars für verschiedene Konfidenzniveaus (Zentren + Halbbreiten)
    """
    log.info(m)
    # (1) Mittelwert & SE (Errorbar)
    plt.figure(figsize=(8, 4))
    plt.errorbar(["ȳ"], [m.mean], yerr=[m.se], fmt="o", capsize=5, color="tab:blue")
    plt.title(f"{title_prefix} – Mittelwert & Standardfehler")
    plt.ylabel("Tägliche Rendite")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # (2) t-Konfidenzintervalle als Errorbars
    plt.figure(figsize=(7,6))
    xs = list(ci_tbl["Konfidenzniveau"])
    # Mittelpunkt und Halbbreite je Intervall für die Darstellung
    centers = [ (lo + hi) / 2 for lo, hi in zip(ci_tbl["CI_unten"], ci_tbl["CI_oben"]) ]
    half = [ (hi - lo) / 2 for lo, hi in zip(ci_tbl["CI_unten"], ci_tbl["CI_oben"]) ]
    plt.errorbar(xs, centers, yerr=half, fmt="o", capsize=6)
    plt.axhline(0, color="black", linewidth=1)  # Referenzlinie bei 0-Rendite
    plt.title(f"{title_prefix} – t-Konfidenzintervalle (df={m.df})")
    plt.ylabel("Tägliche Rendite (Mittelwert ± Halbbreite)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==============================================================
# (e) VERGLEICH DER CI-BREITEN (Grafik)
# ==============================================================

def plot_ci_widths(ci_table: pd.DataFrame, title: str = "Breite der Konfidenzintervalle (n=10)"):
    """
    Balkendiagramm der Intervallbreiten für verschiedene Konfidenzniveaus.
    Interpretation: Höheres Konfidenzniveau → breiteres Intervall.
    """
    plt.figure(figsize=(7,4))
    x = ci_table["Konfidenzniveau"]
    y = ci_table["Breite"]
    plt.bar(x, y)
    plt.ylabel("Breite des Intervalls")
    plt.xlabel("Konfidenzniveau")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==============================================================
# ERWEITERUNG: rollierende 10-Tages-Intervalle & Chunking
# ==============================================================

def rolling_ci_analysis(df: pd.DataFrame, window: int = 10, conf: float = 0.95) -> pd.DataFrame:
    """
    Rollierende Analyse: Für jedes 10-Tages-Fenster (oder 'window') wird
    der Mittelwert der Renditen samt t-Konfidenzintervall berechnet.

    Rückgabe-Spalten:
    - EndDate: Enddatum des Fensters
    - Mean: 10-Tages-Mittelrendite
    - Lower/Upper: CI-Grenzen
    - Width: CI-Breite
    """
    rets = df["Return"].dropna()
    results = []
    # Fenster über die Renditeserie schieben
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
    Visualisiert eine rollierende 10-Tages-Mittelrendite samt CI-Band.
    - ylims/ystep erlauben einheitliche Skalen (vergleichbar über Plots)
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

    # Einheitliche Skala erzwingen (optional)
    if ylims is not None:
        plt.ylim(*ylims)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(ystep))

    plt.tight_layout()
    plt.show()


def plot_rolling_ci_all(df_dict: Dict[float, pd.DataFrame],
                        ylims: Tuple[float, float] | None = None,
                        ystep: float = 0.01):
    """
    Zeichnet 90%, 95% und 99% rollierende CIs gemeinsam.
    - Breite Intervalle zuerst füllen (Z-Ordnung), schmalere obenauf.
    - Eine gemeinsame Mean-Linie reicht (identisch über Konfidenzniveaus).
    """
    plt.figure(figsize=(11, 6))

    # Zeichenreihenfolge: von breit (99) zu schmal (90)
    conf_order = sorted(df_dict.keys(), reverse=True)
    # Farblegende (frei wählbar, nur zur Unterscheidung)
    colors = {0.90: "#4CAF50", 0.95: "#FFC107", 0.99: "#F44336"}  # grün, gelb, rot

    # Konfidenzbänder füllen
    for conf in conf_order:
        df = df_dict[conf]
        plt.fill_between(df["EndDate"], df["Lower"], df["Upper"],
                         color=colors.get(conf, "gray"), alpha=0.3,
                         label=f"{int(conf*100)}% CI")

    # Eine Mean-Linie (gleich für alle conf)
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

    plt.tight_layout()
    plt.show()


def chunked_ci_analysis(df: pd.DataFrame, window: int = 10, conf: float = 0.95) -> pd.DataFrame:
    """
    "Chunking" statt rollierend:
    - Die Renditeserie wird in disjunkte Blöcke der Länge 'window' zerlegt.
    - Für jeden vollen Block wird ein CI berechnet (nicht überlappend).

    Rückgabe-Spalten:
    - Period (laufende Nummer), Start, End, Mean, Lower, Upper, Width
    """
    rets = df["Return"].dropna().reset_index()
    chunks = [rets.iloc[i:i+window] for i in range(0, len(rets), window)]
    results = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) < window:
            continue  # letzte Restlänge verwerfen
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

