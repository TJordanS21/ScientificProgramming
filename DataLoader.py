from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from analysis import ticker, log


# ==============================================================
# (a) DATENBEREINIGUNG & -VORBEREITUNG
# ==============================================================

@dataclass
class DataLoader:
    """
    Loader-Klasse für Kursdaten via yfinance.

    Attribute:
    - lookback_days: Zeitraum in Kalendertagen, den wir laden
    """
    lookback_days: int = 180  # ~6 Monate

    def fetch(self) -> pd.DataFrame:
        """
        Lädt Kursdaten von Yahoo Finance.
        auto_adjust=True → Adjustierte Schlusskurse ("Close") ohne Dividenden/Splits.
        group_by="column" → flacheres Spaltenlayout.

        Rückgabe:
        - DataFrame mit OHLCV-Spalten (abhängig von yfinance) und Datumsindex.
        """
        log.info("Downloading %s (~%d days) from Yahoo Finance...",
                 ticker, self.lookback_days)
        df = yf.download(
            ticker,
            period=f"{self.lookback_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,     # Adjustierter 'Close'
            group_by="column"
        )
        return df


def prepare_data(raw: pd.DataFrame, n_trading_days: Optional[int] = 10) -> pd.DataFrame:
    """
    Bereinigung & Feature-Engineering:

    Schritte:
    1) Spalten „flatten“, falls MultiIndex (z. B. "UBS|Close")
    2) Preis-Spalte robust finden (jede Spalte mit "close" im Namen)
    3) Index sortieren, fehlende Werte vorwärts auffüllen
    4) Tägliche Renditen Return = Price.pct_change()
    5) Optional: auf die letzten n Handelstage beschränken

    Parameter:
    - raw: ungefilterter yfinance-DataFrame
    - n_trading_days: Anzahl der zu behaltenden Handelstage (None = alle)

    Rückgabe:
    - DataFrame mit Spalten: ["Price", "Return"] und Datumsindex
    """
    log.info("Preparing data (cleaning + engineering)...")
    if raw is None or raw.empty:
        raise RuntimeError("Empty dataframe from yfinance.")

    df = raw.copy()
    # (1) MultiIndex-Spalten (z. B. "UBS|Close") zu einfachen Strings zusammenführen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join([str(x) for x in tup if x is not None]) for tup in df.columns]

    # Clean column names: remove special chars, normalise whitespace using regex
    df.columns = [re.sub(r"[^\w|]+", "_", col).strip("_") for col in df.columns]

    # (2) Preis-Spalte heuristisch finden (beliebige Schreibweise von 'Close' erlaubt)
    price_col = next((col for col in df.columns if re.search(r"(?i)close", col)), None)
    if price_col is None:
        raise RuntimeError(f"No 'Close' column found in columns: {list(df.columns)}")

    # Nur die Preis-Spalte behalten und sauber benennen
    df = df[[price_col]].rename(columns={price_col: "Price"}).sort_index()

    # Preis als float; Lücken (z. B. feiertagsbedingt) werden vorwärts aufgefüllt
    df["Price"] = df["Price"].astype(float).ffill()
    # Einfache (nicht log-) Rendite: r_t = (P_t / P_{t-1}) - 1
    df["Return"] = df["Price"].pct_change()

    # Erste Rendite ist NaN (kein Vortag) → entfernen
    df = df.dropna(subset=["Return"])

    # (5) Optional auf die letzten n Handelstage beschränken
    if n_trading_days is not None:
        if len(df) < n_trading_days:
            raise RuntimeError(f"Not enough trading days: need {n_trading_days}, got {len(df)}")
        df = df.iloc[-n_trading_days:]
        log.info("Prepared %d trading days; last index=%s", len(df), df.index.max())
    else:
        log.info("Prepared full return series with %d rows; last index=%s", len(df), df.index.max())
    return df
