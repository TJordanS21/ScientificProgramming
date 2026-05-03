from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Dict

import pandas as pd
import yfinance as yf

from analysis import ticker, TICKERS, log


# ==============================================================
# Data Loading & Preparation
# ==============================================================

@dataclass
class DataLoader:
    """
    Loader for stock price data via Yahoo Finance API.

    Attributes:
        tickers: list of ticker symbols to fetch
        lookback_days: calendar days to look back
    """
    tickers: list = field(default_factory=lambda: list(TICKERS))
    lookback_days: int = 180

    def fetch(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance for a single ticker.

        Args:
            symbol: ticker to fetch (defaults to first in self.tickers)

        Returns:
            DataFrame with OHLCV columns and date index.
        """
        sym = symbol or self.tickers[0]
        log.info("Downloading %s (~%d days) from Yahoo Finance...",
                 sym, self.lookback_days)
        df = yf.download(
            sym,
            period=f"{self.lookback_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="column"
        )
        return df

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured tickers."""
        return {sym: self.fetch(sym) for sym in self.tickers}


def prepare_data(raw: pd.DataFrame, n_trading_days: Optional[int] = 10) -> pd.DataFrame:
    """
    Clean and engineer features from raw price data.

    Steps:
        1. Flatten MultiIndex columns
        2. Find price column using regex
        3. Sort index, forward-fill missing values
        4. Compute daily returns
        5. Optionally trim to last n trading days
    """
    log.info("Preparing data (cleaning + feature engineering)...")
    if raw is None or raw.empty:
        raise RuntimeError("Empty dataframe from yfinance.")

    df = raw.copy()
    # (1) Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join([str(x) for x in tup if x is not None]) for tup in df.columns]

    # Clean column names with regex
    df.columns = [re.sub(r"[^\w|]+", "_", col).strip("_") for col in df.columns]

    # (2) Find price column heuristically
    price_col = next((col for col in df.columns if re.search(r"(?i)close", col)), None)
    if price_col is None:
        raise RuntimeError(f"No 'Close' column found in columns: {list(df.columns)}")

    df = df[[price_col]].rename(columns={price_col: "Price"}).sort_index()

    df["Price"] = df["Price"].astype(float).ffill()
    df["Return"] = df["Price"].pct_change()
    df = df.dropna(subset=["Return"])

    if n_trading_days is not None:
        if len(df) < n_trading_days:
            raise RuntimeError(f"Not enough trading days: need {n_trading_days}, got {len(df)}")
        df = df.iloc[-n_trading_days:]
        log.info("Prepared %d trading days; last index=%s", len(df), df.index.max())
    else:
        log.info("Prepared full return series with %d rows; last index=%s", len(df), df.index.max())
    return df
