"""
SQLite database layer for storing and querying stock data and analysis results.
Demonstrates: SQL queries, database integration (bonus criterion 3).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "stocks.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a connection, creating the parent directory if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


# ── Schema ──────────────────────────────────────────────────────
def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create tables if they don't exist."""
    conn = conn or get_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker   TEXT    NOT NULL,
            date     TEXT    NOT NULL,
            price    REAL    NOT NULL,
            return_  REAL,
            PRIMARY KEY (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS ci_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            confidence      REAL   NOT NULL,
            df              INTEGER,
            mean            REAL,
            se              REAL,
            ci_lower        REAL,
            ci_upper        REAL,
            width           REAL,
            window_start    TEXT,
            window_end      TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()


# ── Write helpers ───────────────────────────────────────────────
def store_prices(df: pd.DataFrame, ticker: str,
                 conn: Optional[sqlite3.Connection] = None) -> int:
    """
    Insert price/return rows from a prepared DataFrame.
    Returns number of rows inserted.
    """
    conn = conn or get_connection()
    init_db(conn)
    rows = []
    for date, row in df.iterrows():
        rows.append((ticker, str(date.date()), float(row["Price"]),
                      float(row["Return"]) if pd.notna(row.get("Return")) else None))
    conn.executemany(
        "INSERT OR REPLACE INTO stock_prices (ticker, date, price, return_) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def store_ci_results(ci_tbl: pd.DataFrame, ticker: str,
                     window_start: str = "", window_end: str = "",
                     conn: Optional[sqlite3.Connection] = None) -> None:
    """Persist a CI table (output of build_ci_table) into the database."""
    conn = conn or get_connection()
    init_db(conn)
    for _, r in ci_tbl.iterrows():
        conn.execute(
            """INSERT INTO ci_results
               (ticker, confidence, df, mean, se, ci_lower, ci_upper, width,
                window_start, window_end)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (ticker,
             r.get("1 - α", r.get("Konfidenzniveau", 0)),
             int(r.get("df", 0)),
             float(r.get("Mittelwert (ȳ)", r.get("Mean", 0))),
             float(r.get("SE(ȳ)", r.get("se", 0))),
             float(r.get("CI_unten", r.get("Lower", 0))),
             float(r.get("CI_oben", r.get("Upper", 0))),
             float(r.get("Breite", r.get("Width", 0))),
             window_start, window_end),
        )
    conn.commit()


# ── Read helpers (SQL queries) ──────────────────────────────────
def query_prices(ticker: str, limit: int = 100,
                 conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """SELECT prices for a given ticker, most recent first."""
    conn = conn or get_connection()
    return pd.read_sql_query(
        "SELECT * FROM stock_prices WHERE ticker = ? ORDER BY date DESC LIMIT ?",
        conn, params=(ticker, limit),
    )


def query_ci_results(ticker: str,
                     conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """SELECT all stored CI results for a ticker."""
    conn = conn or get_connection()
    return pd.read_sql_query(
        "SELECT * FROM ci_results WHERE ticker = ? ORDER BY created_at DESC",
        conn, params=(ticker,),
    )


def query_avg_return_by_ticker(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """Aggregate query: average return per ticker."""
    conn = conn or get_connection()
    return pd.read_sql_query(
        "SELECT ticker, COUNT(*) as n_days, AVG(return_) as avg_return, "
        "MIN(date) as first_date, MAX(date) as last_date "
        "FROM stock_prices GROUP BY ticker ORDER BY avg_return DESC",
        conn,
    )

