"""
Streamlit web app for Swiss stock return analysis.
Demonstrates: simple web app (bonus criterion 5).
Run with: streamlit run app.py
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from analysis import (TICKERS, summarize_returns_for_ci, build_ci_table,
                      t_confidence_interval, rolling_ci_analysis)
from data_loader import DataLoader, prepare_data
from scenarios import generate_scenario

st.set_page_config(page_title="Swiss Stock Analysis", page_icon="\U0001F4C8", layout="wide")
st.title("\U0001F4C8 Swiss Blue-Chip Stock Return Analysis")
st.markdown("**Tyler Storz, Alen Rama, Noel Mörgeli**")
st.markdown("---")

st.sidebar.header("Settings")
selected = st.sidebar.multiselect("Tickers", TICKERS, default=TICKERS)
lookback = st.sidebar.slider("Lookback (days)", 30, 365, 180)
n_ci = st.sidebar.slider("Days for CI", 5, 60, 10)
confs = st.sidebar.multiselect("Confidence levels", [0.80, 0.90, 0.95, 0.98, 0.99], default=[0.90, 0.95, 0.99])


@st.cache_data(ttl=3600, show_spinner="Fetching data...")
def load(tickers, lb):
    loader = DataLoader(tickers=tickers, lookback_days=lb)
    raw = loader.fetch_all()
    prep = {s: prepare_data(d, n_trading_days=None) for s, d in raw.items()}
    return raw, prep


if not selected:
    st.warning("Select at least one ticker.")
    st.stop()

raw_data, prepared = load(selected, lookback)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "CIs", "Tests", "Rolling", "Scenarios", "LLM Summary"]
)

with tab1:
    st.header("Data Overview")
    for sym in selected:
        df = prepared[sym]
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df.index, df["Price"])
            ax.set_title(f"{sym} Price")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(df["Return"], bins=30, edgecolor="black", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--")
            ax.set_title(f"{sym} Returns")
            st.pyplot(fig)
            plt.close()
        st.dataframe(df.describe().T, use_container_width=True)

    if len(selected) > 1:
        st.subheader("Correlation Matrix")
        rm = pd.DataFrame({s: prepared[s]["Return"] for s in selected}).dropna()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(rm.corr(), annot=True, cmap="RdYlGn", center=0, ax=ax, fmt=".3f")
        st.pyplot(fig)
        plt.close()

with tab2:
    st.header(f"Confidence Intervals ({n_ci} days)")
    for sym in selected:
        st.subheader(sym)
        try:
            sn = prepare_data(raw_data[sym], n_trading_days=n_ci)
        except RuntimeError as e:
            st.error(str(e))
            continue
        m = summarize_returns_for_ci(sn["Return"])
        ci = build_ci_table(m, confs)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean", f"{m.mean:.6f}")
            st.metric("SE", f"{m.se:.6f}")
        with col2:
            st.dataframe(ci, use_container_width=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(ci["Confidence Level"], ci["Width"])
        ax.set_ylabel("Width")
        ax.set_title(f"{sym} CI Widths")
        st.pyplot(fig)
        plt.close()

with tab3:
    st.header("Hypothesis Tests")
    st.subheader("One-Sample t-Test (H0: mu=0)")
    res = []
    for sym in selected:
        r = prepared[sym]["Return"]
        t_s, p_v = stats.ttest_1samp(r, 0)
        res.append({
            "Ticker": sym, "n": len(r),
            "Mean": f"{r.mean():.6f}",
            "t": f"{t_s:.4f}",
            "p": f"{p_v:.6f}",
            "Sig": "Yes" if p_v < 0.05 else "No"
        })
    st.dataframe(pd.DataFrame(res), use_container_width=True)

    if len(selected) >= 2:
        st.subheader(f"Two-Sample: {selected[0]} vs {selected[1]}")
        t2, p2 = stats.ttest_ind(
            prepared[selected[0]]["Return"],
            prepared[selected[1]]["Return"]
        )
        verdict = "Different means" if p2 < 0.05 else "No significant difference"
        st.write(f"t={t2:.4f}, p={p2:.6f} -> {verdict}")

    st.subheader("Autocorrelation (lag-1)")
    ar = []
    for sym in selected:
        r = prepared[sym]["Return"].dropna()
        c, cp = stats.pearsonr(r.iloc[1:].values, r.shift(1).dropna().values)
        ar.append({
            "Ticker": sym,
            "Pearson r": f"{c:.4f}",
            "p": f"{cp:.6f}",
            "Sig": "Yes" if cp < 0.05 else "No"
        })
    st.dataframe(pd.DataFrame(ar), use_container_width=True)

with tab4:
    st.header("Rolling 10-Day CI")
    colors_map = {0.80: "#2196F3", 0.90: "#4CAF50", 0.95: "#FFC107",
                  0.98: "#FF9800", 0.99: "#F44336"}
    for sym in selected:
        st.subheader(sym)
        rds = {c: rolling_ci_analysis(prepared[sym], window=10, conf=c) for c in confs}
        fig, ax = plt.subplots(figsize=(10, 5))
        for cf in sorted(rds, reverse=True):
            d = rds[cf]
            ax.fill_between(d["EndDate"], d["Lower"], d["Upper"],
                            color=colors_map.get(cf, "gray"), alpha=0.25,
                            label=f"{int(cf*100)}%")
        md = next(iter(rds.values()))
        ax.plot(md["EndDate"], md["Mean"], color="black", linewidth=1.2, label="Mean")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{sym} Rolling CI")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab5:
    st.header("Synthetic Scenarios")
    primary = selected[0]
    try:
        s60 = prepare_data(raw_data[primary], n_trading_days=60)
    except RuntimeError:
        s60 = prepared[primary].iloc[-60:]
    bp = float(s60["Price"].iloc[0])

    scens = {f"Real {primary}": s60}
    for s in ["bull", "bear", "volatile", "crisis", "neutral"]:
        scens[s.capitalize()] = generate_scenario(s, 60, 42, bp)

    fig, ax = plt.subplots(figsize=(10, 5))
    for nm, sdf in scens.items():
        pr = sdf["Price"].to_numpy()
        if pr[0] != 0:
            pr = pr / pr[0] * bp
        ax.plot(np.arange(len(pr)), pr, label=nm)
    ax.set_title("Rebased Prices")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close()

    st.subheader("CI Width Comparison")
    wd = []
    for nm, sdf in scens.items():
        sa = sdf["Return"].dropna().iloc[:10]
        if len(sa) < 10:
            continue
        m = summarize_returns_for_ci(sa)
        for c in confs:
            lo, hi = t_confidence_interval(m, c)
            wd.append({"Scenario": nm, "Conf": f"{int(c*100)}%", "Width": hi - lo})
    wdf = pd.DataFrame(wd)
    fig, ax = plt.subplots(figsize=(10, 4))
    wdf.pivot(index="Scenario", columns="Conf", values="Width").plot(kind="bar", ax=ax)
    ax.set_ylabel("Width")
    ax.set_title("CI Widths by Scenario")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab6:
    st.header("LLM-Powered Analysis Summary")
    st.markdown("Uses OpenAI API to generate natural-language interpretations of the statistical results.")
    st.markdown("> Requires `OPENAI_API_KEY` in environment or `.env` file.")

    if st.button("Generate LLM Summary"):
        try:
            from llm_summary import summarise_statistics, compare_tickers_summary

            # Single-ticker summary
            primary_sym = selected[0]
            primary_rets = prepared[primary_sym]["Return"]
            t_s, p_v = stats.ttest_1samp(primary_rets, 0)
            try:
                p10 = prepare_data(raw_data[primary_sym], n_trading_days=10)
            except RuntimeError:
                p10 = prepared[primary_sym].iloc[-10:]
            from analysis import summarize_returns_for_ci, build_ci_table
            m = summarize_returns_for_ci(p10["Return"])
            ci = build_ci_table(m, [0.90, 0.95, 0.99])

            with st.spinner("Calling OpenAI API..."):
                summary = summarise_statistics(
                    ticker=primary_sym,
                    descriptive_stats=prepared[primary_sym].describe(),
                    t_test_result=(t_s, p_v),
                    ci_table=ci,
                )
            st.subheader(f"Summary for {primary_sym}")
            st.markdown(summary)

            # Multi-ticker comparison
            if len(selected) > 1:
                ticker_stats = {}
                for sym in selected:
                    r = prepared[sym]["Return"]
                    ts, pv = stats.ttest_1samp(r, 0)
                    ticker_stats[sym] = {"mean": r.mean(), "std": r.std(), "t_stat": ts, "p_value": pv}

                with st.spinner("Generating comparison..."):
                    comparison = compare_tickers_summary(ticker_stats)
                st.subheader("Cross-Ticker Comparison")
                st.markdown(comparison)

        except RuntimeError as e:
            st.error(f"LLM unavailable: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("*Scientific Programming Project - FS2026*")
