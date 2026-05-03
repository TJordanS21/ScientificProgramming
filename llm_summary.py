"""
LLM-powered analysis summary generator.
Uses OpenAI API to produce natural-language interpretations of statistical results.
Demonstrates: LLM integration (bonus criterion 4).
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def _get_client():
    """Lazy-import openai and return a client."""
    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY environment variable to use LLM features. "
                "Example: export OPENAI_API_KEY='sk-...'"
            )
        return OpenAI(api_key=api_key)
    except ImportError:
        raise RuntimeError("Install openai package: pip install openai")


def summarise_statistics(
    ticker: str,
    descriptive_stats: pd.DataFrame,
    t_test_result: tuple[float, float],
    ci_table: pd.DataFrame,
    correlation_info: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Send statistical results to an LLM and return a plain-English summary.

    Args:
        ticker: stock ticker symbol
        descriptive_stats: output of df.describe()
        t_test_result: (t_statistic, p_value) from one-sample t-test
        ci_table: CI table DataFrame
        correlation_info: optional string with correlation details
        model: OpenAI model to use

    Returns:
        LLM-generated summary string
    """
    client = _get_client()

    prompt = f"""You are a financial data analyst. Summarise the following statistical 
analysis results for the stock ticker "{ticker}" in clear, concise English. 
Include interpretation of significance levels and practical implications.

## Descriptive Statistics
{descriptive_stats.to_string()}

## One-Sample t-Test (H₀: μ = 0)
t-statistic: {t_test_result[0]:.4f}
p-value: {t_test_result[1]:.6f}

## Confidence Intervals
{ci_table.to_string()}
"""
    if correlation_info:
        prompt += f"\n## Correlation Analysis\n{correlation_info}\n"

    prompt += """
Please provide:
1. A brief summary of the return distribution
2. Interpretation of the hypothesis test result
3. What the confidence intervals tell us
4. Any notable patterns or concerns
Keep it under 300 words.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a quantitative finance analyst providing statistical summaries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


def compare_tickers_summary(
    ticker_stats: dict[str, dict],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a comparative LLM summary across multiple tickers.

    Args:
        ticker_stats: dict mapping ticker -> {"mean": float, "std": float, "p_value": float, ...}
        model: OpenAI model to use

    Returns:
        LLM-generated comparative summary
    """
    client = _get_client()

    stats_text = "\n".join(
        f"- {sym}: mean_return={s['mean']:.6f}, std={s['std']:.6f}, "
        f"t_stat={s['t_stat']:.4f}, p_value={s['p_value']:.6f}"
        for sym, s in ticker_stats.items()
    )

    prompt = f"""Compare the following Swiss blue-chip stocks based on their return statistics:

{stats_text}

Provide a brief comparative analysis (under 200 words) covering:
1. Which stock has the highest/lowest average return?
2. Which is most/least volatile?
3. Are any returns statistically significant (p < 0.05)?
4. Investment implications
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a quantitative finance analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content

