# Scientific Programming – Project Work (FS2026)

## Swiss Blue-Chip Stock Return Analysis
### Confidence Intervals, Statistical Testing & Market Scenarios

**Authors:** Tyler Storz, Alen Rama, Noel Mörgeli

---

### Overview
This project analyses daily returns of Swiss blue-chip stocks (UBS, Nestlé, Novartis) using statistical methods (confidence intervals, hypothesis testing) and compares real market data against synthetic scenarios (bull, bear, volatile, crisis, neutral).

### Features
- **Data Collection**: Real-world stock data via Yahoo Finance API (`yfinance`) for multiple tickers
- **Data Preparation**: Regex-based column cleaning, forward-filling, return calculation
- **Statistical Analysis**: t-confidence intervals, one-sample & two-sample t-tests (p-values), Pearson correlation, rolling CI analysis
- **Visualisations**: Matplotlib/Seaborn charts (price series, return distributions, Q-Q plots, heatmaps)
- **SQLite Database**: Persistent storage of prices and CI results with SQL queries
- **LLM Integration**: OpenAI API-powered natural-language summaries of statistical results
- **Web Application**: Interactive Streamlit dashboard with 6 tabs
- **Synthetic Scenarios**: Market regime simulations (bull, bear, volatile, crisis, neutral)
- **Jupyter Notebook**: Full interactive walkthrough of the analysis

### Project Structure
```
├── analysis.py        # Core statistical functions (CI, reliability tables, rolling analysis)
├── scenarios.py       # Synthetic scenario generation & comparison plots
├── data_loader.py     # Yahoo Finance data fetching & preparation (regex, OOP)
├── database.py        # SQLite storage & SQL queries
├── llm_summary.py     # OpenAI LLM integration for automated analysis summaries
├── app.py             # Streamlit web application (6 tabs)
├── main.py            # CLI entry point
├── project.ipynb      # Jupyter notebook (main deliverable)
├── requirements.txt   # Python dependencies
├── .env               # OpenAI API key (gitignored, create manually)
├── .gitignore         # Git ignore rules
├── data/              # SQLite database (gitignored)
└── README.md
```

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd ScientificProgramming

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up OpenAI API key for LLM features
echo 'OPENAI_API_KEY=sk-your-key-here' > .env
```

### Usage

| Method | Command | Description |
|--------|---------|-------------|
| **Jupyter Notebook** | `jupyter notebook project.ipynb` | Main deliverable — full interactive analysis |
| **Streamlit Web App** | `python -m streamlit run app.py` | Interactive dashboard with 6 tabs |
| **CLI Script** | `python main.py` | Quick run, generates all plots |

### Requirements
- Python 3.9+
- See `requirements.txt` for all packages:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy` — data analysis & plotting
  - `yfinance` — Yahoo Finance API
  - `streamlit` — web application
  - `openai` — LLM integration
  - `jupyter`, `notebook` — Jupyter notebook

### Grading Criteria Covered

#### Minimum Requirements (8 points)
| # | Criterion | Implementation |
|---|-----------|---------------|
| 1 | Real-world data collection | `data_loader.py` — yfinance API for UBS, Nestlé, Novartis |
| 2 | Data preparation (regex) | `data_loader.py` — `re.sub()`, `re.search()` for column cleaning |
| 3 | Python data structures | `project.ipynb` §3 — lists, dicts, sets, tuples, DataFrames |
| 4 | Conditionals & loops | Throughout all modules |
| 5 | OOP / procedural | `DataLoader`, `MeanCI` dataclasses + procedural functions |
| 6 | Tables & visualisations | Notebook §4-6, Streamlit Overview tab |
| 7 | Statistical analysis (p-value) | One-sample t-test, two-sample t-test, Pearson correlation |
| 8 | Code available | Jupyter notebook + all `.py` modules |

#### Bonus Points (6 points)
| # | Criterion | Implementation |
|---|-----------|---------------|
| 1 | Creativity | Multi-ticker analysis, cross-correlation matrix, scenario comparison |
| 2 | Web API | `data_loader.py` — Yahoo Finance via `yfinance` |
| 3 | Database + SQL | `database.py` — SQLite with CREATE, INSERT, SELECT, GROUP BY |
| 4 | LLM integration | `llm_summary.py` — OpenAI GPT-4o-mini for automated summaries |
| 5 | Web application | `app.py` — Streamlit with 6 interactive tabs |
| 6 | GitHub repo | Public repository with `.gitignore` |

### AI Disclosure
Parts of this project were developed with the assistance of AI tools (GitHub Copilot / ChatGPT). All AI-generated code was reviewed, tested, and adapted by the team members. The statistical methodology, research question, analysis design, and interpretation of results are entirely our own work. AI was used as a productivity tool for code scaffolding, documentation, and boilerplate — similar to using Stack Overflow or library documentation.

### License
MIT

