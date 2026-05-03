# Scientific Programming – Project Work (FS2026)

## UBS Stock Return Analysis: Confidence Intervals & Market Scenarios

### Overview
This project analyses daily returns of UBS stock using statistical methods (confidence intervals, hypothesis testing) and compares real market data against synthetic scenarios (bull, bear, volatile, crisis, neutral).

### Features
- **Data Collection**: Real-world stock data via Yahoo Finance API (`yfinance`)
- **Data Preparation**: Regex-based column cleaning, forward-filling, return calculation
- **Statistical Analysis**: t-confidence intervals, one-sample t-test (p-value), rolling CI analysis
- **Visualisations**: Matplotlib/Seaborn charts for data exploration
- **SQLite Database**: Persistent storage of prices and analysis results
- **Synthetic Scenarios**: Monte Carlo-style market regime simulations
- **Jupyter Notebook**: Interactive walkthrough of the full analysis

### Project Structure
```
├── analysis.py        # Core statistical functions (CI, reliability tables, plots)
├── scenarios.py       # Synthetic scenario generation & comparison plots
├── DataLoader.py      # Yahoo Finance data fetching & preparation
├── database.py        # SQLite storage & SQL queries
├── main.py            # CLI entry point
├── project.ipynb      # Jupyter notebook (main deliverable)
├── data/              # Database & cached data (gitignored)
├── models/            # Saved models (if any)
└── README.md
```

### Setup
```bash
pip install -r requirements.txt
```

### Usage
```bash
# Run via notebook
jupyter notebook project.ipynb

# Or run via CLI
python main.py
```

### Requirements
- Python 3.10+
- See `requirements.txt`

### Authors
- [Your names here]

### License
MIT
