# 🇮🇳 OptiMarket — AI-Powered Portfolio Optimization Agent for Indian Markets

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18.0-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An intelligent portfolio optimization agent built entirely on real Indian market data.**  
Zero hardcoded values. Zero paid APIs. 100% AMFI + RBI + NSE data.

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Data Sources](#-data-sources) • [ML Model](#-ml-model)

</div>

---

## 🎯 What is OptiMarket?

OptiMarket is a full-stack AI-powered portfolio optimization system designed specifically for **Indian retail investors**. It combines:

- 📈 **Machine Learning** — Gradient Boosting Classifier trained on 46,997 rows of real AMFI NAV data to generate BUY / SELL / HOLD signals
- 📊 **Markowitz Mean-Variance Optimization** — 10,000 Monte Carlo simulations to find the maximum Sharpe ratio portfolio
- 💹 **Bond Pricing Engine** — DCF-based pricing with Modified Duration, Convexity, and DV01 for Indian fixed-income instruments
- ⚡ **Stress Testing** — Portfolio P&L simulation under 9 RBI rate shock scenarios (−300 to +300 bps)
- 🌐 **Live Market Data** — Real-time RBI repo rate, G-Sec yields, NIFTY 50, and AMFI NAV data

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🏦 **Fund Universe** | 25+ Indian mutual fund schemes across 9 SEBI categories |
| 🤖 **AI Recommender** | GBM classifier with TimeSeriesSplit CV — no data leakage |
| 📉 **Risk Classifier** | Random Forest model classifying funds as Low / Medium / High risk |
| 🎯 **Gap Analysis** | Compares current portfolio vs. target yield, duration & volatility |
| 🧠 **5-Step Reasoning** | Plain-language AI explanation for every recommendation |
| 💰 **Bond Pricer** | G-Sec, PSU, Corporate bond pricing with rate shock analysis |
| 📋 **Stress Test** | Interest rate scenario analysis with P&L bar chart |
| 📡 **Live Rates** | RBI rate, 10Y G-Sec yield, T-Bill proxy — all from free APIs |

---

## 🏗️ Architecture

```
optimarket/
├── ml/
│   ├── step1_fetch_data.py        ← Downloads AMFI NAV + NSE + ETF data
│   ├── step2_build_features.py    ← Engineers 15 financial features from NAV
│   ├── step3_train_models.py      ← (Legacy) Original training script
│   └── step4_live_rates.py        ← Fetches RBI/FRED rates, builds fund universe
│
├── fix_and_train.py               ← ✅ Use this instead of step3
│
├── backend/
│   ├── main.py                    ← FastAPI application entry point
│   └── services/
│       ├── bond_pricer.py         ← DCF pricing, Duration, Convexity, DV01
│       ├── optimizer.py           ← Portfolio metrics + stress test
│       └── mv_optimizer.py        ← Markowitz MVO (Monte Carlo)
│
├── frontend/src/pages/
│   ├── Dashboard.jsx              ← Live Indian market indicators
│   ├── Portfolio.jsx              ← Fund builder with real AMFI data
│   ├── Optimizer.jsx              ← Gap analysis + AI reasoning
│   ├── BondPricer.jsx             ← Bond calculator with presets
│   └── StressTest.jsx             ← Rate shock P&L visualizer
│
├── models/                        ← Trained .pkl model files (auto-generated)
└── data/
    ├── raw/                       ← Downloaded market data
    └── processed/
        ├── fund_features.csv      ← Engineered ML features
        ├── fund_universe.csv      ← Fund metrics from NAV history
        ├── market_summary.json    ← Live RBI/NSE rates
        └── optimizer_scores.csv   ← ML predictions per fund
```

---

## 📡 Data Sources

| Data | Source | Series / API | Frequency |
|------|--------|-------------|-----------|
| Mutual Fund NAVs | AMFI via mfapi.in | 25+ scheme codes | Daily |
| NIFTY 50 & Stocks | NSE via Yahoo Finance | `^NSEI`, NIFTY50 tickers | Daily |
| RBI Repo Rate | FRED / IMF | `IRSTCI01INM156N` | Monthly |
| 10Y G-Sec Yield | FRED / IMF | `IRLTLT01INM156N` | Monthly |
| CPI Inflation | FRED / IMF | `INDCPIALLMINMEI` | Monthly |
| T-Bill Rate | Computed from AMFI Liquid NAVs | Annualized 91-day return | Daily |

> ✅ All sources are **free and publicly accessible** — no paid subscriptions required.

---

## 🤖 ML Model

### Feature Engineering (15 features from raw NAV data)

| Feature | Computation | Purpose |
|---------|------------|---------|
| `return_7d` | 7-day NAV % change | Short-term momentum |
| `return_30d` | 30-day NAV % change | Monthly return |
| `return_90d` | 90-day NAV % change | Quarterly return |
| `return_1y` | 252-day NAV % change | Annual return |
| `volatility_30d` | 30-day rolling std | Short-term risk |
| `sharpe_30d` | (return − RFR) / vol | Risk-adjusted return |
| `sortino_30d` | Downside deviation ratio | Downside risk |
| `momentum_score` | Composite trend indicator | Trend strength |
| `max_drawdown_1y` | Peak-to-trough decline | Tail risk |
| `above_ma20` | NAV > 20-day MA | Technical signal |
| `above_ma50` | NAV > 50-day MA | Technical signal |
| `ma_crossover` | MA20 crossed MA50 | Crossover signal |
| `category_code` | SEBI category encoded | Fund type |

### Model Configuration

```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_split=30,
    random_state=42
)
# Validation: TimeSeriesSplit(n_splits=5)  ← prevents data leakage
# Training rows: ~46,997 from real AMFI NAV history
```

### Label Distribution
```
BUY  ████████████░░░░░░░░  ~30%
HOLD █████████████████████  ~53%
SELL ███████░░░░░░░░░░░░░░  ~17%
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11+ → [python.org](https://python.org/downloads)
- Node.js LTS → [nodejs.org](https://nodejs.org)

### Step 1 — Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

### Step 2 — Download Indian Market Data
```bash
python ml/step1_fetch_data.py
# ✅ Wait for: "ALL DATA DOWNLOADED SUCCESSFULLY"

python ml/step2_build_features.py
# ✅ Wait for: "FEATURES BUILT SUCCESSFULLY"
```

### Step 3 — Train AI Models
```bash
python fix_and_train.py
# ✅ Wait for: "ALL MODELS TRAINED SUCCESSFULLY"
```

### Step 4 — Fetch Live Market Rates
```bash
python ml/step4_live_rates.py
# ✅ Wait for: "LIVE MARKET DATA FETCHED"
```

> 💡 Steps 1–4 only need to be run **once**.

---

## 🖥️ Running the App

**Terminal 1 — Backend**
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
# → Running on http://127.0.0.1:8000
```

**Terminal 2 — Frontend**
```bash
cd frontend
npm install   # first time only
npm start
# → Opens http://localhost:3000 automatically 🎉
```

---

## 📊 Sample Results

### Bond Pricing — GOI 7.26% 2033 (10-Year G-Sec)
| Metric | Value |
|--------|-------|
| Bond Price (YTM 7.12%) | ₹1,009.19 |
| Modified Duration | 6.540 years |
| Convexity | 52.34 |
| DV01 | ₹0.66 per ₹1,000 face value |

### Stress Test — Portfolio Duration 3.5 Years
| Rate Shock | P&L |
|-----------|-----|
| −300 bps | +10.5% |
| −100 bps | +3.5% |
| Base (0) | 0.00% |
| +100 bps | −3.5% |
| +300 bps | −10.5% |

---

## 👥 Team

| Name | Roll Number |
|------|------------|
| Aaryan Kumar | 24WU0104082 |
| Teja Venkatesh | 24WU0101010 |
| Aditya Ayush Patel | 24WU0101069 |
| Avanish Lonare | 24WU0104074 |
| Kanishk Raj | 24WU0104035 |

**Guide:** Dr. N. Karthick, Assistant Professor, School of Technology, Woxsen University

---

## 📄 License

This project is submitted as an academic project at Woxsen University under the B.Tech. AI & ML program (2024–2028). All data is sourced from publicly available APIs.

---

<div align="center">
Made with ❤️ for Indian investors · Woxsen University · 2024–2025
</div>
