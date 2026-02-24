"""
step4_live_rates.py
Fetches live Indian market rates from:
  - FRED (IMF series) → RBI Repo Rate, 10Y G-Sec Yield, CPI
  - Yahoo Finance     → NIFTY 50 level and 1Y return
  - AMFI Liquid NAVs  → T-Bill proxy rate
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

os.makedirs("data/processed", exist_ok=True)

print("=" * 60)
print("📡  OptiMarket — Step 4: Fetching Live Indian Market Rates")
print("=" * 60)

market = {}

# ─── RBI Repo Rate from FRED ──────────────────────────────────
print("\n🔄 [1/4] Fetching RBI Repo Rate from FRED (IMF series)...")
try:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRSTCI01INM156N"
    df_rbi = pd.read_csv(url, parse_dates=['DATE'])
    df_rbi = df_rbi.dropna()
    latest_rbi = float(df_rbi.iloc[-1]['IRSTCI01INM156N'])
    market['rbi_repo_rate'] = round(latest_rbi, 2)
    market['rbi_repo_date']  = str(df_rbi.iloc[-1]['DATE'].date())
    print(f"  ✅ RBI Repo Rate: {latest_rbi:.2f}%")
except Exception as e:
    market['rbi_repo_rate'] = 6.50
    print(f"  ⚠️  Using fallback: 6.50% ({e})")

# ─── 10Y G-Sec Yield from FRED ───────────────────────────────
print("\n🔄 [2/4] Fetching 10Y G-Sec Yield from FRED (IMF series)...")
try:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01INM156N"
    df_gsec = pd.read_csv(url, parse_dates=['DATE'])
    df_gsec = df_gsec.dropna()
    latest_gsec = float(df_gsec.iloc[-1]['IRLTLT01INM156N'])
    market['gsec_10y_yield'] = round(latest_gsec, 2)
    print(f"  ✅ 10Y G-Sec Yield: {latest_gsec:.2f}%")
except Exception as e:
    market['gsec_10y_yield'] = 7.12
    print(f"  ⚠️  Using fallback: 7.12% ({e})")

# ─── NIFTY 50 from Yahoo Finance ─────────────────────────────
print("\n🔄 [3/4] Fetching NIFTY 50 data from NSE via Yahoo Finance...")
try:
    nifty = yf.Ticker("^NSEI")
    hist  = nifty.history(period="1y")
    nifty_level  = round(float(hist['Close'].iloc[-1]), 2)
    nifty_1y_ret = round((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100, 2)
    nifty_vol    = round(hist['Close'].pct_change().std() * np.sqrt(252) * 100, 2)
    market['nifty_level']    = nifty_level
    market['nifty_1y_return'] = nifty_1y_ret
    market['nifty_volatility'] = nifty_vol
    print(f"  ✅ NIFTY 50: {nifty_level:,.2f} | 1Y Return: {nifty_1y_ret:.2f}% | Vol: {nifty_vol:.2f}%")
except Exception as e:
    market['nifty_level'] = 22500
    print(f"  ⚠️  Using fallback ({e})")

# ─── T-Bill Proxy from AMFI Liquid Fund ──────────────────────
print("\n🔄 [4/4] Computing T-Bill proxy from AMFI Liquid Fund NAVs...")
try:
    url  = "https://api.mfapi.in/mf/119568"  # SBI Liquid Fund
    resp = requests.get(url, timeout=15)
    data = resp.json()['data']
    nav_df = pd.DataFrame(data)
    nav_df['date'] = pd.to_datetime(nav_df['date'], format='%d-%m-%Y')
    nav_df['nav']  = pd.to_numeric(nav_df['nav'])
    nav_df = nav_df.sort_values('date').tail(95)
    tbill_proxy = ((nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[-92]) ** (365/91) - 1) * 100
    market['tbill_proxy_rate'] = round(tbill_proxy, 2)
    print(f"  ✅ T-Bill Proxy Rate (91-day): {tbill_proxy:.2f}%")
except Exception as e:
    market['tbill_proxy_rate'] = 6.85
    print(f"  ⚠️  Using fallback: 6.85% ({e})")

market['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open("data/processed/market_summary.json", "w") as f:
    json.dump(market, f, indent=2)

print(f"\n  💾 Saved → data/processed/market_summary.json")
print(f"\n  Summary:\n{json.dumps(market, indent=4)}")

print("\n" + "=" * 60)
print("✅  LIVE MARKET DATA FETCHED")
print("=" * 60)
print("\nNext → cd backend && python -m uvicorn main:app --reload --port 8000")
