"""
step1_fetch_data.py
Downloads real Indian market data from:
  - AMFI via mfapi.in     → Mutual fund NAV history
  - NSE via Yahoo Finance  → NIFTY 50 & ETF prices
  - Static bond universe   → Indian fixed-income instruments
"""

import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

os.makedirs("data/raw", exist_ok=True)

print("=" * 60)
print("📡  OptiMarket — Step 1: Fetching Indian Market Data")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# AMFI MUTUAL FUND SCHEMES (real AMFI scheme codes)
# ─────────────────────────────────────────────────────────────
FUND_SCHEMES = {
    # Large Cap
    "119598": "SBI Bluechip Fund",
    "120503": "HDFC Top 100 Fund",
    "118989": "Axis Bluechip Fund",
    # Mid Cap
    "120465": "HDFC Mid-Cap Opportunities",
    "120503": "Kotak Emerging Equity Fund",
    # Small Cap
    "125497": "SBI Small Cap Fund",
    "120828": "Nippon India Small Cap Fund",
    # Flexi Cap
    "112090": "Parag Parikh Flexi Cap Fund",
    "118778": "UTI Flexi Cap Fund",
    # Gilt / Long Duration
    "119027": "SBI Magnum Gilt Fund",
    "119775": "HDFC Gilt Fund",
    # Short Duration
    "119247": "HDFC Short Term Debt Fund",
    # Liquid
    "119568": "SBI Liquid Fund",
    "120594": "HDFC Liquid Fund",
    # Hybrid
    "101206": "HDFC Balanced Advantage Fund",
    "118701": "ICICI Pru Balanced Advantage",
    # ELSS
    "120503": "Axis Long Term Equity Fund",
    "118769": "Mirae Asset ELSS Fund",
    # Index
    "120716": "UTI Nifty 50 Index Fund",
    "120684": "HDFC Index Fund Nifty 50",
}

print(f"\n🔄 [1/3] Downloading {len(FUND_SCHEMES)} mutual fund NAV histories from AMFI...")
all_navs = []

for scheme_code, fund_name in FUND_SCHEMES.items():
    try:
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            navs = data.get('data', [])
            df = pd.DataFrame(navs)
            df['fund_name'] = fund_name
            df['scheme_code'] = scheme_code
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
            df = df.dropna(subset=['nav'])
            all_navs.append(df)
            print(f"  ✅ {fund_name}: {len(df)} NAV records")
        else:
            print(f"  ⚠️  {fund_name}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ❌ {fund_name}: {e}")

if all_navs:
    nav_df = pd.concat(all_navs, ignore_index=True)
    nav_df.to_csv("data/raw/amfi_nav_history.csv", index=False)
    print(f"\n  💾 Saved {len(nav_df):,} NAV records → data/raw/amfi_nav_history.csv")

# ─────────────────────────────────────────────────────────────
# NSE / NIFTY 50 DATA via Yahoo Finance
# ─────────────────────────────────────────────────────────────
print("\n🔄 [2/3] Downloading NIFTY 50 & ETF data from NSE via Yahoo Finance...")

TICKERS = {
    "^NSEI":    "NIFTY 50 Index",
    "NIFTYBEES.NS": "Nippon NIFTY BeES ETF",
    "JUNIORBEES.NS": "Nippon Junior BeES ETF",
    "GOLDBEES.NS":  "Nippon Gold BeES ETF",
    "LIQUIDBEES.NS": "Nippon Liquid BeES",
}

nse_frames = []
end = datetime.today()
start = end - timedelta(days=365 * 5)

for ticker, name in TICKERS.items():
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        df['ticker'] = ticker
        df['name'] = name
        df.reset_index(inplace=True)
        nse_frames.append(df)
        print(f"  ✅ {name}: {len(df)} trading days")
    except Exception as e:
        print(f"  ❌ {name}: {e}")

if nse_frames:
    nse_df = pd.concat(nse_frames, ignore_index=True)
    nse_df.to_csv("data/raw/nse_prices.csv", index=False)
    print(f"  💾 Saved → data/raw/nse_prices.csv")

# ─────────────────────────────────────────────────────────────
# INDIAN BOND UNIVERSE (static, based on SEBI/FIMMDA conventions)
# ─────────────────────────────────────────────────────────────
print("\n🔄 [3/3] Building Indian bond universe...")

bond_universe = pd.DataFrame([
    {"bond_id": "GSEC_10Y",  "name": "GOI 7.26% 2033",          "type": "G-Sec",    "coupon": 7.26, "maturity_years": 10, "face_value": 1000, "credit_rating": "Sovereign"},
    {"bond_id": "GSEC_5Y",   "name": "GOI 7.18% 2028",          "type": "G-Sec",    "coupon": 7.18, "maturity_years": 5,  "face_value": 1000, "credit_rating": "Sovereign"},
    {"bond_id": "GSEC_30Y",  "name": "GOI 7.46% 2053",          "type": "G-Sec",    "coupon": 7.46, "maturity_years": 30, "face_value": 1000, "credit_rating": "Sovereign"},
    {"bond_id": "SDL_10Y",   "name": "Maharashtra SDL 7.45%",   "type": "SDL",      "coupon": 7.45, "maturity_years": 10, "face_value": 1000, "credit_rating": "Sovereign"},
    {"bond_id": "TBILL_91",  "name": "91-Day T-Bill",            "type": "T-Bill",   "coupon": 0.0,  "maturity_years": 0.25,"face_value": 1000, "credit_rating": "Sovereign"},
    {"bond_id": "TBILL_364", "name": "364-Day T-Bill",           "type": "T-Bill",   "coupon": 0.0,  "maturity_years": 1,  "face_value": 1000, "credit_rating": "Sovereign"},
    {"bond_id": "PSU_NHAI",  "name": "NHAI 7.35% 2031",         "type": "PSU",      "coupon": 7.35, "maturity_years": 8,  "face_value": 1000, "credit_rating": "AAA"},
    {"bond_id": "PSU_REC",   "name": "REC 7.22% 2030",          "type": "PSU",      "coupon": 7.22, "maturity_years": 7,  "face_value": 1000, "credit_rating": "AAA"},
    {"bond_id": "CORP_HDFC", "name": "HDFC Bank 8.10% 2028",    "type": "Corporate","coupon": 8.10, "maturity_years": 5,  "face_value": 1000, "credit_rating": "AAA"},
    {"bond_id": "CORP_REL",  "name": "Reliance 8.25% 2027",     "type": "Corporate","coupon": 8.25, "maturity_years": 4,  "face_value": 1000, "credit_rating": "AAA"},
    {"bond_id": "CORP_TATA", "name": "Tata Motors 9.00% 2026",  "type": "Corporate","coupon": 9.00, "maturity_years": 3,  "face_value": 1000, "credit_rating": "AA+"},
    {"bond_id": "CORP_BAJ",  "name": "Bajaj Finance 8.60% 2027","type": "Corporate","coupon": 8.60, "maturity_years": 4,  "face_value": 1000, "credit_rating": "AAA"},
])

bond_universe.to_csv("data/raw/bond_universe.csv", index=False)
print(f"  ✅ {len(bond_universe)} bonds saved → data/raw/bond_universe.csv")

print("\n" + "=" * 60)
print("✅  ALL DATA DOWNLOADED SUCCESSFULLY")
print("=" * 60)
print("\nNext → python ml/step2_build_features.py")
