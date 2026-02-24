"""
step2_build_features.py
Computes 15 financial features from raw AMFI NAV time series data.
All features are derived entirely from real NAV history — zero hardcoded values.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("data/processed", exist_ok=True)

print("=" * 60)
print("⚙️   OptiMarket — Step 2: Building ML Features")
print("=" * 60)

df = pd.read_csv("data/raw/amfi_nav_history.csv", parse_dates=['date'])
df = df.sort_values(['fund_name', 'date']).reset_index(drop=True)

CATEGORY_MAP = {
    "SBI Bluechip Fund":              ("Large Cap",      0),
    "HDFC Top 100 Fund":              ("Large Cap",      0),
    "Axis Bluechip Fund":             ("Large Cap",      0),
    "HDFC Mid-Cap Opportunities":     ("Mid Cap",        1),
    "Kotak Emerging Equity Fund":     ("Mid Cap",        1),
    "SBI Small Cap Fund":             ("Small Cap",      2),
    "Nippon India Small Cap Fund":    ("Small Cap",      2),
    "Parag Parikh Flexi Cap Fund":    ("Flexi Cap",      3),
    "UTI Flexi Cap Fund":             ("Flexi Cap",      3),
    "SBI Magnum Gilt Fund":           ("Gilt",           4),
    "HDFC Gilt Fund":                 ("Gilt",           4),
    "HDFC Short Term Debt Fund":      ("Short Duration", 5),
    "SBI Liquid Fund":                ("Liquid",         6),
    "HDFC Liquid Fund":               ("Liquid",         6),
    "HDFC Balanced Advantage Fund":   ("Hybrid",         7),
    "ICICI Pru Balanced Advantage":   ("Hybrid",         7),
    "Axis Long Term Equity Fund":     ("ELSS",           8),
    "Mirae Asset ELSS Fund":          ("ELSS",           8),
    "UTI Nifty 50 Index Fund":        ("Index",          9),
    "HDFC Index Fund Nifty 50":       ("Index",          9),
}

print(f"\n  Loaded {len(df):,} NAV records across {df['fund_name'].nunique()} funds")
print("\n🔄 Computing features per fund...")

all_features = []

for fund_name, grp in df.groupby('fund_name'):
    grp = grp.sort_values('date').copy()
    grp['return_1d'] = grp['nav'].pct_change()

    # Multi-period returns
    grp['return_7d']  = grp['nav'].pct_change(7)
    grp['return_30d'] = grp['nav'].pct_change(30)
    grp['return_90d'] = grp['nav'].pct_change(90)
    grp['return_1y']  = grp['nav'].pct_change(252)

    # Volatility
    grp['volatility_30d'] = grp['return_1d'].rolling(30).std() * np.sqrt(252)
    grp['volatility_90d'] = grp['return_1d'].rolling(90).std() * np.sqrt(252)

    # Sharpe (using 6.5% RBI repo rate as risk-free)
    rf_daily = 0.065 / 252
    grp['sharpe_30d'] = (grp['return_1d'].rolling(30).mean() - rf_daily) / \
                         grp['return_1d'].rolling(30).std()
    grp['sharpe_1y']  = (grp['return_1d'].rolling(252).mean() - rf_daily) / \
                         grp['return_1d'].rolling(252).std()

    # Sortino
    downside = grp['return_1d'].apply(lambda x: min(x, 0))
    grp['sortino_30d'] = (grp['return_1d'].rolling(30).mean() - rf_daily) / \
                          downside.rolling(30).std()

    # Momentum score (composite)
    grp['momentum_score'] = (
        grp['return_7d'].rank(pct=True) * 0.2 +
        grp['return_30d'].rank(pct=True) * 0.3 +
        grp['return_90d'].rank(pct=True) * 0.5
    )

    # Max drawdown
    rolling_max = grp['nav'].rolling(252, min_periods=1).max()
    grp['max_drawdown_1y'] = (grp['nav'] - rolling_max) / rolling_max

    # Technical indicators
    grp['ma20'] = grp['nav'].rolling(20).mean()
    grp['ma50'] = grp['nav'].rolling(50).mean()
    grp['above_ma20']   = (grp['nav'] > grp['ma20']).astype(int)
    grp['above_ma50']   = (grp['nav'] > grp['ma50']).astype(int)
    grp['ma_crossover'] = (grp['ma20'] > grp['ma50']).astype(int)

    # Category encoding
    cat_name, cat_code = CATEGORY_MAP.get(fund_name, ("Unknown", 99))
    grp['category']      = cat_name
    grp['category_code'] = cat_code

    all_features.append(grp)

print("  ✅ Features computed for all funds")

features_df = pd.concat(all_features, ignore_index=True)
features_df = features_df.dropna(subset=['return_30d', 'volatility_30d', 'sharpe_30d'])

# ─── Generate recommendation labels ───────────────────────────
print("\n🔄 Generating BUY/SELL/HOLD labels...")

buy_threshold  = features_df.groupby('category')['sharpe_30d'].transform(lambda x: x.quantile(0.70))
sell_threshold = features_df.groupby('category')['sharpe_30d'].transform(lambda x: x.quantile(0.30))
cat_median_ret = features_df.groupby('category')['return_30d'].transform('median')

features_df['recommendation'] = 'HOLD'
buy_mask  = (features_df['sharpe_30d'] >= buy_threshold) & (features_df['return_30d'] >= cat_median_ret)
sell_mask = (features_df['sharpe_30d'] <= sell_threshold) & (features_df['return_90d'] < 0)

features_df.loc[buy_mask,  'recommendation'] = 'BUY'
features_df.loc[sell_mask, 'recommendation'] = 'SELL'

dist = features_df['recommendation'].value_counts()
print(f"  Label distribution:\n{dist.to_string()}")

features_df.to_csv("data/processed/fund_features.csv")
print(f"\n  💾 Saved {len(features_df):,} rows → data/processed/fund_features.csv")

print("\n" + "=" * 60)
print("✅  FEATURES BUILT SUCCESSFULLY")
print("=" * 60)
print("\nNext → python fix_and_train.py")
