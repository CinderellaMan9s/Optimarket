"""
bond_pricer.py
DCF-based bond pricing engine for Indian fixed-income instruments.
Computes: Price, Modified Duration, Macaulay Duration, Convexity, DV01
Convention: Semi-annual compounding (standard for Indian G-Secs)
"""

import numpy as np


def price_bond(face_value: float, coupon_rate: float, ytm: float, maturity_years: float) -> dict:
    """
    Price a bond using DCF with semi-annual compounding.
    
    Args:
        face_value:    Face value in INR (typically 1000)
        coupon_rate:   Annual coupon rate as decimal (e.g. 0.0726 for 7.26%)
        ytm:           Yield to maturity as decimal (e.g. 0.0712 for 7.12%)
        maturity_years: Years to maturity
    
    Returns:
        dict with price, duration, convexity, dv01, current_yield
    """
    n_periods   = int(maturity_years * 2)   # semi-annual periods
    semi_coupon = face_value * coupon_rate / 2
    semi_ytm    = ytm / 2

    # Edge case: zero-coupon / T-bill
    if n_periods == 0:
        price = face_value / (1 + ytm) ** maturity_years
        return {
            "price": round(price, 4),
            "macaulay_duration": round(maturity_years, 4),
            "modified_duration": round(maturity_years / (1 + ytm), 4),
            "convexity": 0.0,
            "dv01": round(price * maturity_years / (1 + ytm) / 10000, 4),
            "current_yield": 0.0,
            "price_to_face": round(price / face_value * 100, 4),
            "ytm_pct": round(ytm * 100, 4),
            "coupon_pct": round(coupon_rate * 100, 4),
        }

    periods = np.arange(1, n_periods + 1)
    cash_flows = np.full(n_periods, semi_coupon)
    cash_flows[-1] += face_value  # principal at maturity

    discount_factors = (1 + semi_ytm) ** periods
    pv_cash_flows    = cash_flows / discount_factors
    price            = pv_cash_flows.sum()

    # Macaulay Duration (in years)
    mac_duration = (pv_cash_flows * periods).sum() / price / 2

    # Modified Duration
    mod_duration = mac_duration / (1 + semi_ytm)

    # Convexity
    convexity_sum = (pv_cash_flows * periods * (periods + 1)).sum()
    convexity     = convexity_sum / (price * (1 + semi_ytm) ** 2 * 4)

    # DV01 (price change per 1 bps shift in yield)
    dv01 = mod_duration * price / 10000

    # Current yield
    annual_coupon = face_value * coupon_rate
    current_yield = annual_coupon / price if price > 0 else 0

    return {
        "price":             round(price, 4),
        "macaulay_duration": round(mac_duration, 4),
        "modified_duration": round(mod_duration, 4),
        "convexity":         round(convexity, 4),
        "dv01":              round(dv01, 4),
        "current_yield":     round(current_yield * 100, 4),
        "price_to_face":     round(price / face_value * 100, 4),
        "ytm_pct":           round(ytm * 100, 4),
        "coupon_pct":        round(coupon_rate * 100, 4),
    }


def stress_test_bond(face_value: float, coupon_rate: float, ytm: float,
                     maturity_years: float, quantity: int = 1) -> list:
    """
    Simulate bond P&L under 9 parallel yield curve shocks.
    Shocks in basis points: -300, -200, -100, -50, 0, +50, +100, +200, +300
    """
    base   = price_bond(face_value, coupon_rate, ytm, maturity_years)
    shocks = [-300, -200, -100, -50, 0, 50, 100, 200, 300]
    results = []

    for shock_bps in shocks:
        shocked_ytm   = max(0.0001, ytm + shock_bps / 10000)
        shocked_price = price_bond(face_value, coupon_rate, shocked_ytm, maturity_years)
        pnl_per_bond  = shocked_price['price'] - base['price']
        total_pnl     = pnl_per_bond * quantity

        results.append({
            "shock_bps":     shock_bps,
            "shocked_ytm":   round(shocked_ytm * 100, 4),
            "price":         shocked_price['price'],
            "pnl_per_bond":  round(pnl_per_bond, 4),
            "total_pnl":     round(total_pnl, 2),
            "pnl_pct":       round(pnl_per_bond / base['price'] * 100, 4),
        })

    return results


# Indian bond presets
INDIAN_BOND_PRESETS = {
    "GOI_10Y": {
        "name": "GOI 7.26% 2033 (10Y Benchmark G-Sec)",
        "face_value": 1000, "coupon_rate": 0.0726,
        "ytm": 0.0712, "maturity_years": 10,
    },
    "NHAI_PSU": {
        "name": "NHAI 7.35% 2031 (PSU Bond)",
        "face_value": 1000, "coupon_rate": 0.0735,
        "ytm": 0.0748, "maturity_years": 8,
    },
    "HDFC_CORP": {
        "name": "HDFC Bank 8.10% 2028 (Corporate AAA)",
        "face_value": 1000, "coupon_rate": 0.0810,
        "ytm": 0.0825, "maturity_years": 5,
    },
    "TBILL_91": {
        "name": "91-Day T-Bill (Zero Coupon)",
        "face_value": 1000, "coupon_rate": 0.0,
        "ytm": 0.0685, "maturity_years": 0.25,
    },
}
