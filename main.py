import os
import time
import pandas as pd
import requests

# 1. 设置 API Key
FRED_API_KEY = "C4526744E95B3CBC93C23E9746574C96"
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY is not configured.")

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
TOL = {"daily": 7, "weekly": 10, "monthly": 3}

# 第一版核心数据字典
INDICATOR_MAP = {
    # 模块 01：资本成本与金融条件
    "DGS10": {"name": "10Y Treasury Yield", "type": "level", "freq": "daily"},
    "DGS2": {"name": "2Y Treasury Yield", "type": "level", "freq": "daily"},
    "DFII10": {"name": "10Y Real Yield (TIPS)", "type": "level", "freq": "daily"},
    "SOFR": {"name": "SOFR Rate", "type": "level", "freq": "daily"},
    "NFCI": {"name": "Financial Conditions Index", "type": "level", "freq": "weekly"},
    
    # 模块 02：增长与通胀机制
    "CPILFESL": {"name": "Core CPI Index", "type": "price_index", "freq": "monthly"},
    "PCEPILFE": {"name": "Core PCE Index", "type": "price_index", "freq": "monthly"},
    "UNRATE": {"name": "Unemployment Rate", "type": "level", "freq": "monthly"},
    "SAHMREALTIME": {"name": "Sahm Rule Recession Indicator", "type": "level", "freq": "monthly"},
    "PAYEMS": {"name": "Nonfarm Payrolls (Thousands)", "type": "level", "freq": "monthly"},
    "ICSA": {"name": "Initial Jobless Claims", "type": "level", "freq": "weekly"},
    
    # 模块 03：信用与流动性压力
    "BAMLH0A0HYM2": {"name": "US High Yield Option-Adjusted Spread", "type": "level", "freq": "daily"},
    "WALCL": {"name": "Fed Total Assets", "type": "level", "freq": "weekly"},
    "WRBWFRBL": {"name": "Reserve Balances with Fed", "type": "level", "freq": "weekly"},
    "RRPONTSYD": {"name": "Overnight Reverse Repurchase Agreements", "type": "level", "freq": "daily"}
}

def fetch_fred_series(series_id, retries=3):
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"{series_id}: HTTP {r.status_code} Client Error")
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == retries - 1:
                status_code = getattr(getattr(e, "response", None), "status_code", "N/A")
                raise RuntimeError(f"{series_id}: Fetch failed (HTTP Status: {status_code})") from None
            time.sleep(2**attempt)

    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").drop_duplicates("date")
    return df.reset_index(drop=True)

def lagged(s, months, tol_days):
    target = s.index - pd.DateOffset(months=months)
    vals = s.reindex(target, method="ffill", tolerance=pd.Timedelta(days=tol_days)).values
    return pd.Series(vals, index=s.index)

def process_price_index(df, freq="monthly"):
    tol = TOL[freq]
    s = df.set_index("date")["value"]
    out = pd.DataFrame({"value": s})
    out["yoy"] = (s / lagged(s, 12, tol) - 1) * 100
    out["ann_3m"] = ((s / lagged(s, 3, tol)) ** 4 - 1) * 100
    out["acceleration"] = out["ann_3m"] - lagged(out["ann_3m"], 3, tol)
    return out

def process_level_series(df, freq="daily"):
    tol = TOL[freq]
    s = df.set_index("date")["value"]
    out = pd.DataFrame({"value": s})
    out["change_1m"] = s - lagged(s, 1, tol)
    out["change_3m"] = s - lagged(s, 3, tol)
    out["acceleration"] = out["change_3m"] - lagged(out["change_3m"], 3, tol)
    return out

# --- 运行验证 ---
summary_data = []

for series_id, cfg in INDICATOR_MAP.items():
    try:
        raw_df = fetch_fred_series(series_id)
        if cfg["type"] == "price_index":
            processed = process_price_index(raw_df, freq=cfg["freq"])
            latest = processed.iloc[-1]
            summary_data.append({
                "Series ID": series_id,
                "Name": cfg["name"],
                "Latest Date": raw_df["date"].iloc[-1].strftime("%Y-%m-%d"),
                "Level / YoY": f"{latest['yoy']:.2f}% (YoY)",
                "3M Ann / 1M Chg": f"{latest['ann_3m']:.2f}% (3M Ann)",
                "Acceleration": f"{latest['acceleration']:.2f}"
            })
        else:
            processed = process_level_series(raw_df, freq=cfg["freq"])
            latest = processed.iloc[-1]
            summary_data.append({
                "Series ID": series_id,
                "Name": cfg["name"],
                "Latest Date": raw_df["date"].iloc[-1].strftime("%Y-%m-%d"),
                "Level / YoY": f"{latest['value']:.2f}",
                "3M Ann / 1M Chg": f"{latest['change_1m']:+.2f} (1M Chg)",
                "Acceleration": f"{latest['acceleration']:+.2f}"
            })
    except Exception as e:
        print(f"Error processing {series_id}: {e}")

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))