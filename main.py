import os
import time
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许你的 WordPress 网站跨域调用这个 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 正式上线后可以改成你的 WordPress 域名 (https://minerva-invest.eu)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRED_API_KEY = "c4526744e95b3cbc93c23e9746574c96".strip().lower()
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
TOL = {"daily": 7, "weekly": 10, "monthly": 3}

INDICATOR_MAP = {
    "DGS10": {"name": "10Y Treasury Yield", "type": "level", "freq": "daily"},
    "DGS2": {"name": "2Y Treasury Yield", "type": "level", "freq": "daily"},
    "DFII10": {"name": "10Y Real Yield (TIPS)", "type": "level", "freq": "daily"},
    "T10YIE": {"name": "10Y Breakeven Inflation Rate", "type": "level", "freq": "daily"},
    "THREEFYTP10": {"name": "10Y Treasury Term Premium", "type": "level", "freq": "daily"},
    "SOFR": {"name": "SOFR Rate", "type": "level", "freq": "daily"},
    "IORB": {"name": "Interest Rate on Reserve Balances", "type": "level", "freq": "daily"},
    "NFCI": {"name": "Financial Conditions Index", "type": "level", "freq": "weekly"},
    "CPILFESL": {"name": "Core CPI Index", "type": "price_index", "freq": "monthly"},
    "PCEPILFE": {"name": "Core PCE Index", "type": "price_index", "freq": "monthly"},
    "UNRATE": {"name": "Unemployment Rate", "type": "level", "freq": "monthly"},
    "SAHMREALTIME": {"name": "Sahm Rule Recession Indicator", "type": "level", "freq": "monthly"},
    "PAYEMS": {"name": "Nonfarm Payrolls (Thousands)", "type": "level", "freq": "monthly"},
    "ICSA": {"name": "Initial Jobless Claims", "type": "level", "freq": "weekly"},
    "BAMLH0A0HYM2": {"name": "US High Yield Spread", "type": "level", "freq": "daily"},
    "WALCL": {"name": "Fed Total Assets", "type": "level", "freq": "weekly"},
    "WRBWFRBL": {"name": "Reserve Balances", "type": "level", "freq": "weekly"},
    "RRPONTSYD": {"name": "Overnight Reverse Repo", "type": "level", "freq": "daily"}
}

def fetch_fred_series(series_id, retries=3):
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if r.status_code != 200:
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    raise RuntimeError(f"{series_id}: HTTP {r.status_code}")
                r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise RuntimeError(f"{series_id}: Fetch failed") from None
            time.sleep(2**attempt)
    data = r.json()
    df = pd.DataFrame(data["observations"])[["date", "value"]]
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

@app.get("/api/minerva-macro")
def get_macro_data():
    summary_data = []
    raw_cache = {} # 用于暂存原始数据以计算扩展指标（如价差）

    for series_id, cfg in INDICATOR_MAP.items():
        try:
            raw_df = fetch_fred_series(series_id)
            raw_cache[series_id] = raw_df
            
            if cfg["type"] == "price_index":
                processed = process_price_index(raw_df, freq=cfg["freq"])
                latest = processed.iloc[-1]
                summary_data.append({
                    "series_id": series_id,
                    "name": cfg["name"],
                    "latest_date": raw_df["date"].iloc[-1].strftime("%Y-%m-%d"),
                    "level_or_yoy": round(float(latest['yoy']), 2),
                    "metric_type": "yoy_percent",
                    "short_term": round(float(latest['ann_3m']), 2),
                    "acceleration": round(float(latest['acceleration']), 2)
                })
            else:
                processed = process_level_series(raw_df, freq=cfg["freq"])
                latest = processed.iloc[-1]
                summary_data.append({
                    "series_id": series_id,
                    "name": cfg["name"],
                    "latest_date": raw_df["date"].iloc[-1].strftime("%Y-%m-%d"),
                    "level_or_yoy": round(float(latest['value']), 2),
                    "metric_type": "absolute_level",
                    "short_term": round(float(latest.get('change_1m', 0)), 2),
                    "acceleration": round(float(latest['acceleration']), 2)
                })
        except Exception as e:
            summary_data.append({"series_id": series_id, "name": cfg["name"], "error": str(e)})
            
    # 自动计算衍生指标：SOFR - IORB Spread (单位: bp)
    try:
        if "SOFR" in raw_cache and "IORB" in raw_cache:
            sofr_latest = raw_cache["SOFR"].iloc[-1]["value"]
            iorb_latest = raw_cache["IORB"].iloc[-1]["value"]
            spread_bp = (float(sofr_latest) - float(iorb_latest)) * 100.0
            latest_date = raw_cache["SOFR"].iloc[-1]["date"].strftime("%Y-%m-%d")
            
            summary_data.append({
                "series_id": "SOFR_IORB",
                "name": "SOFR - IORB Spread",
                "latest_date": latest_date,
                "level_or_yoy": round(spread_bp, 1),
                "metric_type": "absolute_level",
                "short_term": 0.0,
                "acceleration": 0.0
            })
    except Exception:
        pass # 若计算失败则静默跳过，不影响主流程

    return {
        "status": "success",
        "updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data": summary_data
    }