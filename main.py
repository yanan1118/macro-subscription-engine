import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://minerva-invest.eu", "https://www.minerva-invest.eu"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY is not set")

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
OBS_START = "2000-01-01"   # 够算 20 年分位，也避免每次拉全历史
CACHE_TTL = 3600           # 秒
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
    "CPILFESL": {"name": "Core CPI (YoY)", "type": "price_index", "freq": "monthly"},
    "PCEPILFE": {"name": "Core PCE (YoY)", "type": "price_index", "freq": "monthly"},
    "UNRATE": {"name": "Unemployment Rate", "type": "level", "freq": "monthly"},
    "SAHMREALTIME": {"name": "Sahm Rule Recession Indicator", "type": "level", "freq": "monthly"},
    "PAYEMS": {"name": "Nonfarm Payrolls", "type": "level", "freq": "monthly"},
    "ICSA": {"name": "Initial Jobless Claims", "type": "level", "freq": "weekly"},
    "BAMLH0A0HYM2": {"name": "US High Yield Spread", "type": "level", "freq": "daily"},
    "WALCL": {"name": "Fed Total Assets", "type": "level", "freq": "weekly"},
    "WRBWFRBL": {"name": "Reserve Balances", "type": "level", "freq": "weekly"},
    "RRPONTSYD": {"name": "Overnight Reverse Repo", "type": "level", "freq": "daily"},
}


def clean(x, nd=2):
    """NaN/inf/None -> None，避免 JSON 序列化失败"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def fetch_fred_series(series_id, retries=3):
    params = {"series_id": series_id, "api_key": FRED_API_KEY,
              "file_type": "json", "observation_start": OBS_START}
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"{series_id}: HTTP {r.status_code}")
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise RuntimeError(f"{series_id}: Fetch failed") from None
            time.sleep(2 ** attempt)
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)


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


def pct_rank(s, years=20):
    """最新值在过去 N 年中的分位数 (0-100)，仅用于展示"""
    s = s.dropna()
    if s.empty:
        return None
    w = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    return clean((w <= w.iloc[-1]).mean() * 100, 0)


def build_one(item):
    series_id, cfg = item
    base = {"series_id": series_id, "name": cfg["name"], "freq": cfg["freq"]}
    try:
        raw = fetch_fred_series(series_id)
        base["latest_date"] = raw["date"].iloc[-1].strftime("%Y-%m-%d")
        if cfg["type"] == "price_index":
            p = process_price_index(raw, cfg["freq"])
            last = p.iloc[-1]
            return {**base, "metric_type": "yoy_percent",
                    "level_or_yoy": clean(last["yoy"]),
                    "short_term": clean(last["ann_3m"]),
                    "acceleration": clean(last["acceleration"]),
                    "percentile": pct_rank(p["yoy"])}, raw
        p = process_level_series(raw, cfg["freq"])
        last = p.iloc[-1]
        return {**base, "metric_type": "absolute_level",
                "level_or_yoy": clean(last["value"]),
                "change_1m": clean(last["change_1m"]),
                "change_3m": clean(last["change_3m"]),
                "short_term": clean(last["change_1m"]),
                "acceleration": clean(last["acceleration"]),
                "percentile": pct_rank(p["value"])}, raw
    except Exception as e:
        return {**base, "error": str(e)}, None


def streak(diffs):
    """末尾连续同向的周数：正数=连续上升，负数=连续下降"""
    n, d = 0, 0
    for x in reversed(list(diffs)):
        sgn = 1 if x > 0 else -1 if x < 0 else 0
        if sgn == 0 or (d and sgn != d):
            break
        d, n = sgn, n + 1
    return n * d


def build_liquidity_trigger(raw):
    """SOFR-IORB 10 日均值 + 准备金连续变动周数 -> 触发状态"""
    try:
        sofr, iorb, res = raw["SOFR"], raw["IORB"], raw["WRBWFRBL"]
        m = pd.merge_asof(
            sofr[["date", "value"]], iorb[["date", "value"]],
            on="date", direction="backward", suffixes=("_s", "_i")).dropna()
        m["bp"] = (m["value_s"] - m["value_i"]) * 100
        avg10 = float(m["bp"].tail(10).mean())
        weeks = streak(res["value"].diff().dropna().tail(6))

        if avg10 > 5 and weeks <= -3:
            status = "escalation_confirmed"
        elif avg10 <= 0 and weeks >= 3:
            status = "invalidated"
        else:
            status = "neutral"

        card = {"series_id": "SOFR_IORB", "name": "SOFR − IORB Spread (derived)",
                "derived": True, "freq": "daily", "metric_type": "absolute_level",
                "latest_date": m["date"].iloc[-1].strftime("%Y-%m-%d"),
                "level_or_yoy": clean(m["bp"].iloc[-1], 1),
                "avg_10d": clean(avg10, 1),
                "short_term": None, "acceleration": None}
        trigger = {"spread_10d_avg_bp": clean(avg10, 1),
                   "reserves_streak_weeks": weeks,   # 负数=连续下降周数
                   "status": status}
        return card, trigger
    except Exception as e:
        print(f"[liquidity_trigger] failed: {type(e).__name__}: {e}")
        return None, None


def build_payload():
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(build_one, INDICATOR_MAP.items()))
    data = [r[0] for r in results]
    raw = {sid: r[1] for sid, r in zip(INDICATOR_MAP, results) if r[1] is not None}

    card, trigger = build_liquidity_trigger(raw)
    if card:
        data.append(card)
    return {
        "status": "success",
        "updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data": data,
        "triggers": {"liquidity_transmission": trigger},
    }


CACHE = {"ts": 0.0, "payload": None}
LOCK = threading.Lock()


def get_cached():
    with LOCK:
        if CACHE["payload"] and time.time() - CACHE["ts"] < CACHE_TTL:
            return CACHE["payload"]
        try:
            payload = build_payload()
            CACHE.update(ts=time.time(), payload=payload)
            return payload
        except Exception:
            if CACHE["payload"]:       # 刷新失败时回退到旧缓存
                return CACHE["payload"]
            raise


@app.on_event("startup")
def warm_cache():
    threading.Thread(target=get_cached, daemon=True).start()


@app.get("/api/minerva-macro")
def get_macro_data():
    return get_cached()