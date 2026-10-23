"""
ai_capital.py
从 SEC EDGAR（XBRL companyfacts）生成 AI Capital Cycle 数据草稿。

原则：
1. 自动抓取的数值状态一律为 "pending"，脚本永远不会写入 "verified"。
2. 已经 "verified" 的数值绝不覆盖；若新抓到的数与之不一致，只在该指标上
   加一个 "auto_check" 字段提醒人工复核。
3. 你手工填写的待核验数值（没有 "auto": true）同样不会被覆盖。
4. 抓不到的指标（云收入、尚未开始的租赁承诺等）保持原样，仍需手工录入。

用法（本地）：
    set SEC_USER_AGENT=Minerva Invest you@example.com
    python ai_capital.py --base ai-capital-data.json --out ai-capital-draft.json

Railway 环境变量：
    SEC_USER_AGENT   必填，SEC 要求声明身份，例如 "Minerva Invest you@example.com"
    BASE_JSON_URL    当前线上 ai-capital-data.json 的公开地址（草稿接口以它为底稿）
    DRAFT_TOKEN      可选，设置后访问草稿接口必须带 ?token=...
"""
import argparse
import copy
import hmac
import json
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone

import requests

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# 第一批：披露最规范的四家。ORCL / NVDA / AVGO 之后再加。
COMPANIES = {
    "MSFT": {"cik": 789019, "name_hint": "MICROSOFT", "fye_month": 6},
    "AMZN": {"cik": 1018724, "name_hint": "AMAZON", "fye_month": 12},
    "GOOGL": {"cik": 1652044, "name_hint": "ALPHABET", "fye_month": 12},
    "META": {"cik": 1326801, "name_hint": "META", "fye_month": 12},
}

# 标签按优先级排列，取第一个有数据的；实际用了哪个标签会写进 source_location
DURATION_METRICS = {
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "total_revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
}
INSTANT_METRICS = {
    "debt": ["LongTermDebt", "DebtLongtermAndShorttermCombinedAmount", "LongTermDebtNoncurrent"],
    "finance_lease_liabilities": ["FinanceLeaseLiability"],
    "operating_lease_liabilities": ["OperatingLeaseLiability"],
}
LEASE_COMPONENTS = {
    "finance_lease_liabilities": ("FinanceLeaseLiabilityCurrent", "FinanceLeaseLiabilityNoncurrent"),
    "operating_lease_liabilities": ("OperatingLeaseLiabilityCurrent", "OperatingLeaseLiabilityNoncurrent"),
}
AUTO_KEYS = set(DURATION_METRICS) | set(INSTANT_METRICS)

CAPEX_DEFINITION_TEXT = "SEC XBRL us-gaap:{tag} (cash flow statement, auto-fetched)"
DEFAULT_METRIC_KEYS = [
    "capex", "cloud_revenue", "operating_cash_flow", "debt",
    "finance_lease_liabilities", "operating_lease_liabilities", "leases_not_yet_commenced",
]

Q_MIN, Q_MAX = 75, 105      # 单季度持续天数范围
REL_TOL = 0.005             # 与已核验数值的相对容差（0.5%）


# ---------------------------------------------------------------- 抓取
def fetch_companyfacts(cik, name_hint, retries=3):
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC_USER_AGENT is not set (e.g. 'Minerva Invest you@example.com')")
    time.sleep(0.25)  # SEC 公平访问：不超过 10 次/秒
    url = FACTS_URL.format(cik=cik)
    data = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": ua}, timeout=60)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"CIK {cik}: HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"CIK {cik}: fetch failed ({type(e).__name__})") from None
            time.sleep(2 ** attempt + 1)
    entity = str(data.get("entityName", "")).upper()
    if name_hint not in entity:
        raise RuntimeError(f"CIK {cik}: entity mismatch ({entity!r}), expected '{name_hint}'")
    return data


# ---------------------------------------------------------------- 事实选取
def _d(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def _facts(cf, tag):
    node = cf.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return []
    return node.get("units", {}).get("USD", [])


def _dedupe(facts, instant):
    """同一 (start, end) 只保留最早提交的一份（首次披露版本），仅限 10-Q / 10-K"""
    best = {}
    for f in facts:
        if f.get("form") not in ("10-Q", "10-K"):
            continue
        if "end" not in f or "val" not in f or "filed" not in f:
            continue
        if not instant and "start" not in f:
            continue
        key = (f.get("start"), f["end"])
        cur = best.get(key)
        if cur is None or f["filed"] < cur["filed"]:
            best[key] = f
    return list(best.values())


def _dur(f):
    return (_d(f["end"]) - _d(f["start"])).days


def quarter_value(facts, end):
    """单季度数值。返回 (value, 用到的事实列表, 说明) 或 None。
    直接有 3 个月数据就用；否则用 YTD 相邻两期相减（Q2=6M−3M，Q3=9M−6M，Q4=全年−9M）。"""
    e = _d(end)
    cands = [f for f in facts if abs((_d(f["end"]) - e).days) <= 3]
    if not cands:
        return None
    direct = [f for f in cands if Q_MIN <= _dur(f) <= Q_MAX]
    if direct:
        f = min(direct, key=lambda x: x["filed"])
        return f["val"], [f], "3M"
    longer = [f for f in cands if _dur(f) > Q_MAX]
    if not longer:
        return None
    ytd = min(longer, key=_dur)
    s = _d(ytd["start"])
    prevs = [
        f for f in facts
        if abs((_d(f["start"]) - s).days) <= 3 and Q_MIN <= (e - _d(f["end"])).days <= Q_MAX
    ]
    if not prevs:
        return None
    p = min(prevs, key=lambda x: x["filed"])
    return ytd["val"] - p["val"], [ytd, p], f"YTD({_dur(ytd)}d) - YTD({_dur(p)}d)"


def instant_value(facts, end):
    e = _d(end)
    c = [f for f in facts if abs((_d(f["end"]) - e).days) <= 3]
    if not c:
        return None
    f = min(c, key=lambda x: x["filed"])
    return f["val"], [f]


def extract_metric(cf, metric, end):
    """返回 dict(value_usd, tag, how, accn, filed, warning) 或 None"""
    if metric in DURATION_METRICS:
        for tag in DURATION_METRICS[metric]:
            r = quarter_value(_dedupe(_facts(cf, tag), instant=False), end)
            if r:
                val, fs, how = r
                last = max(fs, key=lambda x: x["filed"])
                return {"value_usd": val, "tag": tag, "how": how,
                        "accn": last.get("accn"), "filed": last["filed"], "warning": None}
        return None

    if metric in INSTANT_METRICS:
        for tag in INSTANT_METRICS[metric]:
            r = instant_value(_dedupe(_facts(cf, tag), instant=True), end)
            if r:
                val, fs = r
                warn = None
                if tag == "LongTermDebtNoncurrent":
                    warn = "noncurrent portion only (excludes current maturities) - verify against filing"
                return {"value_usd": val, "tag": tag, "how": "instant",
                        "accn": fs[0].get("accn"), "filed": fs[0]["filed"], "warning": warn}
        comps = LEASE_COMPONENTS.get(metric)
        if comps:
            parts = []
            for tag in comps:
                r = instant_value(_dedupe(_facts(cf, tag), instant=True), end)
                if not r:
                    parts = []
                    break
                parts.append((tag, r))
            if parts:
                total = sum(p[1][0] for p in parts)
                fs = [p[1][1][0] for p in parts]
                last = max(fs, key=lambda x: x["filed"])
                return {"value_usd": total, "tag": "+".join(p[0] for p in parts),
                        "how": "instant (sum of components)", "accn": last.get("accn"),
                        "filed": last["filed"], "warning": None}
    return None


# ---------------------------------------------------------------- 季度与标签
def discover_quarter_ends(cf, fye_month, n):
    """从经营现金流事实里找出最近 n 个季度的截止日"""
    ends = set()
    for tag in DURATION_METRICS["operating_cash_flow"]:
        for f in _dedupe(_facts(cf, tag), instant=False):
            e = _d(f["end"])
            if e.day >= 28 and (e.month - fye_month) % 3 == 0:
                ends.add(e)
    return sorted(ends)[-n:]


def period_label(end, fye_month):
    if fye_month == 12:
        q = ((end.month - 1) // 3) + 1
        return f"{end.year} Q{q}"
    fy = end.year + (1 if end.month > fye_month else 0)
    q = ((end.month - fye_month - 1) % 12) // 3 + 1
    return f"FY{fy} Q{q}"


def offset_days(end):
    """与最近日历季度末的相差天数（带符号）"""
    cands = []
    for y in (end.year - 1, end.year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            cands.append(date(y, m, d))
    nearest = min(cands, key=lambda c: abs((end - c).days))
    return (end - nearest).days


def accession_index_url(cik, accn):
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{accn}-index.htm"


# ---------------------------------------------------------------- 合并
def _new_period(label, end_s, keys):
    return {
        "fiscal_period": label,
        "period_end_date": end_s,
        "offset_days_vs_calendar_quarter_end": None,
        "release_state": "released",
        "expected_release_date": None,
        "filing_date": None,
        "source_url": None,
        "capex_definition_used": None,
        "unit": "USD billion",
        "metrics": {k: {"value": None, "source_location": None, "status": "not_entered"} for k in keys},
    }


def merge_company(comp, cf, cfg, n_quarters, now_iso, report):
    ticker = comp.get("ticker")
    periods = comp.setdefault("periods", [])
    by_end = {p.get("period_end_date"): p for p in periods if p.get("period_end_date")}
    keys = list(periods[0]["metrics"].keys()) if periods and periods[0].get("metrics") else DEFAULT_METRIC_KEYS

    for e in discover_quarter_ends(cf, cfg["fye_month"], n_quarters):
        end_s = e.isoformat()
        p = by_end.get(end_s)
        if p is None:
            p = _new_period(period_label(e, cfg["fye_month"]), end_s, keys)
            periods.append(p)
            by_end[end_s] = p

        p["release_state"] = "released"
        if p.get("offset_days_vs_calendar_quarter_end") is None:
            p["offset_days_vs_calendar_quarter_end"] = offset_days(e)

        latest = None  # (filed, accn)
        for key, m in p.get("metrics", {}).items():
            if key not in AUTO_KEYS:
                if m.get("status") == "not_reported":
                    m["status"] = "not_entered"   # 财报已发布，但这一项只能手工录入
                continue
            res = extract_metric(cf, key, end_s)
            if res is None:
                if m.get("status") == "not_reported":
                    m["status"] = "not_entered"   # 财报已出但这一项抓不到，需手工录入
                continue

            value_b = round(res["value_usd"] / 1e9, 3)
            if latest is None or res["filed"] > latest[0]:
                latest = (res["filed"], res["accn"])

            status = m.get("status")
            existing = m.get("value")
            protected = status == "verified" or (existing is not None and not m.get("auto"))
            if protected:
                if existing is not None and abs(value_b - existing) > max(0.001, REL_TOL * abs(existing)):
                    m["auto_check"] = {
                        "value": value_b, "accession": res["accn"], "fetched_at": now_iso,
                        "note": "differs from existing value - please re-check against the filing",
                    }
                    report["conflicts"].append({
                        "ticker": ticker, "period": p["fiscal_period"], "metric": key,
                        "existing": existing, "auto": value_b, "accession": res["accn"],
                    })
                else:
                    m.pop("auto_check", None)
                continue

            m["value"] = value_b
            m["status"] = "pending"
            m["auto"] = True
            m["auto_fetched_at"] = now_iso
            m["accession"] = res["accn"]
            m["filed"] = res["filed"]
            m["source_location"] = (
                f"SEC XBRL us-gaap:{res['tag']} | {res['how']} | accn {res['accn']} | "
                "auto-fetched, check against the filing"
            )
            m.pop("auto_check", None)
            report["filled"] += 1
            if res["warning"]:
                report["warnings"].append(f"{ticker} {p['fiscal_period']} {key}: {res['warning']}")
            if key == "capex" and not p.get("capex_definition_used"):
                p["capex_definition_used"] = CAPEX_DEFINITION_TEXT.format(tag=res["tag"])

        if latest:
            if not p.get("filing_date"):
                p["filing_date"] = latest[0]
            if not p.get("source_url") and latest[1]:
                p["source_url"] = accession_index_url(cfg["cik"], latest[1])

    periods.sort(key=lambda p: p.get("period_end_date") or "", reverse=True)


def build_draft(base, fetcher=None, n_quarters=4, tickers=None, now=None):
    fetcher = fetcher or fetch_companyfacts
    draft = copy.deepcopy(base)
    now = now or datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {"filled": 0, "conflicts": [], "warnings": [], "errors": {}}

    for comp in draft.get("companies", []):
        tk = comp.get("ticker")
        cfg = COMPANIES.get(tk)
        if not cfg or (tickers and tk not in tickers):
            continue
        try:
            cf = fetcher(cfg["cik"], cfg["name_hint"])
            merge_company(comp, cf, cfg, n_quarters, now_iso, report)
        except Exception as e:  # 一家失败不影响其他家
            report["errors"][tk] = f"{type(e).__name__}: {e}"

    draft["auto_update"] = {
        "last_run_utc": now_iso,
        "source": "SEC EDGAR XBRL companyfacts",
        "covered_tickers": sorted(COMPANIES),
        "note": "Auto-fetched values are 'pending'. Only a human may set 'verified'.",
        "errors": report["errors"],
    }
    return draft, report


# ---------------------------------------------------------------- Railway 接口
try:
    from fastapi import APIRouter, HTTPException

    router = APIRouter()
    _CACHE = {"ts": 0.0, "payload": None}
    _LOCK = threading.Lock()
    CACHE_TTL = 6 * 3600

    @router.get("/api/ai-capital/draft")
    def ai_capital_draft(token: str = ""):
        required = os.environ.get("DRAFT_TOKEN", "")
        if required and not hmac.compare_digest(token, required):
            raise HTTPException(status_code=403, detail="forbidden")
        base_url = os.environ.get("BASE_JSON_URL", "").strip()
        if not base_url:
            raise HTTPException(status_code=500, detail="BASE_JSON_URL is not set")
        with _LOCK:
            if _CACHE["payload"] and time.time() - _CACHE["ts"] < CACHE_TTL:
                return _CACHE["payload"]
            try:
                r = requests.get(base_url, timeout=20)
                r.raise_for_status()
                base = r.json()
                draft, report = build_draft(base)
            except Exception as e:
                if _CACHE["payload"]:
                    return _CACHE["payload"]
                raise HTTPException(status_code=502, detail=f"draft failed: {type(e).__name__}")
            payload = {"report": report, "draft": draft}
            _CACHE.update(ts=time.time(), payload=payload)
            return payload
except ImportError:  # 本地命令行使用时不需要 fastapi
    router = None


# ---------------------------------------------------------------- 命令行
def _main():
    ap = argparse.ArgumentParser(description="Build AI Capital Cycle draft from SEC EDGAR")
    ap.add_argument("--base", required=True, help="existing ai-capital-data.json")
    ap.add_argument("--out", required=True, help="output draft path")
    ap.add_argument("--quarters", type=int, default=4)
    args = ap.parse_args()

    with open(args.base, encoding="utf-8") as f:
        base = json.load(f)
    draft, report = build_draft(base, n_quarters=args.quarters)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)

    print(f"filled (pending): {report['filled']}")
    for c in report["conflicts"]:
        print("CONFLICT:", c)
    for w in report["warnings"]:
        print("WARNING:", w)
    for tk, err in report["errors"].items():
        print("ERROR:", tk, err)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(_main())