"""
journal.py
Minerva Research Journal 的自动化（第 2 步）：
  1. 校验预测的 frozen_hash —— 任何人改动过已冻结的预测都会被发现
  2. 基线（baseline）只写一次：值为 null 时才写入，已有值永不覆盖
  3. 生成 H3 物理约束数据：FRED 电力设备 PPI / 电价 + 10-Q/10-K 电力相关措辞统计
  4. 有新数据时追加一条快照（snapshot），旧快照永不修改

和 ai_capital.py 同样的工作方式：接口返回"草稿"，你确认后再上传 journal-001.json。

Railway 环境变量（新增一个）：
    JOURNAL_JSON_URL   线上 journal-001.json 的公开地址
    （复用：SEC_USER_AGENT、FRED_API_KEY、DRAFT_TOKEN）
"""
import copy
import hashlib
import hmac
import json
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone

import requests

from ai_capital import (
    COMPANIES, _d, _dedupe, _facts, discover_quarter_ends, fetch_companyfacts, quarter_value,
)

# ---------------------------------------------------------------- 配置
# 需求端 5 家。ORCL 只在这里加入，不影响 ai_capital.py 现有流程。
DEMAND = {**{k: COMPANIES[k] for k in ("MSFT", "AMZN", "GOOGL", "META")},
          "ORCL": {"cik": 1341439, "name_hint": "ORACLE", "fye_month": 5}}

TAGS = {
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    # T2 用"新发行债务"，不是债务余额
    "debt_issued": ["ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt",
                    "ProceedsFromIssuanceOfSeniorLongTermDebt", "ProceedsFromIssuanceOfUnsecuredDebt"],
}
MAX_ALIGN_DAYS = 46   # 公司季度末与日历季度末允许的偏差（ORCL 约 30 天）

# H3：FRED 序列。ID 首次部署后请在 failed 列表里确认，失败的会单独标出，不影响其他。
H3_FRED = {
    "PCU335311335311": "PPI: Electric Power & Specialty Transformer Mfg",
    "PCU335313335313": "PPI: Switchgear & Switchboard Apparatus Mfg",
    "APU000072610": "Average Price: Electricity per kWh (US city avg)",
}
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# 10-Q/10-K 里的电力约束措辞。刻意不用单独的 "power"（会命中 PowerPoint、purchasing power）
POWER_TERMS = ["electricity", "power capacity", "power availability", "power supply",
               "power constraint", "power constrained", "energy supply", "electrical grid",
               "power grid", "transformers", "grid interconnection"]

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{doc}"


# ---------------------------------------------------------------- 冻结校验
def _canon_hash(obj):
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


def verify_predictions(journal):
    """返回被改动过的预测编号列表（空列表 = 全部完好）"""
    bad = []
    for p in journal.get("predictions", []):
        body = {k: v for k, v in p.items() if k not in ("frozen_hash", "result")}
        if _canon_hash(body) != p.get("frozen_hash"):
            bad.append(p.get("id"))
    return bad


# ---------------------------------------------------------------- 日历季度对齐
def calendar_quarter_end(d):
    """d 所在日历季度的季末"""
    m = ((d.month - 1) // 3 + 1) * 3
    return date(d.year, m, 31 if m in (3, 12) else 30)


def prev_calendar_quarter_end(q):
    m, y = q.month - 3, q.year
    if m <= 0:
        m, y = m + 12, y - 1
    return date(y, m, 31 if m in (3, 12) else 30)


def latest_complete_calendar_quarter(as_of):
    """as_of 当天之前已结束的最近一个日历季度"""
    return prev_calendar_quarter_end(calendar_quarter_end(as_of))


def label_cq(q):
    return f"{q.year} CQ{(q.month - 1) // 3 + 1}"


# ---------------------------------------------------------------- 单家公司、单季度（时点数据）
def _metric_pit(cf, tags, end_s, as_of):
    """只用 as_of 当天或之前提交的事实 —— 保证基线可复现，不受之后重述影响"""
    for tag in tags:
        facts = [f for f in _dedupe(_facts(cf, tag), instant=False) if f["filed"] <= as_of.isoformat()]
        r = quarter_value(facts, end_s)
        if r:
            val, fs, how = r
            return {"value_b": round(val / 1e9, 3), "tag": tag, "how": how,
                    "accn": max(fs, key=lambda x: x["filed"]).get("accn")}
    return None


def company_quarter(cf, cfg, cal_q, as_of):
    ends = discover_quarter_ends(cf, cfg["fye_month"], 16)
    near = [e for e in ends if abs((e - cal_q).days) <= MAX_ALIGN_DAYS]
    if not near:
        return None
    e = min(near, key=lambda x: abs((x - cal_q).days))
    out = {"period_end": e.isoformat()}
    for k, tags in TAGS.items():
        out[k] = _metric_pit(cf, tags, e.isoformat(), as_of)
    if out["capex"] is None or out["ocf"] is None:
        return None
    if out["debt_issued"] is None:
        out["debt_tag_candidates"] = debt_tag_candidates(cf, e.isoformat(), as_of)
    return out


DEBT_TAG_RE = re.compile(r"^ProceedsFrom.*(Debt|Notes|Borrowing)")


def debt_tag_candidates(cf, end_s, as_of):
    """抓不到发债时，列出该公司本季度实际存在的候选标签，方便下次把正确标签加进 TAGS"""
    out = {}
    for tag in cf.get("facts", {}).get("us-gaap", {}):
        if DEBT_TAG_RE.match(tag) and tag not in TAGS["debt_issued"]:
            r = _metric_pit(cf, [tag], end_s, as_of)
            if r:
                out[tag] = r["value_b"]
    return out


def aggregate(per_co):
    """per_co: {ticker: company_quarter 结果 或 None}"""
    missing = [t for t, v in per_co.items() if v is None]
    ok = {t: v for t, v in per_co.items() if v is not None}
    if missing:
        return {"complete": False, "missing": missing}
    capex = sum(v["capex"]["value_b"] for v in ok.values())
    ocf = sum(v["ocf"]["value_b"] for v in ok.values())
    ratios = {t: v["ocf"]["value_b"] / v["capex"]["value_b"] for t, v in ok.items() if v["capex"]["value_b"]}
    debt_missing = [t for t, v in ok.items() if v["debt_issued"] is None]
    t2 = None if debt_missing or not capex else round(
        sum(v["debt_issued"]["value_b"] for v in ok.values()) / capex, 4)
    return {
        "complete": True, "missing": [],
        "capex_aggregate": round(capex, 3),
        "ocf_aggregate": round(ocf, 3),
        "T2b_1": round(ocf / capex, 4) if capex else None,          # OCF ÷ CapEx（按 CapEx 加权）
        "T2b_2": sum(1 for r in ratios.values() if r < 1.0),         # 覆盖率 < 100% 的公司数
        "T2": t2, "T2_missing_debt_tag": debt_missing,
        "per_company": {t: {"period_end": v["period_end"],
                            "capex_b": v["capex"]["value_b"], "ocf_b": v["ocf"]["value_b"],
                            "ocf_to_capex": round(ratios[t], 4) if t in ratios else None,
                            "debt_issued_b": v["debt_issued"]["value_b"] if v["debt_issued"] else None,
                            "accn": v["capex"]["accn"],
                            **({"debt_tag_candidates": v["debt_tag_candidates"]}
                               if "debt_tag_candidates" in v else {})} for t, v in ok.items()},
    }


def financial_block(cfs, cal_q, as_of):
    cur = aggregate({t: company_quarter(cfs[t], DEMAND[t], cal_q, as_of) if cfs.get(t) else None
                     for t in DEMAND})
    yq = date(cal_q.year - 1, cal_q.month, cal_q.day)
    prev = aggregate({t: company_quarter(cfs[t], DEMAND[t], yq, as_of) if cfs.get(t) else None
                      for t in DEMAND})
    if cur.get("complete") and prev.get("complete") and prev["capex_aggregate"]:
        cur["capex_aggregate_yoy"] = round(cur["capex_aggregate"] / prev["capex_aggregate"] - 1, 4)
    else:
        cur["capex_aggregate_yoy"] = None
    cur["calendar_quarter"] = label_cq(cal_q)
    return cur


# ---------------------------------------------------------------- H3：FRED
def fetch_fred(series_id):
    key = os.environ.get("FRED_API_KEY", "").strip()
    r = requests.get(FRED_URL, params={"series_id": series_id, "api_key": key, "file_type": "json",
                                       "observation_start": "2015-01-01"}, timeout=30)
    r.raise_for_status()
    obs = [(o["date"], float(o["value"])) for o in r.json()["observations"] if o["value"] not in (".", "")]
    if not obs:
        raise RuntimeError("no observations")
    return obs


def fred_block(fred_fetch, today):
    out, failed = {}, []
    for sid, name in H3_FRED.items():
        try:
            obs = [o for o in fred_fetch(sid) if o[0] <= today.isoformat()]
            d, v = obs[-1]

            def yoy_at(i):
                di, vi = obs[i]
                ld = _d(di)
                target = date(ld.year - 1, ld.month, min(ld.day, 28))
                past = [o for o in obs[:i] if _d(o[0]) <= target and (target - _d(o[0])).days <= 40]
                return round(vi / past[-1][1] - 1, 4) if past and past[-1][1] else None

            hist = [[obs[i][0], yoy_at(i)] for i in range(max(0, len(obs) - 6), len(obs))]
            out[sid] = {"name": name, "latest_date": d, "value": v, "yoy": hist[-1][1],
                        "yoy_history": hist}   # 最近 6 期同比，供 H3 证伪规则使用
        except Exception as e:
            failed.append(sid)
            out[sid] = {"name": name, "error": f"{type(e).__name__}"}
    return out, failed


# ---------------------------------------------------------------- H3：财报电力措辞
def _sec_get(url):
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC_USER_AGENT is not set")
    time.sleep(0.25)
    r = requests.get(url, headers={"User-Agent": ua}, timeout=60)
    r.raise_for_status()
    return r


def fetch_recent_filings(cik, n):
    rec = _sec_get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
    rows = [{"form": f, "accn": a, "filed": d, "doc": p, "report_date": rd}
            for f, a, d, p, rd in zip(rec["form"], rec["accessionNumber"], rec["filingDate"],
                                      rec["primaryDocument"], rec["reportDate"])
            if f in ("10-Q", "10-K")]
    return sorted(rows, key=lambda x: x["filed"], reverse=True)[:n]


def fetch_filing_text(cik, accn, doc):
    return _sec_get(ARCHIVE_URL.format(cik=cik, nodash=accn.replace("-", ""), doc=doc)).text


def count_terms(html):
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z#0-9]+;", " ", text).lower()
    words = len(text.split())
    counts = {t: len(re.findall(r"\b" + re.escape(t) + r"s?\b", text)) for t in POWER_TERMS}
    total = sum(counts.values())
    # 10-K 比 10-Q 长得多，所以同时给出"每万词密度"，比较时用密度
    return {"total": total, "per_10k_words": round(total / words * 1e4, 3) if words else None,
            "words": words, "by_term": {k: v for k, v in counts.items() if v}}


def filings_block(list_filings, get_text, n=3):
    out = {}
    for t, cfg in DEMAND.items():
        try:
            rows = []
            for f in list_filings(cfg["cik"], n):
                c = count_terms(get_text(cfg["cik"], f["accn"], f["doc"]))
                rows.append({"form": f["form"], "accn": f["accn"], "filed": f["filed"],
                             "report_date": f["report_date"], **c})
            out[t] = rows
        except Exception as e:
            out[t] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ---------------------------------------------------------------- 主流程
def build_journal_draft(journal, cf_fetcher=None, fred_fetch=None,
                        list_filings=None, get_text=None, now=None):
    cf_fetcher = cf_fetcher or fetch_companyfacts
    fred_fetch = fred_fetch or fetch_fred
    list_filings = list_filings or fetch_recent_filings
    get_text = get_text or fetch_filing_text
    now = now or datetime.now(timezone.utc)
    today = now.date()
    draft = copy.deepcopy(journal)
    report = {"tampered_predictions": verify_predictions(journal), "baseline": "unchanged",
              "snapshot": "none", "errors": {}}

    cfs = {}
    for t, cfg in DEMAND.items():
        try:
            cfs[t] = cf_fetcher(cfg["cik"], cfg["name_hint"])
        except Exception as e:
            report["errors"][t] = f"{type(e).__name__}: {e}"

    # --- 基线：只在全部为 null 时写一次
    b = draft.setdefault("baseline", {})
    if all(b.get(k) is None for k in ("T2", "T2b_1", "T2b_2", "capex_aggregate")):
        as_of = _d(b.get("as_of") or draft["created_at"])
        cq = latest_complete_calendar_quarter(as_of)
        fin = financial_block(cfs, cq, as_of)
        if fin.get("complete"):
            ratios = {t: v["ocf_to_capex"] for t, v in fin["per_company"].items()}
            b.update({
                "calendar_quarter": fin["calendar_quarter"],
                "capex_aggregate": fin["capex_aggregate"],
                "capex_aggregate_yoy": fin["capex_aggregate_yoy"],
                "T2b_1": fin["T2b_1"], "T2b_2": fin["T2b_2"], "T2": fin["T2"],
                "t2b_groups": {"high": sorted(t for t, r in ratios.items() if r is not None and r >= 1.0),
                               "low": sorted(t for t, r in ratios.items() if r is not None and r < 1.0)},
                "per_company": fin["per_company"],
                "written_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": "pending",   # 与 Monitor 一致：自动值待你核验
            })
            if fin["T2"] is None:
                b["T2_note"] = f"debt issuance tag not found for {fin['T2_missing_debt_tag']} - fill manually"
            if not b["t2b_groups"]["low"] or not b["t2b_groups"]["high"]:
                b["P2_note"] = "one T2b group is empty - P2 cannot be evaluated as defined"
            report["baseline"] = "written"
        else:
            report["baseline"] = f"not written - missing {fin.get('missing')}"

    # --- 快照：数据有更新才追加
    cq_now = latest_complete_calendar_quarter(today)
    fin_now = financial_block(cfs, cq_now, today)
    if not fin_now.get("complete"):   # 本季财报还没出齐 → 用上一季
        cq_now = prev_calendar_quarter_end(cq_now)
        fin_now = financial_block(cfs, cq_now, today)
    fred, fred_failed = fred_block(fred_fetch, today)
    filings = filings_block(list_filings, get_text)

    fred_key = ",".join(f"{k}:{v.get('latest_date')}" for k, v in sorted(fred.items()))
    filing_key = ",".join(f"{t}:{r[0]['accn']}" for t, r in sorted(filings.items())
                          if isinstance(r, list) and r)
    key = f"{fin_now.get('calendar_quarter')}|{fred_key}|{filing_key}"
    snaps = draft.setdefault("snapshots", [])
    filing_errors = [t for t, r in filings.items() if not isinstance(r, list)]
    if report["errors"] or filing_errors or not fin_now.get("complete"):
        # 快照只追加、永不删除，所以数据不完整时宁可不写
        report["snapshot"] = f"skipped - incomplete data ({sorted(set(report['errors']) | set(filing_errors))})"
    elif not snaps or snaps[-1].get("data_key") != key:
        snaps.append({
            "as_of": today.isoformat(), "data_key": key, "status": "pending",
            "financials": fin_now,
            "h3": {"fred": fred, "fred_failed": fred_failed, "filing_power_mentions": filings},
        })
        report["snapshot"] = "appended"
    report["fred_failed"] = fred_failed
    draft["auto_update"] = {"last_run_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "tampered_predictions": report["tampered_predictions"]}
    return draft, report


# ---------------------------------------------------------------- Railway 接口
try:
    from fastapi import APIRouter, HTTPException

    router = APIRouter()
    _CACHE = {"ts": 0.0, "payload": None}
    _LOCK = threading.Lock()
    CACHE_TTL = 6 * 3600

    @router.get("/api/journal/draft")
    def journal_draft(token: str = ""):
        required = os.environ.get("DRAFT_TOKEN", "")
        if required and not hmac.compare_digest(token, required):
            raise HTTPException(status_code=403, detail="forbidden")
        url = os.environ.get("JOURNAL_JSON_URL", "").strip()
        if not url:
            raise HTTPException(status_code=500, detail="JOURNAL_JSON_URL is not set")
        with _LOCK:
            if _CACHE["payload"] and time.time() - _CACHE["ts"] < CACHE_TTL:
                return _CACHE["payload"]
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                draft, report = build_journal_draft(r.json())
            except Exception as e:
                if _CACHE["payload"]:
                    return _CACHE["payload"]
                raise HTTPException(status_code=502, detail=f"journal draft failed: {type(e).__name__}")
            payload = {"report": report, "draft": draft}
            _CACHE.update(ts=time.time(), payload=payload)
            return payload
except ImportError:
    router = None
