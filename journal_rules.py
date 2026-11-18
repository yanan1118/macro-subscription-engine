"""
journal_rules.py
规则引擎：
  1. 证伪检查（H1/H2/H3）——每次运行都计算，结果写入 status_board，并随新快照一起冻结
  2. 预测到期判定（P1/P2/P3）——到期后自动计算一次，写入 result；
     最终分类（✅/❌/⏱/⚪）由你人工确认，程序只给 auto_outcome 建议
"""
import csv
import io
from datetime import datetime, timezone

import requests

from ai_capital import _d, fetch_companyfacts
from journal import (DEMAND, financial_block, latest_complete_calendar_quarter,
                     prev_calendar_quarter_end)


# ---------------------------------------------------------------- 证伪
def _by_quarter(doc):
    """每个日历季度取最后一条快照的财务数据，按时间排序"""
    q = {}
    for s in doc.get("snapshots", []):
        f = s.get("financials") or {}
        if f.get("complete") and f.get("calendar_quarter"):
            q[f["calendar_quarter"]] = f
    return [q[k] for k in sorted(q)]


def check_h2(doc):
    """H2 证伪：T2b-1 连续 2 季 ≥ 100%，且最新 T2 < 40%"""
    qs = _by_quarter(doc)
    if len(qs) < 2:
        return {"status": "insufficient_data", "detail": f"{len(qs)} quarter(s) available, need 2"}
    last2 = qs[-2:]
    t2b_ok = all(f.get("T2b_1") is not None and f["T2b_1"] >= 1.0 for f in last2)
    detail = {"quarters": [f["calendar_quarter"] for f in last2],
              "T2b_1": [f.get("T2b_1") for f in last2], "T2": last2[-1].get("T2")}
    if not t2b_ok:
        return {"status": "not_triggered", "detail": detail}
    if last2[-1].get("T2") is None:
        return {"status": "partial", "detail": {**detail, "note": "T2b condition met, T2 missing"}}
    return {"status": "triggered" if last2[-1]["T2"] < 0.40 else "not_triggered", "detail": detail}


H3_SERIES = ("PCU335311335311", "PCU335313335313")   # 变压器 PPI、开关设备 PPI


def check_h3(doc, months=3):
    """H3 证伪：变压器 PPI 与开关设备 PPI 同比均 ≤ 0，且连续 3 个月"""
    snaps = doc.get("snapshots", [])
    if not snaps:
        return {"status": "insufficient_data", "detail": "no snapshot"}
    fred = snaps[-1].get("h3", {}).get("fred", {})
    detail, all_neg = {}, True
    for sid in H3_SERIES:
        hist = [y for _, y in (fred.get(sid, {}).get("yoy_history") or []) if y is not None]
        if len(hist) < months:
            return {"status": "insufficient_data", "detail": f"{sid}: {len(hist)} month(s) of YoY"}
        detail[sid] = hist[-months:]
        all_neg = all_neg and all(y <= 0 for y in hist[-months:])
    return {"status": "triggered" if all_neg else "not_triggered", "detail": detail}


def check_h1(doc):
    # CapEx 指引不在 XBRL 里，需要从财报新闻稿 / 电话会提取（后续用 Claude API 做）
    return {"status": "manual", "detail": "guidance extraction not automated yet"}


def status_board(doc, today):
    return {"as_of": today.isoformat(), "H1": check_h1(doc), "H2": check_h2(doc), "H3": check_h3(doc)}


# ---------------------------------------------------------------- 预测判定
def fetch_close(ticker, on_or_before):
    """收盘价（取该日或之前最近一个交易日）。数据源 stooq；到期前会再测试一次。"""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    rows = [row for row in csv.DictReader(io.StringIO(r.text))
            if row.get("Date") and row["Date"] <= on_or_before.isoformat() and row.get("Close")]
    if not rows:
        raise RuntimeError(f"{ticker}: no price on or before {on_or_before}")
    return float(rows[-1]["Close"])


def _fin_as_of(cfs, as_of):
    cq = latest_complete_calendar_quarter(as_of)
    f = financial_block(cfs, cq, as_of)
    if not f.get("complete"):
        f = financial_block(cfs, prev_calendar_quarter_end(cq), as_of)
    return f


def evaluate_prediction(p, doc, cfs, price_fetch):
    due = _d(p["due_date"])
    m = p["metric"]
    if m == "capex_aggregate_yoy":
        f = _fin_as_of(cfs, due)
        v = f.get("capex_aggregate_yoy")
        if v is None:
            raise RuntimeError(f"capex yoy unavailable ({f.get('missing')})")
        return v, v > p["rule"]["value"], {"calendar_quarter": f["calendar_quarter"]}
    if m == "T2b_1":
        f = _fin_as_of(cfs, due)
        base = doc["baseline"]["T2b_1"]
        v = f.get("T2b_1")
        if v is None or base is None:
            raise RuntimeError("T2b_1 or baseline unavailable")
        return v, v < base, {"calendar_quarter": f["calendar_quarter"], "baseline": base}
    if m == "relative_return_t2b_groups":
        g = doc["baseline"]["t2b_groups"]
        start = _d(p["created_at"])
        rets = {t: price_fetch(t, due) / price_fetch(t, start) - 1 for t in g["high"] + g["low"]}
        hi = sum(rets[t] for t in g["high"]) / len(g["high"])
        lo = sum(rets[t] for t in g["low"]) / len(g["low"])
        v = round(hi - lo, 4)
        return v, v > p["rule"]["value"], {"returns": {t: round(x, 4) for t, x in rets.items()}}
    raise RuntimeError(f"unknown metric {m}")


def evaluate_due_predictions(doc, today, cf_fetcher=None, price_fetch=None, now=None):
    """只处理已到期且仍为 pending 的预测；只写 result，不碰冻结字段"""
    due = [p for p in doc.get("predictions", [])
           if p["result"]["status"] == "pending" and _d(p["due_date"]) <= today]
    if not due:
        return []
    cf_fetcher = cf_fetcher or fetch_companyfacts
    price_fetch = price_fetch or fetch_close
    now = now or datetime.now(timezone.utc)
    cfs = {}
    for t, cfg in DEMAND.items():
        try:
            cfs[t] = cf_fetcher(cfg["cik"], cfg["name_hint"])
        except Exception:
            pass
    done = []
    for p in due:
        try:
            v, ok, extra = evaluate_prediction(p, doc, cfs, price_fetch)
        except Exception as e:
            p["result"]["last_error"] = f"{type(e).__name__}: {e}"   # 下次运行重试
            continue
        p["result"].update({
            "status": "evaluated",
            "evaluated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observed_value": v,
            "auto_outcome": "correct" if ok else "incorrect",
            "evidence": extra,
            "category": None,   # ✅ / ❌ / ⏱ / ⚪ —— 由你人工确认
        })
        p["result"].pop("last_error", None)
        done.append(p["id"])
    return done