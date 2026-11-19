"""
journal_ai.py
用 Claude API 做三件事：
  1. 翻译：中文原文缺少英文/法文翻译时自动补齐（中文仍是原文）
  2. CapEx 指引提取（H1）：从 SEC 8-K 业绩新闻稿、或你粘贴的电话会文字里提取，写入 guidance_log（只追加）
  3. Review 起草：有新数据、状态变化或预测判定时起草一条复盘，放进 review_drafts，由你批准后才发布

Railway 环境变量：
    ANTHROPIC_API_KEY   必填（不设置则以上功能全部跳过，其余自动化照常运行）
    CLAUDE_MODEL        可选，默认 claude-sonnet-5-5
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import requests

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"
GUIDANCE_START = "2026-07-01"   # 从这天之后的业绩新闻稿开始提取（含上一季，作为"此前指引"）


def model_name():
    return os.environ.get("CLAUDE_MODEL", "").strip() or DEFAULT_MODEL


def ai_enabled():
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _now_iso(now=None):
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- API 客户端
def claude_json(system, user, schema, max_tokens=4000):
    """调用 Claude，用结构化输出（output_config.format）保证返回符合 schema 的 JSON。
    注：Sonnet 5.5 不支持强制工具调用（tool_choice any/tool），所以用结构化输出。"""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    body = {
        "model": model_name(),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    r = requests.post(API_URL, json=body, timeout=180, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"Claude API HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    if data.get("stop_reason") in ("refusal", "max_tokens"):
        raise RuntimeError(f"Claude stopped: {data.get('stop_reason')}")
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return json.loads(text)


def _obj(props, required=None):
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


NUM_OR_NULL = {"anyOf": [{"type": "number"}, {"type": "null"}]}


# ---------------------------------------------------------------- 1. 翻译
TRANSLATE_SYSTEM = """You translate investment research notes written in Chinese by SAS Minerva into English and French.
Rules:
- Translate faithfully. Do not add, remove, soften or strengthen any claim. The Chinese text is authoritative.
- Keep every number, percentage, date, ticker and metric code (T2, T2b-1, T2b-2, H1-H3, P1-P3, PPI, CapEx, OCF, 10-Q, 10-K, XBRL) exactly as written.
- Terminology: 证伪 = falsification / réfutation; 基线 = baseline / référence; 置信度 = confidence / confiance;
  资本开支 = capital expenditure (CapEx) / dépenses d'investissement (capex); 经营现金流 = operating cash flow / flux de trésorerie opérationnel;
  PPI in French = IPP.
- Professional, concise register. French uses standard French typography (space before % and :)."""

TRANSLATE_SCHEMA = _obj({"items": {"type": "array", "items": _obj({
    "key": {"type": "string"}, "en": {"type": "string"}, "fr": {"type": "string"}})}})


def _numbers(s):
    """提取数字并统一格式：86,2 → 86.2（法文小数）；1,000 / 1 000 → 1000（千位分隔）"""
    s = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}\b)", "", s or "")
    out = []
    for t in re.findall(r"\d[\d,.]*\d|\d", s):
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", t):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
        out.append(t)
    return out


def numbers_preserved(src, dst):
    """原文里的每个数字都必须在译文里出现（防止翻译改动数字）"""
    have = _numbers(dst)
    for n in _numbers(src):
        if n in have:
            have.remove(n)
        else:
            return False
    return True


def missing_translations(doc):
    from journal_store import localized_text
    zh = localized_text(doc, "zh")["text"]
    keys = set(localized_text(doc, "en")["fallback_to_zh"]) | set(localized_text(doc, "fr")["fallback_to_zh"])
    return {k: zh[k] for k in sorted(keys) if zh.get(k)}


def translate_missing(doc, llm=None, now=None):
    llm = llm or claude_json
    rej = doc.setdefault("i18n_rejected", {})
    todo = {k: v for k, v in missing_translations(doc).items() if rej.get(k, 0) < 3}
    if not todo:
        return {"translated": [], "rejected": []}
    user = "Translate each item. Return one entry per key.\n\n" + json.dumps(
        [{"key": k, "zh": v} for k, v in todo.items()], ensure_ascii=False, indent=1)
    out = llm(TRANSLATE_SYSTEM, user, TRANSLATE_SCHEMA, max_tokens=8000)
    i18n = doc.setdefault("i18n", {})
    meta = doc.setdefault("i18n_meta", {})
    done, rejected = [], []
    for it in out.get("items", []):
        k = it.get("key")
        if k not in todo:
            continue
        if not (numbers_preserved(todo[k], it["en"]) and numbers_preserved(todo[k], it["fr"])):
            rejected.append(k)   # 数字对不上就不写；最多重试 3 次，之后请在管理页手工翻译
            rej[k] = rej.get(k, 0) + 1
            continue
        for lang in ("en", "fr"):
            i18n.setdefault(lang, {})
            if not i18n[lang].get(k):
                i18n[lang][k] = it[lang]
        meta[k] = {"by": "claude", "model": model_name(), "at": _now_iso(now)}
        done.append(k)
    return {"translated": done, "rejected": rejected}


# ---------------------------------------------------------------- 2. CapEx 指引提取
GUIDANCE_SYSTEM = """You extract capital expenditure (CapEx) guidance from company earnings materials for a research log.
Only report what the text explicitly says. Never infer or estimate a number that is not stated.
- A guidance statement is a forward-looking statement about future CapEx (a range, a point figure, a growth rate, or a qualitative direction such as "increase meaningfully").
- Historical CapEx figures for past periods are NOT guidance.
- period_covered: the period the guidance refers to, as stated (e.g. "FY2026", "calendar 2026", "Q4 2026", "FY2027").
- basis: "incl_finance_leases" if the text says the figure includes finance leases; "cash_capex" if it says purchases of property and equipment / cash capex; otherwise "unspecified".
- Amounts in USD billions. For a range give low and high; for a point give the same value for both.
- direction_stated: how the text itself characterises the change vs previous guidance ("raised", "maintained", "lowered", "new" if first time given, "unclear" if not said).
- reason_category: the reason the text gives for a change: "demand", "returns" (ROI / capital discipline), "supply_constraint", "timing_or_delay", "accounting", "not_stated", "other".
- quote: the shortest exact sentence fragment supporting the statement, at most 30 words.
If there is no forward-looking CapEx statement, return guidance_found = false and an empty list."""

GUIDANCE_SCHEMA = _obj({
    "guidance_found": {"type": "boolean"},
    "statements": {"type": "array", "items": _obj({
        "period_covered": {"type": "string"},
        "basis": {"type": "string", "enum": ["cash_capex", "incl_finance_leases", "unspecified"]},
        "guidance_type": {"type": "string", "enum": ["range", "point", "growth", "qualitative"]},
        "low_usd_b": NUM_OR_NULL, "high_usd_b": NUM_OR_NULL, "growth_pct": NUM_OR_NULL,
        "direction_stated": {"type": "string", "enum": ["raised", "maintained", "lowered", "new", "unclear"]},
        "reason_category": {"type": "string", "enum": ["demand", "returns", "supply_constraint", "timing_or_delay",
                                                       "accounting", "not_stated", "other"]},
        "quote": {"type": "string"}})},
    "notes": {"type": "string"},
})

CAPEX_KEYWORDS = re.compile(r"capital expenditure|capex|property and equipment|outlook|guidance|expect", re.I)


def capex_windows(text, width=1200, limit=30000):
    """只把提到 CapEx / 展望 的段落发给 Claude，减少 token"""
    spans = []
    for m in re.finditer(r"capital expenditure|capex", text, re.I):
        a, b = max(0, m.start() - width), min(len(text), m.end() + width)
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    out = "\n...\n".join(text[a:b] for a, b in spans)
    return out[:limit]


def html_to_text(html):
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&nbsp;|&#160;", " ", t)
    t = re.sub(r"&amp;", "&", t)
    t = re.sub(r"&[a-z#0-9]+;", " ", t)
    return re.sub(r"\s+", " ", t)


def extract_guidance(ticker, text, source, llm=None, now=None):
    """返回一条 guidance_log 记录"""
    llm = llm or claude_json
    window = capex_windows(text)
    entry = {"id": source["id"], "ticker": ticker, "source": {k: v for k, v in source.items() if k != "id"},
             "extracted_at": _now_iso(now), "status": "pending"}
    if not window:
        entry.update(guidance_found=False, statements=[], notes="no CapEx mention in document", model=None)
        return entry
    user = (f"Company: {ticker}\nDocument: {source.get('label', source.get('type'))} (date {source.get('date')})\n\n"
            f"Relevant excerpts:\n{window}")
    out = llm(GUIDANCE_SYSTEM, user, GUIDANCE_SCHEMA, max_tokens=3000)
    stmts = []
    for s in out.get("statements", []):
        if s.get("quote") and len(s["quote"].split()) > 40:
            s["quote"] = " ".join(s["quote"].split()[:40]) + "…"
        stmts.append(s)
    entry.update(guidance_found=bool(out.get("guidance_found")) and bool(stmts), statements=stmts,
                 notes=out.get("notes", ""), model=model_name())
    return entry


def list_earnings_releases(cik, since):
    """SEC 8-K，Item 2.02（经营业绩），since 之后提交的"""
    from journal import SUBMISSIONS_URL, _sec_get
    rec = _sec_get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
    out = []
    for form, accn, filed, items, doc in zip(rec["form"], rec["accessionNumber"], rec["filingDate"],
                                            rec.get("items", [""] * len(rec["form"])), rec["primaryDocument"]):
        if form == "8-K" and "2.02" in (items or "") and filed >= since:
            out.append({"accn": accn, "filed": filed, "primary": doc})
    return sorted(out, key=lambda x: x["filed"])


EX99_NAME = re.compile(r"ex(hibit)?[-_.]?99", re.I)


def find_ex99(index_html):
    """从 SEC 文件索引页按官方文件类型找 EX-99.1；返回文件名或 None"""
    best = None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", index_html, re.S | re.I):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S | re.I)
        typ = next((re.sub(r"<[^>]+>|\s", "", c) for c in cells if re.fullmatch(r"\s*EX-99[.\d]*\s*", re.sub(r"<[^>]+>", "", c))), None)
        href = re.search(r'href="([^"]+\.html?)"', row, re.I)
        if typ and href:
            name = href.group(1).rsplit("/", 1)[-1]
            if typ.upper() == "EX-99.1":
                return name
            best = best or name
    return best


def is_exhibit_url(url):
    return bool(url) and bool(EX99_NAME.search(url.rsplit("/", 1)[-1]))


def fetch_press_release(cik, accn, primary):
    """取 EX-99.1（新闻稿）。先看 SEC 索引页的文件类型，再按文件名，最后才用主文档"""
    from journal import _sec_get
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/"
    name = None
    try:
        name = find_ex99(_sec_get(base + f"{accn}-index.htm").text)
    except Exception:
        pass
    if not name:
        try:
            items = _sec_get(base + "index.json").json()["directory"]["item"]
            ex = sorted((i["name"] for i in items if re.search(r"\.html?$", i["name"], re.I) and EX99_NAME.search(i["name"])),
                        key=lambda n: (0 if re.search(r"99[-_.]?0?1(?!\d)", n) else 1, n))
            name = ex[0] if ex else None
        except Exception:
            pass
    name = name or primary
    return base + name, html_to_text(_sec_get(base + name).text)


def update_guidance(doc, companies, llm=None, list_releases=None, get_release=None, now=None):
    """对每家公司处理新的业绩新闻稿；已处理过的（按 id）跳过。
    早期版本读错文件（读了 8-K 封面而非 EX-99.1）的记录会被标为 superseded 并重新提取。"""
    llm = llm or claude_json
    list_releases = list_releases or list_earnings_releases
    get_release = get_release or fetch_press_release
    log = doc.setdefault("guidance_log", [])
    by_id = {e["id"]: e for e in log}
    added, superseded, errors = [], [], {}
    for ticker, cfg in companies.items():
        try:
            for rel in list_releases(cfg["cik"], GUIDANCE_START):
                gid = f"{ticker}-{rel['accn']}"
                old = by_id.get(gid)
                if old is not None:
                    redo = (old.get("status") != "superseded" and not is_exhibit_url(old.get("source", {}).get("url"))
                            and f"{gid}-r2" not in by_id)
                    if not redo:
                        continue
                    new_id = f"{gid}-r2"
                else:
                    new_id = gid
                if new_id in by_id:
                    continue
                url, text = get_release(cfg["cik"], rel["accn"], rel["primary"])
                if old is not None and not is_exhibit_url(url):
                    continue   # 重新取还是拿不到附件 → 保留原记录，不重复调用 API
                e = extract_guidance(ticker, text, {"id": new_id, "type": "8-K EX-99", "accn": rel["accn"],
                                                    "date": rel["filed"], "url": url,
                                                    "label": "earnings press release"}, llm=llm, now=now)
                if old is not None:
                    old["status"] = "superseded"
                    old["superseded_by"] = new_id
                    superseded.append(gid)
                log.append(e)
                by_id[new_id] = e
                added.append(new_id)
        except Exception as ex:
            errors[ticker] = f"{type(ex).__name__}: {ex}"
    return {"added": added, "superseded": superseded, "errors": errors}


def add_manual_guidance(doc, ticker, label, date_s, text, llm=None, now=None):
    """你粘贴的电话会文字 / CFO 发言 → 同样提取后追加"""
    llm = llm or claude_json
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    gid = f"{ticker}-manual-{h}"
    if any(e["id"] == gid for e in doc.get("guidance_log", [])):
        return None
    e = extract_guidance(ticker, text, {"id": gid, "type": "manual", "date": date_s, "label": label,
                                        "text_sha256_10": h}, llm=llm, now=now)
    doc.setdefault("guidance_log", []).append(e)
    return e


# ---------------------------------------------------------------- 3. Review 起草
REVIEW_SYSTEM = """You draft review entries for SAS Minerva's research journal. The analyst (Yanan) approves every draft before it is published.
Rules:
- Use ONLY facts in the provided context. Quote numbers exactly as given (percentages may be rounded to one decimal).
- Describe what changed and what it means for each hypothesis (H1/H2/H3) and prediction (P1-P3) strictly in terms of the stated conditions. Do not invent new thresholds.
- Neutral, factual research tone. No investment advice, no buy/sell language, no price targets, no forecasts beyond the journal's own frozen predictions.
- If evidence is mixed or incomplete, say so plainly. Never claim a hypothesis is proven.
- 2 to 5 sentences. Write Chinese first (zh), then faithful English (en) and French (fr) translations with identical content.
- basis: list the context fields you relied on (e.g. "snapshot.financials.T2b_1", "fred.PCU335313335313.yoy")."""

REVIEW_SCHEMA = _obj({"zh": {"type": "string"}, "en": {"type": "string"}, "fr": {"type": "string"},
                      "basis": {"type": "array", "items": {"type": "string"}}})


def _fin_brief(f):
    if not f:
        return None
    keys = ("calendar_quarter", "capex_aggregate", "capex_aggregate_yoy", "T2b_1", "T2b_2", "T2")
    out = {k: f.get(k) for k in keys}
    out["per_company_ocf_to_capex"] = {t: v.get("ocf_to_capex") for t, v in (f.get("per_company") or {}).items()}
    return out


def review_context(old, new, triggers):
    from journal_store import localized_text
    zh = localized_text(new, "zh")["text"]
    snaps = new.get("snapshots", [])
    fred = (snaps[-1].get("h3", {}).get("fred", {}) if snaps else {})
    ctx = {
        "thesis": zh.get("thesis"),
        "hypotheses": {h["id"]: {"name": zh.get(h["id"] + ".name"), "falsification": zh.get(h["id"] + ".falsification")}
                       for h in new.get("hypotheses", [])},
        "predictions": [{"id": p["id"], "statement": zh.get(p["id"] + ".statement"), "confidence": p["confidence"],
                         "due_date": p["due_date"], "result": p["result"]} for p in new.get("predictions", [])],
        "baseline": _fin_brief(new.get("baseline")),
        "status_before": {k: v.get("status") for k, v in (old.get("status_board") or {}).items() if k != "as_of"},
        "status_now": {k: v for k, v in (new.get("status_board") or {}).items()},
        "previous_snapshot": _fin_brief(snaps[-2].get("financials")) if len(snaps) > 1 else None,
        "latest_snapshot": _fin_brief(snaps[-1].get("financials")) if snaps else None,
        "fred": {sid: {"latest": v.get("latest_date"), "yoy": v.get("yoy"),
                       "yoy_history": v.get("yoy_history")} for sid, v in fred.items() if not v.get("error")},
        "new_guidance": [e for e in new.get("guidance_log", [])[len(old.get("guidance_log", [])):] if e.get("guidance_found")],
        "triggers": triggers,
    }
    return ctx


def review_triggers(old, new, report):
    t = []
    if report.get("snapshot") == "appended":
        t.append("new_snapshot")
    ob = {k: v.get("status") for k, v in (old.get("status_board") or {}).items() if k != "as_of"}
    nb = {k: v.get("status") for k, v in (new.get("status_board") or {}).items() if k != "as_of"}
    for h, s in nb.items():
        if ob.get(h) and ob.get(h) != s:
            t.append(f"status_change:{h}:{ob.get(h)}->{s}")
    if report.get("evaluated"):
        t.append("predictions_evaluated:" + ",".join(report["evaluated"]))
    if any(e.get("guidance_found") for e in new.get("guidance_log", [])[len(old.get("guidance_log", [])):]):
        t.append("new_guidance")
    return t


def draft_review(old, new, triggers, llm=None, now=None):
    llm = llm or claude_json
    ctx = review_context(old, new, triggers)
    out = llm(REVIEW_SYSTEM, "Context (JSON):\n" + json.dumps(ctx, ensure_ascii=False, indent=1),
              REVIEW_SCHEMA, max_tokens=3000)
    drafts = new.setdefault("review_drafts", [])
    d = {"id": f"D{len(drafts) + 1:03d}", "created_at": _now_iso(now), "triggers": triggers,
         "zh": out["zh"], "en": out["en"], "fr": out["fr"], "basis": out.get("basis", []),
         "model": model_name(), "status": "draft"}
    drafts.append(d)
    return d


def approve_draft(doc, draft_id, zh, en, fr, author, today):
    """批准：追加到 reviews（中文为原文）并写入英法翻译"""
    d = next((x for x in doc.get("review_drafts", []) if x["id"] == draft_id), None)
    if not d or d["status"] != "draft":
        raise ValueError("draft not found or already handled")
    idx = len(doc.setdefault("reviews", []))
    doc["reviews"].append({"date": today, "author": author, "text_zh": zh, "drafted_by": d["model"],
                           "draft_id": draft_id})
    for lang, txt in (("en", en), ("fr", fr)):
        doc.setdefault("i18n", {}).setdefault(lang, {})[f"review.{idx}"] = txt
    doc.setdefault("i18n_meta", {})[f"review.{idx}"] = {"by": "claude+approved", "model": d["model"], "at": today}
    d["status"] = "approved"
    d["approved_at"] = today
    return idx


def reject_draft(doc, draft_id, today):
    d = next((x for x in doc.get("review_drafts", []) if x["id"] == draft_id), None)
    if not d or d["status"] != "draft":
        raise ValueError("draft not found or already handled")
    d["status"] = "rejected"
    d["rejected_at"] = today