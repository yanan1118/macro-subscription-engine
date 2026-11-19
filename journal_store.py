"""
journal_store.py
Journal 的持久存储（MySQL）、写入守卫、定时运行、三语输出、管理页。

Railway 环境变量：
    MYSQL_URL          Railway MySQL 提供的连接串（在后端服务里引用 ${{MySQL.MYSQL_URL}}）
    DRAFT_TOKEN        管理页口令（必填，否则管理页拒绝访问）
    JOURNAL_JSON_URL   仅首次导入时使用
    JOURNAL_RUN_HOURS  可选，自动运行间隔，默认 24 小时
requirements.txt 需新增：sqlalchemy、pymysql
"""
import copy
import hmac
import html
import json
import os
import threading
import time
from datetime import datetime, timezone
from urllib.parse import parse_qs

import requests
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, select
from sqlalchemy.dialects.mysql import LONGTEXT

from journal import build_journal_draft, verify_predictions
from journal_rules import evaluate_due_predictions, status_board
import journal_ai as AI
from journal import DEMAND

LANGS = ("zh", "en", "fr")

# ---------------------------------------------------------------- 存储
_meta = MetaData()
_BIG = Text().with_variant(LONGTEXT(), "mysql")
docs_t = Table("journal_docs", _meta,
               Column("jid", String(32), primary_key=True),
               Column("doc", _BIG, nullable=False),
               Column("updated_at", String(32), nullable=False))
versions_t = Table("journal_versions", _meta,      # 每次有实质变化都存一份完整历史版本
                   Column("id", Integer, primary_key=True, autoincrement=True),
                   Column("jid", String(32), nullable=False, index=True),
                   Column("doc", _BIG, nullable=False),
                   Column("saved_at", String(32), nullable=False),
                   Column("source", String(32), nullable=False))


def _db_url():
    url = (os.environ.get("JOURNAL_DB_URL") or os.environ.get("MYSQL_URL") or "").strip()
    if url.startswith("mysql://"):
        url = "mysql+pymysql://" + url[len("mysql://"):]
    return url


class Store:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True, pool_recycle=1800)
        _meta.create_all(self.engine)

    def get(self, jid):
        with self.engine.connect() as c:
            row = c.execute(select(docs_t.c.doc).where(docs_t.c.jid == jid)).first()
        return json.loads(row[0]) if row else None

    def list_ids(self):
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(select(docs_t.c.jid).order_by(docs_t.c.jid))]

    def save(self, jid, doc, source, version=True):
        s = json.dumps(doc, ensure_ascii=False)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.engine.begin() as c:
            if c.execute(select(docs_t.c.jid).where(docs_t.c.jid == jid)).first():
                c.execute(docs_t.update().where(docs_t.c.jid == jid).values(doc=s, updated_at=now))
            else:
                c.execute(docs_t.insert().values(jid=jid, doc=s, updated_at=now))
            if version:
                c.execute(versions_t.insert().values(jid=jid, doc=s, saved_at=now, source=source))


_STORE = {"obj": None}


def get_store():
    if _STORE["obj"] is None:
        url = _db_url()
        if not url:
            return None
        _STORE["obj"] = Store(url)
    return _STORE["obj"]


# ---------------------------------------------------------------- 写入守卫
def _locked_leaves(old, new, path, errs, skip=()):
    """old 里所有非空的值，在 new 里必须一模一样（空值可以填）"""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            errs.append(f"{path}: structure changed")
            return
        for k, v in old.items():
            if k in skip:
                continue
            _locked_leaves(v, new.get(k), f"{path}.{k}", errs)
    elif isinstance(old, list):
        if old and old != new:
            errs.append(f"{path}: locked list changed")
    elif old is not None and old != new:
        errs.append(f"{path}: locked value changed ({old!r} -> {new!r})")


def validate_update(old, new):
    """返回错误列表；空列表才允许保存"""
    errs = [f"prediction {p} hash mismatch" for p in verify_predictions(new)]
    op = {p["id"]: p for p in old.get("predictions", [])}
    np_ = {p["id"]: p for p in new.get("predictions", [])}
    if set(op) - set(np_):
        errs.append(f"predictions removed: {sorted(set(op) - set(np_))}")
    for pid, p in op.items():
        q = np_.get(pid)
        if not q:
            continue
        if {k: v for k, v in p.items() if k != "result"} != {k: v for k, v in q.items() if k != "result"}:
            errs.append(f"prediction {pid}: frozen fields changed")
        if p["result"].get("status") == "evaluated":   # 判定后只能补分类和备注
            a = {k: v for k, v in p["result"].items() if k not in ("category", "note")}
            b = {k: v for k, v in q["result"].items() if k not in ("category", "note")}
            if a != b:
                errs.append(f"prediction {pid}: evaluated result changed")
    for key in ("snapshots", "reviews"):
        o, n = old.get(key, []), new.get(key, [])
        if n[:len(o)] != o:
            errs.append(f"{key}: existing entries were modified or removed (append only)")
    og, ng = old.get("guidance_log", []), new.get("guidance_log", [])
    strip = lambda e: {k: v for k, v in e.items() if k not in ("status", "verified_at", "superseded_by")}
    if len(ng) < len(og) or [strip(e) for e in ng[:len(og)]] != [strip(e) for e in og]:
        errs.append("guidance_log: existing entries were modified or removed (append only)")
    _locked_leaves(old.get("baseline", {}), new.get("baseline", {}), "baseline", errs,
                   skip=("status",))   # status 允许 pending -> verified
    return errs


# ---------------------------------------------------------------- 三语
def localized_text(doc, lang):
    """中文是原文；英文/法文从 i18n 取，缺失时回退中文并标注"""
    tr = (doc.get("i18n") or {}).get(lang, {}) if lang != "zh" else {}
    src = {"thesis": doc.get("thesis_zh")}
    for h in doc.get("hypotheses", []):
        src[f"{h['id']}.name"] = h.get("name_zh")
        src[f"{h['id']}.falsification"] = (h.get("falsification") or {}).get("rule_zh")
    for p in doc.get("predictions", []):
        src[f"{p['id']}.statement"] = p.get("statement_zh")
        src[f"{p['id']}.measure"] = (p.get("rule") or {}).get("measure")
    for i, r in enumerate(doc.get("reviews", [])):
        src[f"review.{i}"] = r.get("text_zh")
    out, fallback = {}, []
    for k, zh in src.items():
        if zh is None:
            continue
        if lang == "zh" or tr.get(k):
            out[k] = tr.get(k, zh) if lang != "zh" else zh
        else:
            out[k] = zh
            fallback.append(k)
    return {"lang": lang, "text": out, "fallback_to_zh": fallback}


# ---------------------------------------------------------------- 自动运行
def _material(doc):
    d = copy.deepcopy(doc)
    d.pop("auto_update", None)
    (d.get("status_board") or {}).pop("as_of", None)
    return d


def run_once(store, jid, now=None, **fetchers):
    now = now or datetime.now(timezone.utc)
    today = now.date()
    old = store.get(jid)
    if old is None:
        return {"error": f"{jid} not found"}
    draft, report = build_journal_draft(
        old, cf_fetcher=fetchers.get("cf_fetcher"), fred_fetch=fetchers.get("fred_fetch"),
        list_filings=fetchers.get("list_filings"), get_text=fetchers.get("get_text"), now=now)
    llm = fetchers.get("llm") or (AI.claude_json if AI.ai_enabled() else None)
    if llm:
        try:
            report["guidance"] = AI.update_guidance(
                draft, DEMAND, llm=llm, list_releases=fetchers.get("list_releases"),
                get_release=fetchers.get("get_release"), now=now)
        except Exception as e:
            report["guidance"] = {"error": f"{type(e).__name__}: {e}"}
    board = status_board(draft, today, fred=report.pop("fred_current", None))
    draft["status_board"] = board
    if report["snapshot"] == "appended":
        draft["snapshots"][-1]["status_board"] = board        # 随快照冻结
    report["evaluated"] = evaluate_due_predictions(
        draft, today, cf_fetcher=fetchers.get("cf_fetcher"),
        price_fetch=fetchers.get("price_fetch"), now=now)
    if llm:
        try:
            report["translation"] = AI.translate_missing(draft, llm=llm, now=now)
        except Exception as e:
            report["translation"] = {"error": f"{type(e).__name__}: {e}"}
        trig = AI.review_triggers(old, draft, report)
        if trig:
            try:
                report["review_draft"] = AI.draft_review(old, draft, trig, llm=llm, now=now)["id"]
                draft.pop("review_pending_triggers", None)
            except Exception as e:
                report["review_draft"] = {"error": f"{type(e).__name__}: {e}", "will_retry": True}
                draft["review_pending_triggers"] = trig   # 下次运行重试
    else:
        report["ai"] = "disabled (ANTHROPIC_API_KEY not set)"
    errs = validate_update(old, draft)
    if errs:   # 程序自己的改动也要过守卫；不通过就不保存
        report["saved"] = False
        report["guard_errors"] = errs
        return report
    changed = _material(old) != _material(draft)
    store.save(jid, draft, "auto", version=changed)
    report["saved"] = True
    report["new_version"] = changed
    report["status_board"] = board
    return report


_LAST = {"reports": {}, "ran_at": None}


def run_all():
    store = get_store()
    if not store:
        return {"error": "database not configured"}
    out = {}
    for jid in store.list_ids():
        try:
            out[jid] = run_once(store, jid)
        except Exception as e:
            out[jid] = {"error": f"{type(e).__name__}: {e}"}
    _LAST.update(reports=out, ran_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    print("[journal] run:", json.dumps(out, ensure_ascii=False)[:2000], flush=True)
    return out


_RUN_LOCK = threading.Lock()


def _locked_run_all():
    if not _RUN_LOCK.acquire(blocking=False):
        return {"error": "a run is already in progress"}
    try:
        return run_all()
    finally:
        _RUN_LOCK.release()


def start_scheduler():
    hours = float(os.environ.get("JOURNAL_RUN_HOURS", "24"))

    def loop():
        time.sleep(90)   # 等服务启动完成
        while True:
            try:
                _locked_run_all()
            except Exception as e:
                print(f"[journal] scheduler error: {type(e).__name__}: {e}", flush=True)
            time.sleep(hours * 3600)

    if get_store():
        threading.Thread(target=loop, daemon=True).start()
    else:
        print("[journal] MYSQL_URL not set - scheduler disabled", flush=True)


# ---------------------------------------------------------------- 接口
try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import HTMLResponse

    router = APIRouter()

    def _check_token(token):
        required = os.environ.get("DRAFT_TOKEN", "")
        if not required or not hmac.compare_digest(token or "", required):
            raise HTTPException(status_code=403, detail="forbidden")

    def _need_store():
        s = get_store()
        if not s:
            raise HTTPException(status_code=503, detail="database not configured")
        return s

    # ---- 公开接口（前端读取）
    @router.get("/api/journal")
    def journal_list(lang: str = "zh"):
        s = _need_store()
        lang = lang if lang in LANGS else "zh"
        items = []
        for jid in s.list_ids():
            d = s.get(jid)
            items.append({"journal_id": jid, "slug": d.get("slug"), "created_at": d.get("created_at"),
                          "status": d.get("status"),
                          "thesis": localized_text(d, lang)["text"].get("thesis")})
        return {"lang": lang, "journals": items}

    @router.get("/api/journal/{jid}")
    def journal_get(jid: str, lang: str = "zh"):
        d = _need_store().get(jid)
        if d is None:
            raise HTTPException(status_code=404, detail="not found")
        lang = lang if lang in LANGS else "zh"
        return {"journal": d, **localized_text(d, lang)}

    # ---- 管理接口（需要 token）
    @router.get("/admin/journal/seed")
    def journal_seed(token: str = "", jid: str = "J-001"):
        _check_token(token)
        s = _need_store()
        if s.get(jid) is not None:
            return {"seeded": False, "reason": f"{jid} already exists - use the edit page"}
        url = os.environ.get("JOURNAL_JSON_URL", "").strip()
        if not url:
            raise HTTPException(status_code=500, detail="JOURNAL_JSON_URL is not set")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        doc = r.json()
        bad = verify_predictions(doc)
        if bad:
            raise HTTPException(status_code=400, detail=f"prediction hash mismatch: {bad}")
        if doc.get("journal_id") != jid:
            raise HTTPException(status_code=400, detail=f"journal_id in file is {doc.get('journal_id')}")
        s.save(jid, doc, "seed")
        return {"seeded": True, "jid": jid}

    @router.get("/admin/journal/run")
    def journal_run(token: str = ""):
        _check_token(token)
        _need_store()
        return _locked_run_all()

    @router.get("/admin/journal/status")
    def journal_status(token: str = ""):
        _check_token(token)
        return _LAST

    _PAGE = """<!doctype html><meta charset="utf-8"><title>Journal admin</title>
<style>body{{font:14px sans-serif;max-width:1100px;margin:24px auto}}textarea{{width:100%;height:70vh;font:12px monospace}}
.err{{color:#b00;white-space:pre-wrap}}.ok{{color:#070}}</style>
<h2>Journal {jid}</h2><p>只能追加 reviews、填写空值、把 baseline.status 改为 verified、给已判定的预测填 category / note。
其他已有内容被改动会被拒绝。</p>{msg}
<form method="post"><input type="hidden" name="token" value="{token}"><input type="hidden" name="jid" value="{jid}">
<textarea name="doc">{doc}</textarea><p><button type="submit">保存 / Save</button></p></form>"""

    @router.get("/admin/journal/edit", response_class=HTMLResponse)
    def journal_edit_page(token: str = "", jid: str = "J-001"):
        _check_token(token)
        d = _need_store().get(jid)
        if d is None:
            raise HTTPException(status_code=404, detail="not found")
        return _PAGE.format(jid=html.escape(jid), token=html.escape(token), msg="",
                            doc=html.escape(json.dumps(d, ensure_ascii=False, indent=2)))

    @router.post("/admin/journal/edit", response_class=HTMLResponse)
    async def journal_edit_save(request: Request):
        form = parse_qs((await request.body()).decode("utf-8"))
        token = form.get("token", [""])[0]
        jid = form.get("jid", ["J-001"])[0]
        raw = form.get("doc", [""])[0]
        _check_token(token)
        s = _need_store()
        old = s.get(jid)
        if old is None:
            raise HTTPException(status_code=404, detail="not found")
        try:
            new = json.loads(raw)
            errs = validate_update(old, new)
        except json.JSONDecodeError as e:
            errs = [f"JSON error: {e}"]
            new = None
        if errs:
            msg = '<p class="err">未保存 / Not saved:\n' + html.escape("\n".join(errs)) + "</p>"
            return _PAGE.format(jid=html.escape(jid), token=html.escape(token), msg=msg, doc=html.escape(raw))
        s.save(jid, new, "manual")
        return _PAGE.format(jid=html.escape(jid), token=html.escape(token), msg='<p class="ok">已保存 / Saved</p>',
                            doc=html.escape(json.dumps(new, ensure_ascii=False, indent=2)))

    # ---- AI 工作台：批准 Review 草稿、核验 CapEx 指引、粘贴电话会文字
    _AI_CSS = """<style>body{font:14px -apple-system,sans-serif;max-width:1100px;margin:24px auto;color:#24292e;padding:0 16px}
h2{border-bottom:2px solid #2c3e50;padding-bottom:6px;margin-top:34px}.box{border:1px solid #d0d7de;border-radius:8px;padding:14px;margin:12px 0}
textarea{width:100%;font:13px sans-serif;box-sizing:border-box}label{font-size:12px;color:#57606a;display:block;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid #eaecef;padding:6px;text-align:left;vertical-align:top}
.muted{color:#6a737d;font-size:12px}.ok{color:#1a7f37}.err{color:#cf222e;white-space:pre-wrap}button{margin:8px 8px 0 0;padding:6px 14px}
.tag{font-size:11px;background:#eaeef2;border-radius:10px;padding:1px 8px}</style>"""

    def _ai_page(jid, token, doc, msg=""):
        e = html.escape
        hidden = f'<input type="hidden" name="token" value="{e(token)}"><input type="hidden" name="jid" value="{e(jid)}">'
        o = [f"<!doctype html><meta charset='utf-8'><title>Journal AI {e(jid)}</title>{_AI_CSS}",
             f"<h1>Journal {e(jid)} · AI 工作台</h1>{msg}",
             f"<p class='muted'>模型：{e(AI.model_name())} · API：{'已启用' if AI.ai_enabled() else '未设置 ANTHROPIC_API_KEY'}</p>"]
        drafts = [d for d in doc.get("review_drafts", []) if d.get("status") == "draft"]
        o.append(f"<h2>Review 草稿（{len(drafts)} 条待处理）</h2>")
        if not drafts:
            o.append("<p class='muted'>暂无待处理草稿。有新数据、状态变化或预测判定时会自动生成。</p>")
        for d in drafts:
            o.append(f"""<form method="post" class="box">{hidden}<input type="hidden" name="draft_id" value="{e(d['id'])}">
<b>{e(d['id'])}</b> <span class="muted">{e(d['created_at'])} · 触发：{e(', '.join(d.get('triggers', [])))}</span>
<label>中文（原文，发布时以此为准）</label><textarea name="zh" rows="4">{e(d['zh'])}</textarea>
<label>English</label><textarea name="en" rows="4">{e(d['en'])}</textarea>
<label>Français</label><textarea name="fr" rows="4">{e(d['fr'])}</textarea>
<p class="muted">依据：{e(', '.join(d.get('basis', [])))}</p>
<button name="action" value="approve">批准并发布</button><button name="action" value="reject">拒绝</button></form>""")
        log = doc.get("guidance_log", [])
        o.append(f"<h2>CapEx 指引记录（H1，共 {len(log)} 份文件）</h2><table><tr><th>公司</th><th>来源</th><th>提取结果</th><th>状态</th></tr>")
        for g in reversed(log):
            src = g.get("source", {})
            link = f"<a href='{e(src['url'])}' target='_blank'>{e(src.get('label', ''))}</a>" if src.get("url") else e(src.get("label", ""))
            if g.get("guidance_found"):
                rows = []
                for st in g.get("statements", []):
                    num = (f"${st['low_usd_b']}–{st['high_usd_b']}B" if st.get("low_usd_b") is not None
                           else (f"{st['growth_pct']}%" if st.get("growth_pct") is not None else st.get("guidance_type")))
                    rows.append(f"<b>{e(st.get('period_covered'))}</b> {e(num)} · {e(st.get('basis'))} · "
                                f"{e(st.get('direction_stated'))} / {e(st.get('reason_category'))}<br>"
                                f"<span class='muted'>“{e(st.get('quote'))}”</span>")
                res = "<br>".join(rows)
            else:
                res = f"<span class='muted'>未发现指引 · {e(g.get('notes', ''))}</span>"
            btn = (f"<form method='post' style='margin:0'>{hidden}<input type='hidden' name='gid' value='{e(g['id'])}'>"
                   f"<button name='action' value='verify_guidance'>核验无误</button></form>") if g.get("status") == "pending" and g.get("guidance_found") else ""
            o.append(f"<tr><td><b>{e(g['ticker'])}</b></td><td>{link}<br><span class='muted'>{e(src.get('date', ''))}</span></td>"
                     f"<td>{res}</td><td><span class='tag'>{e(g.get('status'))}</span>{btn}</td></tr>")
        o.append("</table>")
        o.append(f"""<h2>粘贴电话会文字 / CFO 发言</h2><p class="muted">MSFT、GOOGL、AMZN、ORCL 的 CapEx 指引多在电话会上给出，新闻稿里没有。
把相关段落粘贴到这里，Claude 会按同样规则提取。</p>
<form method="post" class="box">{hidden}<label>公司</label><select name="ticker">{''.join(f'<option>{t}</option>' for t in DEMAND)}</select>
<label>来源说明（例：Q3 FY2027 earnings call, CFO remarks）</label><input name="label" style="width:100%">
<label>日期（YYYY-MM-DD）</label><input name="date" placeholder="2026-10-29">
<label>文字</label><textarea name="text" rows="10"></textarea>
<button name="action" value="add_guidance">提取并加入</button></form>""")
        rej = {k: v for k, v in (doc.get("i18n_rejected") or {}).items() if v >= 3}
        if rej:
            o.append(f"<h2>需要手工翻译</h2><p class='err'>以下内容自动翻译多次未通过数字核对，请在编辑页的 i18n 里手工填写：{e(', '.join(rej))}</p>")
        o.append(f"<p class='muted'><a href='/admin/journal/edit?token={e(token)}&jid={e(jid)}'>原始 JSON 编辑页</a></p>")
        return "".join(o)

    @router.get("/admin/journal/ai", response_class=HTMLResponse)
    def journal_ai_page(token: str = "", jid: str = "J-001"):
        _check_token(token)
        d = _need_store().get(jid)
        if d is None:
            raise HTTPException(status_code=404, detail="not found")
        return _ai_page(jid, token, d)

    @router.post("/admin/journal/ai", response_class=HTMLResponse)
    async def journal_ai_action(request: Request):
        form = {k: v[0] for k, v in parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True).items()}
        token, jid, action = form.get("token", ""), form.get("jid", "J-001"), form.get("action", "")
        _check_token(token)
        s = _need_store()
        old = s.get(jid)
        if old is None:
            raise HTTPException(status_code=404, detail="not found")
        new = copy.deepcopy(old)
        today = datetime.now(timezone.utc).date().isoformat()
        try:
            if action == "approve":
                if not form.get("zh", "").strip():
                    raise ValueError("中文内容不能为空")
                AI.approve_draft(new, form["draft_id"], form["zh"].strip(), form.get("en", "").strip(),
                                 form.get("fr", "").strip(), "Yanan", today)
                msg = "已批准并发布到复盘记录"
            elif action == "reject":
                AI.reject_draft(new, form["draft_id"], today)
                msg = "已拒绝"
            elif action == "verify_guidance":
                g = next(x for x in new.get("guidance_log", []) if x["id"] == form.get("gid"))
                g["status"], g["verified_at"] = "verified", today
                msg = "已标记为核验无误"
            elif action == "add_guidance":
                if form.get("ticker") not in DEMAND or len(form.get("text", "")) < 50:
                    raise ValueError("请选择公司并粘贴足够的文字")
                if not AI.ai_enabled():
                    raise ValueError("ANTHROPIC_API_KEY 未设置")
                ent = AI.add_manual_guidance(new, form["ticker"], form.get("label", "manual"),
                                             form.get("date", today), form["text"])
                if ent is None:
                    raise ValueError("这段文字已经处理过")
                new["status_board"] = status_board(new, datetime.now(timezone.utc).date())
                msg = "已提取：" + ("找到指引" if ent["guidance_found"] else "未发现指引")
            else:
                raise ValueError("unknown action")
            errs = validate_update(old, new)
            if errs:
                raise ValueError("; ".join(errs))
            s.save(jid, new, "manual")
            return _ai_page(jid, token, new, f"<p class='ok'>✓ {html.escape(msg)}</p>")
        except Exception as ex:
            return _ai_page(jid, token, old, f"<p class='err'>未保存：{html.escape(str(ex))}</p>")
except ImportError:
    router = None
