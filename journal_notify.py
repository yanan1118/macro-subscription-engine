"""
journal_notify.py
用 Resend 发邮件提醒。每次自动运行后汇总成一封邮件（没有事情就不发），同一件事只提醒一次。

提醒内容：
  ⚠️ 假设触发证伪（最重要，标题会标出）
  📝 新的 Review 草稿等你批准
  🔍 新提取的 CapEx 指引等你核验
  📅 预测 30 天内到期 / 已自动判定、等你确认分类
  ⏰ 有待处理事项超过 7 天未处理（每周提醒一次）
  ❗ 运行出错（同一错误 3 天内只提醒一次）

Railway 环境变量：
    RESEND_API_KEY       必填（可以和你其他项目用同一个）
    NOTIFY_EMAIL_TO      收件人，例如 yanan.huang1118@gmail.com
    NOTIFY_EMAIL_FROM    可选，默认 "Minerva Journal <onboarding@resend.dev>"
                         （resend.dev 发件地址只能发给 Resend 账号本人的邮箱；
                          若你在 Resend 验证过 minerva-invest.eu，可改成 journal@minerva-invest.eu）
    PUBLIC_API_BASE      可选，邮件里管理页链接的地址，默认你的 Railway 地址
"""
import hashlib
import html
import os
from datetime import date, datetime, timedelta, timezone

import requests

RESEND_URL = "https://api.resend.com/emails"
DEFAULT_FROM = "Minerva Journal <onboarding@resend.dev>"
DEFAULT_BASE = "https://macro-subscription-engine-production.up.railway.app"

STATUS_ZH = {"manual": "人工判断", "insufficient_data": "数据不足", "not_triggered": "未触发",
             "triggered": "已触发证伪", "partial": "部分满足", "needs_review": "待人工复核"}


def notify_enabled():
    return bool(os.environ.get("RESEND_API_KEY", "").strip() and os.environ.get("NOTIFY_EMAIL_TO", "").strip())


def send_resend(subject, html_body):
    r = requests.post(RESEND_URL, timeout=30, json={
        "from": os.environ.get("NOTIFY_EMAIL_FROM", "").strip() or DEFAULT_FROM,
        "to": [a.strip() for a in os.environ["NOTIFY_EMAIL_TO"].split(",") if a.strip()],
        "subject": subject, "html": html_body,
    }, headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY'].strip()}",
                "Content-Type": "application/json"})
    if r.status_code >= 300:
        raise RuntimeError(f"Resend HTTP {r.status_code}: {r.text[:300]}")
    return r.json().get("id")


def _d(s):
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


# ---------------------------------------------------------------- 收集事件
def collect_events(jid, old, new, report, today):
    """返回 [(key, severity, html_line)]；severity: 3=证伪 2=需要操作 1=信息"""
    e = html.escape
    ev = []
    ob = {k: v.get("status") for k, v in (old.get("status_board") or {}).items() if k != "as_of"}
    for h, v in (new.get("status_board") or {}).items():
        if h == "as_of":
            continue
        st = v.get("status")
        if st == "triggered":
            ev.append((f"triggered:{h}", 3, f"⚠️ <b>{e(h)} 已触发证伪条件</b>（此前：{e(STATUS_ZH.get(ob.get(h), ob.get(h) or '—'))}）。"
                                          f"请检查数据并在 Review 中记录。"))
        elif st == "needs_review":
            ev.append((f"needs_review:{h}:{hashlib.sha1(str(v.get('detail')).encode()).hexdigest()[:8]}", 2,
                       f"🔎 {e(h)} 需要人工复核：{e('; '.join((v.get('detail') or {}).get('needs_review', [])))}"))
        elif ob.get(h) and ob.get(h) != st:
            ev.append((f"status:{h}:{st}:{today.isoformat()}", 1,
                       f"ℹ️ {e(h)} 状态变化：{e(STATUS_ZH.get(ob[h], ob[h]))} → {e(STATUS_ZH.get(st, st))}"))
    for d in new.get("review_drafts", []):
        if d.get("status") == "draft":
            ev.append((f"draft:{d['id']}", 2, f"📝 新的 Review 草稿 <b>{e(d['id'])}</b> 等你批准：{e(d.get('zh', '')[:120])}…"))
    for g in new.get("guidance_log", []):
        if g.get("status") == "pending" and g.get("guidance_found"):
            st = g.get("statements") or [{}]
            s0 = st[0]
            amt = (f"${s0.get('low_usd_b')}–{s0.get('high_usd_b')}B" if s0.get("low_usd_b") is not None else s0.get("guidance_type", ""))
            ev.append((f"guidance:{g['id']}", 2, f"🔍 新的 CapEx 指引等你核验：<b>{e(g['ticker'])}</b> "
                                                f"{e(s0.get('period_covered', ''))} {e(str(amt))}（{e(g.get('source', {}).get('label', ''))}）"))
    for p in new.get("predictions", []):
        r = p.get("result", {})
        if r.get("status") == "evaluated" and not r.get("category"):
            ev.append((f"evaluated:{p['id']}", 2, f"📅 预测 <b>{e(p['id'])}</b> 已自动判定为"
                                                 f"{'正确' if r.get('auto_outcome') == 'correct' else '错误'}，请确认最终分类（✅/❌/⏱/⚪）。"))
        elif r.get("status") == "pending":
            days = (_d(p["due_date"]) - today).days
            if 0 < days <= 30:
                ev.append((f"due_soon:{p['id']}", 1, f"📅 预测 <b>{e(p['id'])}</b> 将在 {days} 天后（{e(p['due_date'])}）到期。"))
            if r.get("last_error"):
                ev.append((f"eval_error:{p['id']}:{today.isoformat()}", 2, f"❗ 预测 {e(p['id'])} 判定失败，将自动重试：{e(r['last_error'])}"))
    errs = {}
    errs.update({f"SEC {k}": v for k, v in (report.get("errors") or {}).items()})
    if report.get("guard_errors"):
        errs["写入守卫拒绝保存"] = "; ".join(report["guard_errors"])
    for k in ("guidance", "translation"):
        x = report.get(k)
        if isinstance(x, dict) and x.get("error"):
            errs[k] = x["error"]
        if isinstance(x, dict) and x.get("errors"):
            errs.update({f"{k} {t}": m for t, m in x["errors"].items()})
    if isinstance(report.get("review_draft"), dict):
        errs["Review 起草"] = report["review_draft"].get("error", "")
    for k, v in errs.items():
        h = hashlib.sha1(f"{k}{v}".encode()).hexdigest()[:10]
        ev.append((f"error:{h}", 2, f"❗ 运行出错 · {e(k)}：{e(str(v)[:300])}"))
    return ev


# ---------------------------------------------------------------- 去重与发送
COOLDOWN_DAYS = {"error": 3}   # 同一错误 3 天内只提醒一次；其他事件只提醒一次
REMIND_EVERY = 7


def build_email(jid, events, pending_count, base):
    sev = max(s for _, s, _ in events)
    n_action = sum(1 for _, s, _ in events if s >= 2)
    if sev == 3:
        subject = f"⚠️ [Minerva Journal {jid}] 假设触发证伪"
    elif n_action:
        subject = f"[Minerva Journal {jid}] {n_action} 项待处理"
    else:
        subject = f"[Minerva Journal {jid}] 数据更新"
    rows = "".join(f"<li style='margin:6px 0'>{line}</li>" for _, _, line in sorted(events, key=lambda x: -x[1]))
    body = f"""<div style="font-family:-apple-system,Segoe UI,sans-serif;font-size:14px;color:#24292e;max-width:640px">
<h2 style="color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:6px">Minerva Research Journal {html.escape(jid)}</h2>
<ul style="padding-left:18px">{rows}</ul>
<p style="margin-top:18px"><b>处理入口</b>（需在网址后加 <code>?token=你的token</code>，建议在浏览器里存成书签）：<br>
AI 工作台：<a href="{base}/admin/journal/ai">{base}/admin/journal/ai</a><br>
公开页面：<a href="https://minerva-invest.eu/research-journal/?j={html.escape(jid)}">minerva-invest.eu/research-journal/</a></p>
<p style="color:#6a737d;font-size:12px">当前共有 {pending_count} 项待处理。此邮件由 Railway 自动运行生成。</p></div>"""
    return subject, body


def pending_items(doc):
    return (sum(1 for d in doc.get("review_drafts", []) if d.get("status") == "draft")
            + sum(1 for g in doc.get("guidance_log", []) if g.get("status") == "pending" and g.get("guidance_found"))
            + sum(1 for p in doc.get("predictions", []) if p["result"].get("status") == "evaluated" and not p["result"].get("category")))


def notify(store, jid, old, new, report, today=None, send=None):
    """在 run_once 之后调用；返回发送结果（或跳过原因）"""
    send = send or send_resend
    today = today or datetime.now(timezone.utc).date()
    events = []
    for key, sev, line in collect_events(jid, old, new, report, today):
        kind = key.split(":", 1)[0]
        if store.notified(jid, key, within_days=COOLDOWN_DAYS.get(kind)):
            continue
        events.append((key, sev, line))
    pend = pending_items(new)
    if not events and pend and not store.notified(jid, "reminder", within_days=REMIND_EVERY):
        events.append(("reminder", 2, f"⏰ 还有 {pend} 项待处理（草稿、指引核验或预测分类），请抽空打开 AI 工作台。"))
    if not events:
        return {"sent": False, "reason": "nothing new"}
    base = (os.environ.get("PUBLIC_API_BASE", "").strip() or DEFAULT_BASE).rstrip("/")
    subject, body = build_email(jid, events, pend, base)
    mid = send(subject, body)
    for key, _, _ in events:
        store.mark_notified(jid, key)
    if any(s >= 2 for _, s, _ in events):
        store.mark_notified(jid, "reminder")   # 刚提醒过，7 天内不再催
    return {"sent": True, "id": mid, "subject": subject, "events": [k for k, _, _ in events]}