"""Provider quota report (GET /usage) and its HTML page (GET /status).

Served by the backend itself (frontend untouched, per AGENTS.md):
http://localhost:8000/status
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from src.config import Settings
from src.usage_tracker import UsageTracker, groq_view

ROLES = {
    "groq": "المولّد الأساسي",
    "mistral": "احتياطي أول",
    "gemini": "احتياطي ثانٍ",
    "tavily": "البحث في موقع الجامعة",
}

# Only numbers the providers themselves return are shown. Gemini has no
# API for the remaining free-tier quota, so only its live status is shown.
NOTES = {
    "gemini": "Gemini لا يتيح رقم الرصيد المتبقي عبر الواجهة البرمجية؛ "
              "الرقم موجود فقط في AI Studio",
}


def _tavily_usage(settings: Settings) -> dict | None:
    """Live Tavily credits from its official GET /usage (never cached)."""
    if not settings.tavily_api_key:
        return None
    try:
        resp = httpx.get("https://api.tavily.com/usage", timeout=8.0,
                         headers={"Authorization": f"Bearer {settings.tavily_api_key}"})
        if resp.status_code != 200:
            return {"error": resp.status_code}
        data = resp.json()
        data["fetched_at"] = time.time()
        return data
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}


_LAST_TAVILY: dict[str, Any] = {"data": None}


def _check_openai_compat(provider: str, base_url: str, api_key: str, model: str,
                         tracker: UsageTracker) -> str:
    """Live headers: GET /models first (free); if the provider sends no
    rate-limit headers there, one tiny chat call. Returns the source used."""
    if not api_key:
        return "not_configured"
    headers = {"Authorization": f"Bearer {api_key}"}
    url = base_url.rstrip("/")
    try:
        resp = httpx.get(f"{url}/models", headers=headers, timeout=8.0)
        if resp.status_code == 200 and any(
                "ratelimit" in k.lower() for k in resp.headers.keys()):
            tracker.record_response(provider, model, resp.headers, count=False)
            return "models"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
        }
        payload["max_completion_tokens" if provider == "groq" else "max_tokens"] = 8
        if provider == "groq":
            payload["reasoning_effort"] = "low"
        resp = httpx.post(f"{url}/chat/completions", headers=headers,
                          json=payload, timeout=10.0)
        if resp.status_code == 200:
            tracker.record_response(provider, model, resp.headers, count=False)
        else:
            tracker.record_error(provider, resp.status_code, resp.text[:200], resp.headers)
        return "chat"
    except Exception as exc:  # noqa: BLE001
        tracker.record_error(provider, 0, type(exc).__name__)
        return "error"


def _check_gemini(settings: Settings, tracker: UsageTracker) -> str:
    """Tiny generateContent call: shows whether the key can serve now."""
    if not settings.gemini_api_key:
        return "not_configured"
    url = (f"{settings.gemini_base_url.rstrip('/')}/models/"
           f"{settings.gemini_model_id}:generateContent")
    try:
        resp = httpx.post(url, timeout=10.0, headers={"x-goog-api-key": settings.gemini_api_key},
                          json={"contents": [{"parts": [{"text": "ping"}]}],
                                "generationConfig": {"maxOutputTokens": 8}})
        if resp.status_code == 200:
            tracker.record_response("gemini", settings.gemini_model_id, resp.headers, count=False)
        else:
            tracker.record_error("gemini", resp.status_code, resp.text[:200], resp.headers)
        return "chat"
    except Exception as exc:  # noqa: BLE001
        tracker.record_error("gemini", 0, type(exc).__name__)
        return "error"


def live_check(settings: Settings, tracker: UsageTracker) -> dict[str, str]:
    """Ask every provider NOW; results land in the tracker."""
    sources = {
        "groq": _check_openai_compat("groq", settings.groq_base_url, settings.groq_api_key,
                                     settings.groq_model_id, tracker),
        "mistral": _check_openai_compat("mistral", settings.mistral_base_url,
                                        settings.mistral_api_key, settings.mistral_model_id,
                                        tracker),
        "gemini": _check_gemini(settings, tracker),
    }
    _LAST_TAVILY["data"] = _tavily_usage(settings)
    sources["tavily"] = "usage_api" if settings.tavily_api_key else "not_configured"
    return sources


def build_usage_report(settings: Settings, tracker: UsageTracker,
                       live: bool = False) -> dict[str, Any]:
    """Provider-reported numbers only (see module docstring)."""
    sources = live_check(settings, tracker) if live else {}
    snap = tracker.snapshot()
    prov, today = snap["providers"], snap["today"]
    keys = {
        "groq": bool(settings.groq_api_key),
        "mistral": bool(settings.mistral_api_key),
        "gemini": bool(settings.gemini_api_key),
        "tavily": bool(settings.tavily_api_key),
    }
    out = []
    for pid in list(settings.default_generation_providers) + ["tavily"]:
        p = prov.get(pid, {})
        item: dict[str, Any] = {
            "id": pid,
            "role": ROLES.get(pid, ""),
            "configured": keys.get(pid, False),
            "model": p.get("model", ""),
            "last_ok_at": p.get("last_ok_at"),
            "last_error": p.get("last_error"),
            "device_today": today.get(pid, {}),
            "note": NOTES.get(pid, ""),
            "headers_at": p.get("rate_headers_at"),
            "check_source": sources.get(pid),
        }
        rate = p.get("rate_headers") or {}
        if pid == "groq" and rate:
            item["groq"] = groq_view(rate, p.get("rate_headers_at") or snap["now"])
        elif rate:
            item["headers"] = rate
        if pid == "tavily" and keys["tavily"]:
            if _LAST_TAVILY["data"] is None:
                _LAST_TAVILY["data"] = _tavily_usage(settings)
            item["account"] = _LAST_TAVILY["data"]
        out.append(item)
    return {
        "now": snap["now"],
        "live": live,
        "fallback_mode": settings.general_fallback_mode,
        "providers": out,
        "local_models": [settings.embedding_model_name, settings.reranker_model_name],
    }


def probe_providers(settings: Settings) -> dict[str, str]:
    """Back-compat: same as a live check."""
    from src.usage_tracker import get_tracker

    return live_check(settings, get_tracker())


STATUS_HTML = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>رصيد المزودين — مُرشِد</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16202c;--muted:#5f6b7a;--line:#e3e7ec;
--ok:#1f8a4c;--warn:#b7791f;--bad:#c53030;--accent:#2b5fd9}
@media (prefers-color-scheme:dark){:root{--bg:#0f141a;--card:#18202a;--ink:#e8edf3;
--muted:#9aa7b5;--line:#2a3542;--ok:#48bb78;--warn:#ecc94b;--bad:#fc8181;--accent:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Segoe UI",Tahoma,"Noto Naskh Arabic",sans-serif;line-height:1.6}
header{padding:20px 16px 8px;max-width:1100px;margin:auto;display:flex;gap:12px;
align-items:center;justify-content:space-between;flex-wrap:wrap}
h1{font-size:1.4rem;margin:0}.sub{color:var(--muted);font-size:.9rem}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 14px;
font:inherit;cursor:pointer}button:disabled{opacity:.6;cursor:wait}
main{max-width:1100px;margin:auto;padding:8px 16px 32px;display:grid;gap:14px;
grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h2{margin:0;font-size:1.1rem;display:flex;justify-content:space-between;gap:8px}
.role{color:var(--muted);font-size:.85rem;margin-bottom:10px}
.badge{font-size:.75rem;border-radius:99px;padding:2px 10px;border:1px solid currentColor}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.muted{color:var(--muted)}
.metric{margin:10px 0}.metric .row{display:flex;justify-content:space-between;font-size:.92rem}
.bar{height:8px;background:var(--line);border-radius:99px;overflow:hidden;margin-top:4px}
.bar>i{display:block;height:100%;background:var(--ok)}
.small{font-size:.8rem;color:var(--muted)}.err{font-size:.8rem;margin-top:8px}
code{direction:ltr;unicode-bidi:embed;font-size:.78rem}
</style>
</head>
<body>
<header>
  <div><h1>رصيد المزودين</h1>
  <div class="sub" id="sub">يتحدث تلقائياً كل 30 ثانية</div></div>
  <button id="probe" title="يسأل كل مزوّد الآن ويعرض أرقامه كما هي">تحديث مباشر</button>
</header>
<main id="cards"></main>
<script>
const NAMES={groq:"Groq",mistral:"Mistral",gemini:"Gemini",tavily:"Tavily"};
const fmtN=n=>n==null?"—":Number(n).toLocaleString("ar-SA-u-nu-latn");
const fmtT=t=>t?new Date(t*1000).toLocaleString("ar-SA-u-nu-latn",{hour:"2-digit",minute:"2-digit",day:"numeric",month:"short"}):"—";
function left(t){if(!t)return"—";let s=Math.max(0,Math.round(t-Date.now()/1000));
 const h=Math.floor(s/3600),m=Math.floor(s%3600/60);s%=60;
 return h?`${h} ساعة ${m} دقيقة`:m?`${m} دقيقة ${s} ثانية`:`${s} ثانية`}
function ago(t){if(!t)return"لم يُستخدم بعد";const s=Math.round(Date.now()/1000-t);
 return s<60?`قبل ${s} ثانية`:s<3600?`قبل ${Math.round(s/60)} دقيقة`:`قبل ${Math.round(s/3600)} ساعة`}
function bar(rem,lim){if(rem==null||!lim)return"";const p=Math.max(0,Math.min(100,rem/lim*100));
 const c=p>40?"var(--ok)":p>15?"var(--warn)":"var(--bad)";
 return `<div class="bar"><i style="width:${p}%;background:${c}"></i></div>`}
function metric(label,rem,lim,reset){return `<div class="metric"><div class="row"><span>${label}</span>
 <b>${fmtN(rem)} / ${fmtN(lim)}</b></div>${bar(rem,lim)}
 <div class="small" data-reset="${reset||""}">يتجدد بعد: <span class="cd">${left(reset)}</span></div></div>`}
function card(p){
 const name=NAMES[p.id]||p.id;let state,cls;
 if(!p.configured){state="غير مفعّل";cls="muted"}
 else if(p.last_error&&(!p.last_ok_at||p.last_error.at>p.last_ok_at)){state=p.last_error.status==429?"خلص الرصيد مؤقتاً":"خطأ";cls="bad"}
 else if(p.last_ok_at){state="شغال";cls="ok"}else{state="لم يُختبر";cls="muted"}
 let h=`<div class="card"><h2>${name}<span class="badge ${cls}">${state}</span></h2>
 <div class="role">${p.role}${p.model?` · <code>${p.model}</code>`:""}</div>`;
 if(p.groq){const g=p.groq;
  h+=metric("الطلبات المتبقية اليوم",g.requests_remaining_day,g.requests_limit_day,g.requests_reset_at);
  h+=metric("التوكنز المتبقية هذه الدقيقة",g.tokens_remaining_minute,g.tokens_limit_minute,g.tokens_reset_at);
  h+=`<div class="small">حسب آخر رد من Groq ${ago(p.headers_at)}</div>`}
 if(p.id==="tavily"&&p.account){const a=p.account;
  if(a.error)h+=`<div class="err bad">تعذر جلب الرصيد (${a.error})</div>`;
  else{const k=a.key||{},ac=a.account||{};const lim=k.limit??ac.plan_limit,used=k.usage??ac.plan_usage;
   h+=`<div class="metric"><div class="row"><span>الرصيد المتبقي هذا الشهر</span><b>${fmtN(lim!=null&&used!=null?lim-used:null)} / ${fmtN(lim)}</b></div>${bar(lim-used,lim)}
   <div class="small">المستخدم حسب Tavily: ${fmtN(used)} · الخطة: ${ac.current_plan||"—"} · وقت القراءة: ${fmtT(a.fetched_at)}</div></div>`}}
  if(p.headers&&!p.groq){h+=`<div class="small">معلومات الحدود من آخر رد:<br>${Object.entries(p.headers).map(([k,v])=>`<code>${k}: ${v}</code>`).join("<br>")}</div>`}
 const t=p.device_today||{};
 h+=`<div class="metric"><div class="row"><span>طلبات هذا الجهاز اليوم</span><b>${fmtN(t.requests||0)} طلب</b></div>
 <div class="small">${p.id==="tavily"?`الرصيد المستخدم: ${fmtN(t.credits||0)}`:`توكنز الإدخال: ${fmtN(t.prompt_tokens||0)} · الإخراج: ${fmtN(t.completion_tokens||0)}`}${t.errors?` · أخطاء: ${fmtN(t.errors)}`:""}</div></div>`;
 if(p.last_error){const e=p.last_error;h+=`<div class="err ${e.at>(p.last_ok_at||0)?"bad":"muted"}">آخر خطأ (${e.status}) ${ago(e.at)}${e.retry_after_s?` · أعد المحاولة بعد ${Math.round(e.retry_after_s)} ثانية`:""}</div>`}
 h+=`<div class="small" style="margin-top:8px">${p.note||""}</div><div class="small">آخر استخدام ناجح: ${ago(p.last_ok_at)}</div></div>`;
 return h}
function render(d){
 const local=`<div class="card"><h2>النماذج المحلية<span class="badge ok">بلا حدود</span></h2>
 <div class="role">تعمل على جهازك: البحث وإعادة الترتيب</div>${d.local_models.map(m=>`<div><code>${m}</code></div>`).join("")}
 <div class="small" style="margin-top:8px">لا تستهلك توكنز ولا رصيداً</div></div>`;
 document.getElementById("cards").innerHTML=d.providers.map(card).join("")+local;
 const modes={off:"مقفل",web:"مفعّل",llm:"جواب عام",web_then_llm:"مفعّل + جواب عام"};
 document.getElementById("sub").textContent=`آخر تحديث: ${fmtT(d.now)} · البحث في الموقع: ${modes[d.fallback_mode]||d.fallback_mode} · يتحدث كل 30 ثانية`}
async function load(live){try{const r=await fetch(live?"/usage?live=true":"/usage");render(await r.json())}
 catch(e){document.getElementById("sub").textContent="تعذر الاتصال بالخادم"}}
setInterval(()=>document.querySelectorAll("[data-reset]").forEach(el=>{
 const t=Number(el.dataset.reset);const cd=el.querySelector(".cd");if(cd&&t)cd.textContent=left(t)}),1000);
document.getElementById("probe").onclick=async e=>{e.target.disabled=true;e.target.textContent="جارٍ الفحص...";
 try{await load(true)}finally{e.target.disabled=false;e.target.textContent="تحديث مباشر"}};
load(true);setInterval(()=>load(false),30000);
</script>
</body>
</html>
"""
