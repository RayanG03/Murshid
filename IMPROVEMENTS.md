# Backend improvements — 2026-10-25

Applied on top of the commit that added the committed PDF transcriptions
(`sources/*/transcribed/`) and the recalibrated KAU gates. All changes are
in `backend/`; the frontend is untouched and no dependency was added. The
code as downloaded before these changes is in `_backup_before_improvements/`
(do not commit that folder).

Tests: 187/188 passing as downloaded (one stale test pinned the old KAU
gate values — updated) → **226 passing** after the changes.

**Rule of thumb:** anything that changes retrieval scores or gate decisions
ships OFF (calibrated behavior stays the default). Turn it on only after
`src.eval_suite` shows it helps.

| # | Change | Default | Files |
|---|--------|---------|-------|
| 1 | Every ingest refuses 0 chunks instead of wiping the collection; clear hint when `backend/data/*.csv` is missing | on | `chroma_store.py`, `ingest/*` |
| 2 | Startup warns about empty collections; `/ask` returns 503 "knowledge base empty" instead of refusing every question; new `GET /health/details` (per-collection counts). `GET /health` unchanged for the launcher | on | `main.py` |
| 3 | Follow-ups: the answer prompt also shows the rewritten standalone question (retrieval already used it) | on | `prompts.py`, `pipeline.py` |
| 4 | Rerank context filter: after the gate passes, items below `max(0.10, 0.20 × top)` no longer reach the LLM | on | `ksu_gate.py`, `pipeline.py` |
| 5 | KAU rerank gate (`kau_gate_mode=rerank`) — the reranker is already loaded at startup; replaces the cosine window, which cannot separate hits from refusals | **on since eval run 2** (`rerank`) | `pipeline.py`, `config.py` |
| 6 | Saudi-dialect → MSA mapping on the search query only | **on since eval run 2** | `dialect.py` |
| 7 | Hybrid retrieval: BM25 keyword candidates added to the rerank pool (pure Python, built from Chroma) | off (`0`) | `lexical.py`, `retrieval.py` |
| 8 | Numeric grounding guard: numbers in the answer missing from the context are logged (`log`) or refused (`refuse`) | `log` | `grounding.py` |
| 9 | `/ask` also returns `sources` (evidence actually sent to the model); `ask.jsonl` gains search_query, gate, gate_top_score, sources, ungrounded_numbers, history_turns | on | `main.py` |
| 10 | Answer cache for repeated questions without history (TTL 1 h) — saves free-tier quota during demos | on (256) | `answer_cache.py` |
| 12 | Failover may start an attempt with ≥ `min_attempt_seconds` (1.5 s; Groq answers in 1.0–1.4 s here) left, timeout clipped to the remaining budget. Before: 8 s had to remain, so slow KSU search + rerank (9–11 s on a laptop CPU) produced 502 / canned UI answers without ever calling the LLM | on | `generators.py`, `config.py` |
| 13 | Rewrite topic drift (live test): the rewrite prompt now forbids adding history topics to a new question (with examples), and a gate refusal on a rewritten query is retried once with the user's own wording when ≥ 9 s remain | on | `query_rewrite.py`, `pipeline.py` |
| 14 | KAU catalogue pages (colleges → programs lists in the admission/transfer guides) are chunked one college per chunk instead of packing ~700 chars of unrelated colleges together (live miss: "التخصصات المتاحة في كلية الحاسبات" scored 0.36). kau_pdf 309 → 342 chunks; re-ingest `run_kau --only pdf` | on | `ingest/kau_pdf.py` |
| 15 | General-information sources: `sources/KAU/web/about_kau.md`, `kau_leadership.md`, `kau_colleges.md` and a new `sources/KSU/web/` folder (`about_ksu.md`, `ksu_leadership.md`, `ksu_colleges.md`), each with source URLs and retrieval date in the YAML header. KSU web pages are ingested into `ksu_txt` (`run_ksu --only txt`) | on | `sources/*/web`, `ingest/ksu_txt.py` |
| 16 | Opt-in fallback after a gate refusal OR when the answer model itself returns the refusal sentence (context passed the gate but did not contain the answer): `web` = official-website answer with links — backend `tavily` (search API with `include_domains` = kau.edu.sa / ksu.edu.sa, page text answered by the normal chain, answers whose numbers are not on the pages are rejected) or `gemini` (Google Search grounding, accepted only when a grounding source is on kau.edu.sa / ksu.edu.sa, answer labeled 🔎 with links; `llm` = general (non-regulation, on-topic) questions only, labeled ⚠️ not verified; `web_then_llm`. Smoke test: `python -m src.general_fallback "سؤال" kau web`. Not live-tested where written (no network to Google) | **off** (needs team agreement; changes the official-sources-only guarantee and the AGENTS.md no-fetch rule) | `general_fallback.py`, `pipeline.py` |
| 17 | Follow-up chips regression fix: dependent follow-ups ("لخّص الإجابة بنقاط", "اشرح لي أكثر", "أعطني مثالاً", "هل هناك مواعيد محددة؟") that the rewrite returns unchanged — or when the rewrite call fails — are anchored to the previous question for retrieval; rewrite prompt gained examples for them | on | `query_rewrite.py` |
| 18 | `[timing]` log line per rerank-gated request (search / rerank / time left for the answer) now visible in the launcher console (capstone logger handler). Startup warm-up: one dummy search + rerank per university before `/health` reports ok. Live: the first question after startup spent ~17 s in search + rerank (lazy Chroma index load + first cross-encoder pass while Angular was building) and returned 502 → canned UI answer | on (`warm_up_on_startup`) | `main.py` |
| 19 | Provider quota page served by the backend: **http://localhost:8000/status** (JSON at `/usage`). Groq: remaining requests today / tokens this minute + reset countdown from its `x-ratelimit-*` headers; Tavily: live credits from `GET https://api.tavily.com/usage`; Gemini: next daily reset (midnight Pacific) + last 429; Mistral: any rate-limit headers it sends; plus our own per-day counters persisted in `data/logs/usage.json` (no keys, no text). "فحص الآن" sends one tiny request per generator. Frontend untouched | on | `usage_tracker.py`, `status_page.py`, `generators.py`, `main.py` |
| 11 | Held-out evaluation set + suite (`eval/questions.csv`, `src.eval_suite`): hit rate, refusal accuracy, score windows for calibration, optional real answers | — | `eval/`, `eval_suite.py` |

## How to use

From `backend/` (prefix `python -m uv run` on machines where `uv` is not on PATH):

```powershell
# 1. measure the current system (no LLM calls)
uv run python -X utf8 -m src.eval_suite --matrix

# 2. try one change
uv run python -X utf8 -m src.eval_suite --university kau --set kau_gate_mode=rerank

# 3. adopt it: add the line to backend/.env, restart the backend
#    KAU_GATE_MODE=rerank
#    HYBRID_LEXICAL_EXTRA=3
#    DIALECT_NORMALIZATION=true
#    NUMERIC_GUARD_MODE=refuse

# 4. real answers through the provider chain (costs API calls)
uv run python -X utf8 -m src.eval_suite --university ksu --generate
```

## Open points for the team

- Not run end-to-end where written (no model weights there). Unit tests
  cover the logic with fakes; thresholds for the new options are starting
  values — calibrate with the `max_refuse_score` / `min_hit_score` window
  printed by `src.eval_suite`.
- `ksu_pdf` now ingests BOTH `اللائحة التنظيمية للشؤون الدراسية` (older
  edition, no executive rules) and `...24` (2024 consolidated edition).
  AGENTS.md previously excluded the older one as redundant; two editions
  of the same articles can put conflicting/outdated text in the context.
  Consider moving it to `sources/KSU/_excluded/`.
- `eval/questions.csv` is a 51-row seed set; extend it with real student
  questions before reporting numbers.

## First eval run (2026-10-25, Khaled's laptop, gate-only)

| config | KAU hit / refuse | KSU hit / refuse | notes |
|---|---|---|---|
| baseline | 16/17 · 4/6 | 14/17 · 7/8 | KSU search+rerank median 7.4 s, max 11 s |
| kau_rerank | fixes calendar (kau16), housing fees (kau_r02), bus (kau_r01); loses MyKAU app (kau11) | – | adds ~7 s to every KAU request |
| dialect | – | fixes visiting student (ksu13), no regressions seen | no latency cost |
| hybrid | no gain | no gain | ~13 s median on KSU — not usable on CPU |

That run lost rows to the request deadline (eval bug, fixed: gate-only runs
no longer apply the deadline) and used raw keyword matching (fixed:
spelling-insensitive). Two labels were corrected (ksu_r04 → ksu20 is
answered by ksu_qa #38; kau09 keyword). Re-run before deciding.

Real issues found:
- `kau_r02` (housing fees) passes the recalibrated cosine gate
  (strong_leader 0.832) on unrelated Q&A pairs (#475, #18, #500).
- `kau16` (final-exam date) is refused at 0.8318 — 0.0002 under the
  leader threshold: the cosine window is still razor-thin.
- KSU latency on a laptop CPU is the main risk for the demo (see #12).

## Second eval run (fixed evaluator) — decision

| config | KAU hit · refuse · median | KSU hit · refuse · median |
|---|---|---|
| baseline | 16/17 · 4/6 · 0.22 s | 17/19 · 7/7 · 7.45 s |
| kau_rerank | 16/17 · 6/6 · 7.23 s | 17/19 · 7/7 · 7.40 s |
| dialect | 16/17 · 4/6 · 0.21 s | 18/19 · 7/7 · 7.51 s |
| hybrid | 16/17 · 4/6 · 0.22 s | 17/19 · 7/7 · 13.14 s |
| **rerank+dialect** | **16/17 · 6/6 · 7.06 s** | **18/19 · 7/7 · 7.33 s** |

Score windows: KAU cosine max refuse 0.8624 > min hit 0.8411 (overlap — no
cosine threshold can separate them); KAU rerank max refuse 0.3399 < min hit
0.5827 with dialect; KSU rerank 0.2787 < 0.4734.

Adopted as defaults: `kau_gate_mode=rerank`, `dialect_normalization=true`.
Hybrid stays off (no gain, ~2x latency). Remaining misses: ksu07 ("انطوى
قيدي") and kau11 (MyKAU download — the page has no real content, a known
gap). Revert either default with `KAU_GATE_MODE=cosine` /
`DIALECT_NORMALIZATION=false` in `backend/.env`.

## Live check of the general-knowledge fallback (2026-10-25) — not recommended

`python -m src.general_fallback "من هو رئيس الجامعة؟" kau llm` (Groq,
openai/gpt-oss-120b) answered with an invented name ("عبدالعزيز بن عبدالعزيز
بن عبدالعزيز السعيد"); the official page names Dr. Tareef Alaama. A warning
label does not make a fabricated fact acceptable, so `llm` / `web_then_llm`
should stay off. The same question is now answered from
`sources/KAU/web/kau_leadership.md`. `web` mode returned HTTP 429 (Gemini
quota) on the team key — it needs a key with Google Search grounding quota.

## Live check of the official-website fallback (2026-10-25, Tavily) — works

`python -m src.general_fallback "متى تبدأ الاختبارات النهائية للفصل الأول؟" ksu web`
→ 4.5 s, answered "الأحد 13/12/2026" from the 2026-2027 calendar on
dar.ksu.edu.sa (official), with links. One returned link was an older (1446)
calendar, so the prompt now prefers the newest academic year and the answer
lists only the pages the model says it used. Tests force the fallback off
and blank TAVILY_API_KEY (`tests/conftest.py`) so they never call the network.

## Web answer quality (2026-10-25, after a live incomplete answer)

"وشو الكليات الي في المزاحمية" returned only one program page. Now: the
search query is the normalized question + university name; Tavily runs at
`advanced` depth (relevant chunks per page, 2 credits/search) with 5 results;
the full page texts are split into ~700-char passages and the best ones are
picked with BM25 (no extra model, no CPU cost) next to Tavily's own snippets;
the model must say when the pages cover only part of the answer; links are
deduplicated per page. Accuracy still depends on what the site publishes —
facts the demo must get right belong in `sources/*/web/*.md`.

## Frontend: provider quota page (2026-10-25, by Khaled — frontend owner)

New standalone component `frontend/src/app/usage-status/` (signals, native
control flow, OnPush default, `output()`), opened from a gauge button in the
top bar ("رصيد المزوّدين", next to the dark-mode toggle) and shown in the main
area instead of the thread; "رجوع للمحادثة", switching university, opening a
conversation or "محادثة جديدة" return to the chat. It reads
`http://localhost:8000/usage` every 30 s, ticks reset countdowns every
second, and "فحص الآن" calls `POST /usage/probe`. Uses the shell's CSS
variables, so university colours and dark mode apply. `ng build` and
`ng test` pass (6 tests: 2 new for the component, 1 new for the view toggle).
The backend page http://localhost:8000/status stays as a standalone view.

### Quota page: provider-reported numbers only (2026-10-25)

`GET /usage?live=true` (page open + "تحديث مباشر") asks every provider now:
Groq `GET /models` (used when it carries `x-ratelimit-*` headers) else one
8-token chat call; Mistral the same; Gemini one tiny call (status only —
Gemini exposes no remaining-quota API on the free tier); Tavily its official
`GET /usage`, never cached. Computed values were removed (Gemini reset time,
documented free-tier text); the per-device counters are labelled "طلبات هذا
الجهاز اليوم" because shared keys are also used by teammates. Live checks
never increment those counters. Tests: backend 227, frontend 6.
