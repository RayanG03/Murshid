# مُرشِد — Murshid · Arabic University Regulations Assistant

An Arabic **RAG (Retrieval-Augmented Generation)** question-answering chatbot over the official
academic regulations of **King Abdulaziz University (KAU)** and **King Saud University (KSU)**.
Students ask in plain Arabic; Murshid answers **only from the ingested official sources** and
cites them — or explicitly refuses when the corpus cannot back an answer.

`Python 3.12 · FastAPI · ChromaDB · sentence-transformers (e5-large) · Angular 22`

---

## What it does

- **Sourced Arabic answers** — every answer carries a citation line, e.g.
  `[المصدر: سؤال رقم 38]` or `[المصدر: دليل القبول … — المادة الثلاثون (صفحة 24)]`.
- **Refusal-first anti-hallucination** — retrieved evidence must pass calibrated relevance
  gates before generation; otherwise the bot replies
  `لا يمكنني تأكيد الإجابة من المصادر المتاحة.` instead of inventing content.
- **Two universities, one UI** — a KAU/KSU switcher selects which document collections are queried.
- **Conversational follow-ups** — the last two turns are sent with each question and rewritten
  into a standalone query before retrieval (with a safe fallback to the raw question).
- **Provider failover under a hard deadline** — the whole request is budgeted at 17 s; rate-limited
  or failing LLM providers are skipped immediately (Groq → Mistral → Gemini).
- **Full observability** — every question/answer and 👍/👎 feedback event is appended to JSONL logs
  with per-stage timings (rewrite / search / rerank / generation / total).


## How it works
```mermaid
flowchart LR
    U["Student"] -->|"Arabic question"| UI["Angular UI :4200"]
    UI -->|"POST /ask"| API["FastAPI :8000"]
    API --> RW["History-aware rewrite<br/>(last 2 turns)"]
    RW --> RET["Retrieval: ChromaDB + e5-large"]
    RET --> GATE{"Relevance gate"}
    GATE -->|"weak evidence"| REF["Deterministic refusal"]
    GATE -->|"strong evidence"| GEN["Generation failover:<br/>Groq, Mistral, Gemini<br/>17 s deadline"]
    GEN --> ANS["Sourced answer + JSONL log"]
    REF --> ANS
    ANS --> UI


The backend pipeline (`backend/src/pipeline.py`) rewrites the follow-up into a standalone search
query (capped at 4 s, falls back to the raw question), retrieves from per-university ChromaDB
collections, applies the university-specific relevance gate, and only then calls the LLM. The
generation prompt receives the **original** user wording, so answers stay natural while retrieval
gets the cleaner query. Provider failures (429/timeouts/zero rate limits) fail over instantly
within the remaining deadline.

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | ≥ 3.12 | backend |
| [uv](https://docs.astral.sh/uv/) | latest | recommended for env management (`pip` works too) |
| Node.js + npm | ≥ 20 / npm 11 | Angular 22 frontend |
| Disk | ~4 GB | models + ChromaDB (first run downloads ~2 GB of model weights from Hugging Face) |

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/RayanG03/code-Agent.git
cd code-Agent

# 1) Backend environment
cd backend
uv sync                       # creates backend/.venv from pyproject.toml
# pip fallback:  python -m venv .venv ; .venv\Scripts\pip install -e .

# 2) API keys — copy the template and fill in your own keys (never commit .env)
Copy-Item .env.example .env
notepad .env                  # GROQ_API_KEY / MISTRAL_API_KEY / GEMINI_API_KEY / DEEPINFRA_API_KEY

# 3) Build the vector store (first time, or after corpus updates)
uv run python -X utf8 -m src.ingest.run_kau    # qa + web + pdf collections
uv run python -X utf8 -m src.ingest.run_ksu    # qa + txt + pdf collections

# 4) Start both servers (backend :8000 + frontend :4200 + browser)
cd ..
python main.py
```

> **QA CSV datasets are gitignored.** `backend/data/` is not in the repository — before running
> the `qa` ingestion, copy `kau_qa.csv` and `ksu_qa.csv` from the team drive into
> `backend/data/`. Without them, ingest only the document collections:
> `run_kau --only web` / `--only pdf` and `run_ksu --only txt` / `--only pdf`.

**First run notes**

- Model weights (~2 GB: `intfloat/multilingual-e5-large` embeddings + `BAAI/bge-reranker-v2-m3`
  reranker) download from Hugging Face once, then load from the local cache.
- The `Warning: You are sending unauthenticated requests to the HF Hub…` message is **benign** —
  it only means lower anonymous rate limits. Silence it with a free `HF_TOKEN`, or set
  `HF_HUB_OFFLINE=1` once models are cached.
- Backend startup takes ~20–60 s (model loading); the launcher waits for `/health` before
  starting the frontend.

## Running the app

```powershell
python main.py                  # backend + frontend + open browser
python main.py --backend-only   # API only (Swagger docs at http://127.0.0.1:8000/docs)
python main.py --frontend-only  # ng serve only
python main.py --reload         # uvicorn autoreload while developing
python main.py --dry-run        # print resolved commands without starting
python main.py --no-browser --health-timeout 600
```

Prefer two terminals? Backend: `uvicorn src.main:app --port 8000` from `backend/`,
frontend: `npm start` from `frontend/`. The frontend hardcodes `http://localhost:8000`, so keep
the backend on port 8000.

## Tests and dev tools

Run from `backend/` (`uv run python -X utf8 -m …`):

| Command | Purpose |
|---|---|
| `python -X utf8 -m pytest` | full test suite (API layer, failover chain, rewrite, gates, ingest units) |
| `python -X utf8 -m src.failover_drill` | end-to-end provider-failover drill; every scenario must finish < 20 s |
| `python -X utf8 -m src.api_smoke` | live `/ask` smoke check against a running backend |
| `python -X utf8 -m src.generation_smoke` | verifies every configured LLM provider accepts our payloads |
| `python -X utf8 -m src.ksu_gate_eval` / `src.retrieval_eval` | retrieval/gate quality evaluation |
| `python -X utf8 -m src.kau_ref_snapshot --check` | citation-snapshot regression check for KAU sources |
| `python -X utf8 -m src.gate_verify` | calibration sanity for the refusal gates |

## Configuration

Secrets live in `backend/.env` (gitignored); every value can also be passed as a plain
environment variable. See `backend/.env.example` and `backend/src/config.py`.

| Key | Used for |
|---|---|
| `GROQ_API_KEY` | primary generator (`openai/gpt-oss-120b`) |
| `MISTRAL_API_KEY` | failover 1 (`mistral-small-latest`) |
| `GEMINI_API_KEY` | failover 2 (`gemini-flash-latest`) |
| `DEEPINFRA_API_KEY` | optional explicit providers (DeepSeek / Qwen / Llama — not in the default chain) |
| `HF_TOKEN` *(optional)* | higher Hugging Face rate limits |

Useful tuning knobs in `src/config.py` (defaults are calibrated — change deliberately):
`request_deadline_s` (17 s end-to-end), `query_rewrite_*` (history rewriting), `ksu_rerank_refuse_threshold`
(0.40), `relevance_floor_qa` / `relevance_floor_docs` (0.83 / 0.79), `embedding_model_name`.

## API reference

| Endpoint | Body | Response |
|---|---|---|
| `POST /ask` | `{"question": "...", "university": "kau"\|"ksu", "history": [{"question": "...", "answer": "..."}]}` | `{"answer": "..."}` (refusals are plain refusal text; `history` optional, malformed history is ignored — never a 400) |
| `POST /feedback` | `{"university": "...", "answer": "...", "feedback": "up"\|"down"}` | `{}` |
| `GET /universities` | — | `[{"id": "kau", "name": "جامعة الملك عبدالعزيز", "available": true}, …]` |
| `GET /health` | — | `{"status": "ok"}` |

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/ask -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"question": "متى يبدأ التسجيل للفصل الدراسي الثاني؟", "university": "ksu"}'
```

Interaction logs are appended to `backend/data/logs/ask.jsonl` and `feedback.jsonl` (gitignored)
with timings, provider/model used, and refusal flags — handy for the demo and for debugging.

## Corpus and re-ingestion

Official documents live in `backend/sources/` (in the repository):

```
backend/sources/
├── KAU/            # admission & regulations PDFs (Arabic) + captured web pages (markdown)
│   └── web/        # facilities, financial services, housing, student affairs, …
├── KSU/            # university documents (txt + PDFs: regulations, allowances)
└── Final-Dataset_KAU_Audit.xlsx
```

Each source type becomes its own ChromaDB collection (`kau_qa`, `kau_web`, `kau_pdf`,
`ksu_qa`, `ksu_txt`, `ksu_pdf`) under `backend/chroma_db/` (gitignored). After adding or updating
a source document, re-run the matching ingest command (see Quick start step 3) — no code changes
needed. Ingestion reports chunk counts and a p95 token-length check per collection.

## Project structure

```
code-Agent/
├── main.py                  # one-command launcher (backend + frontend, stdlib only)
├── backend/
│   ├── pyproject.toml       # Python 3.12 deps (uv)
│   ├── .env.example         # API-key template
│   ├── src/
│   │   ├── main.py          # FastAPI app: /ask, /feedback, /universities, /health
│   │   ├── pipeline.py      # rewrite → retrieve → gate → generate, deadline-aware
│   │   ├── retrieval.py / chroma_store.py / embeddings.py
│   │   ├── generators.py    # OpenAI-compatible providers + failover + rate-limit handling
│   │   ├── query_rewrite.py # follow-up → standalone query (deadline-capped)
│   │   ├── ksu_gate.py / context_assembly.py / prompts.py / config.py
│   │   ├── ingest/          # run_kau / run_ksu + per-source ingestors
│   │   └── failover_drill.py, api_smoke.py, …  # live verification tools
│   ├── tests/               # pytest suite
│   └── sources/             # official KAU / KSU documents (read-only corpus)
├── frontend/                # Angular 22 standalone app (Arabic RTL UI)
└── AGENTS.md                # working conventions for AI-assisted development
```

## Known limitations

- **The corpus is static.** If the official document doesn't cover a topic (e.g. KSU
  graduate/diploma admission requirements or semester registration dates), Murshid correctly
  **refuses** rather than guessing. Adding the source document and re-ingesting closes the gap.
- **UI request timeout.** If the backend takes longer than ~20 s (e.g. cold model load), the UI
  aborts the request and shows a canned demo message — check the `[backend]` logs or retry.
- **CPU inference.** Embedding/reranking run on CPU; a cold start takes tens of seconds and the
  end-to-end deadline is 17 s by design.
- **Windows-first.** Commands and the launcher are PowerShell-oriented; on macOS/Linux use the
  `bin/` venv paths and run `npm start` directly.