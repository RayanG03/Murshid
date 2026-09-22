# Capstone II

Arabic RAG question-answering system over KSU and KAU academic regulations.

## Repo layout (this file lives in `code-Agent/`)
- `backend/`: the Python backend. This is the ONLY place you write code.
  - `backend/sources/`: source data. READ-ONLY. Read from it; never modify, rename, move, or
    delete anything in it. Generated files go elsewhere (see below).
  - `backend/.venv/`: existing virtual environment. Use it via `uv`; do not recreate it, and
    do not commit it.
- `frontend/`: Angular app, already built. NEVER modify anything in it. You MAY read it, and
  should, to confirm the API contract (see below).

## Stack
- Python 3.14, managed with `uv` (never pip, poetry, or conda)
- FastAPI + uvicorn
- ChromaDB, separate collections per university and source type
- Embeddings: `intfloat/multilingual-e5-large` via sentence-transformers (CPU only)
- Generators: DeepSeek, Qwen, Llama via DeepInfra; Gemini via free tier
- Windows environment: use paths and commands that work on Windows

## Commands (run from `code-Agent/backend/`)
- Add a dependency: `uv add <package>`
- Run server: `uv run uvicorn src.main:app --reload`
- Run tests: `uv run pytest`

## API contract (confirmed against frontend/src/app/app.ts)
- POST /ask: {question, university} -> {answer}
  - university is "ksu" | "kau"
  - frontend also accepts a "response" key as a fallback for "answer"; always send "answer"
- POST /feedback: {university, answer, feedback} -> 200 OK, body not read by frontend
  - answer is the FULL answer text (not an ID) — match feedback to a question by this text
  - feedback is "up" | "down" | null (null means the user un-clicked their vote)
- GET /universities: not currently called by the frontend (it hardcodes ksu/kau locally).
  Build it anyway per the contract, but it won't be exercised until the frontend is updated.
- Base URL is hardcoded in the frontend as http://localhost:8000. CORS must allow
  http://localhost:4200 (Angular dev server default).
- KNOWN ISSUE: if a request to /ask fails or times out (20s), the frontend silently shows a
  canned demo answer instead of an error. This will mask real backend failures during testing.
  Flag this to me; do not "fix" it yourself without asking.

## Hard rules
- Load the embedding model ONCE at startup via the FastAPI lifespan handler. Never lazy-load.
- E5 needs prefixes: "query: " for questions, "passage: " for documents.
- Every LLM call needs retry with exponential backoff AND a per-request progress log line.
- Retrieval priority: verified Q&A pairs first, then regulation document chunks.
- All model IDs and settings live in `backend/src/config.py`. Gemini ID is `gemini-flash-latest`
  (`gemini-2.0-flash` is deprecated and returns 404).
- API keys come from environment variables (`backend/.env`, gitignored). Never commit secrets.
- UTF-8 everywhere. Data and prompts are Arabic.
- NEVER fetch, scrape, or download anything. All source data is already in `backend/sources/`.
  Do not call Firecrawl or any web API for data. If something is missing, tell me; do not
  invent it.

## Data (read-only, `backend/sources/`)
- `Final-Dataset_KAU_Audit.xlsx`: 574-row audit of the KSU Q&A dataset against KAU regulations.
- KAU web pages as markdown (`KAU/web/`): financial_services, student_affairs,
  university_housing, facilities_and_services, health_services, mykau_app,
  visitor_student_and_transfer.
- KAU PDFs (`KAU/`):
  - تقويم 26-27.pdf — academic calendar 2026-27
  - دليل التحويل الداخلي.pdf — internal transfer guide
  - دليل القبول لبرامج البكالوريوس والدبلوم.pdf — bachelor's/diploma admission guide
  - دليل القبول للتحويل الخارجي إلى الجامعة.pdf — external transfer admission guide
  - لائحة الدراسة والاختبارات.pdf — study and examinations regulation
  - لائحة الطالب الجامعي الحقوق والواجبات.pdf — student rights and duties regulation
  - لائحة ضبط سلوك الطلاب والطالبات.pdf — student conduct regulation
  # TODO: confirm which of these map to which "المصدر الرسمي" rows in the workbook
- KSU sources (`KSU/`):
  - university documents.txt — # TODO: what is this (plain text version of what?)
  - university documents(pdf).pdf — # TODO: same content as the .txt, or different?
  - اللائحة التنظيمية للشؤون الدراسية.pdf — academic affairs regulation
  - اللائحة التنظيمية للشؤون الدراسية24.pdf — academic affairs regulation (2024 version? or a
    duplicate/updated copy of the one above — # TODO: confirm which is current)
  - لائحة المكافئات.pdf — rewards/stipends regulation

## Generated files (write here, never into `backend/sources/`)
- `backend/data/`: derived datasets (for example `kau_qa.csv`, `kau_unverified.csv`)
- `backend/chroma_db/`: the vector store
Both are gitignored.

## KAU Q&A rules (workbook sheet "مراجعة عبدالعزيز")
Column "حالة المطابقة مع جامعة الملك عبدالعزيز" decides how each row is used:
- "مطابق بالمعنى" (225 rows): use the original `Answer`.
- "غير مطابق" (146 rows): use column "إجابة جامعة الملك عبدالعزيز" INSTEAD of `Answer`.
- "غير موثّق رسميًا" (203 rows): NOT verified for KAU. Do NOT load into the verified KAU Q&A
  collection. Write them to `backend/data/kau_unverified.csv` and leave them out of retrieval
  and out of evaluation ground truth.
The KAU collection therefore has 371 verified pairs. Print these counts after building it.
Keep `Q_ID` and the source URL ("المصدر الرسمي") on every record for citation.

## Markdown source files
Each file has a YAML header (source_url, page_title, last_updated_on_page, retrieved, method,
sometimes note/linked_pages_not_captured), then an H1, then `##` sections.
- Chunk by `##` section. If a section is long, split by paragraph, but keep the heading attached.
- Prepend "page_title > section heading" to every chunk so it makes sense on its own.
- Store source_url and section heading in each chunk's metadata for citations.
- Do NOT edit these files. Do NOT normalize digits inside them, do NOT correct typos (for
  example a space in an email address or a stray "/"), and do NOT fill in `retrieved:`.
- Digit normalization (Arabic-Indic to Western) is allowed ONLY when building the search text
  for a chunk. Keep the original text for display.
- Some sections end with a repeated contact block. Keep it attached to its section; never
  store it as a separate chunk.

## Known gaps (do not invent answers for these)
- The bus timetable on the facilities page is an image; the text has no schedule.
- The housing, health, and student affairs pages are overviews. The detail pages they link to
  were not captured (see `linked_pages_not_captured` in each header).
- The MyKAU app page has no real per-service descriptions (placeholders were excluded).
- If a question depends on missing content, the system must say it cannot confirm, not guess.

## PDF extraction
Use PyMuPDF (`fitz`). Arabic PDFs can extract with reversed words, disconnected letters, or
empty pages. For EACH PDF, print page count, total characters, 10 sample Arabic lines, and every
page under 50 characters. Stop and show me. Do not continue if text is reversed or garbled.
Check that numbers survive extraction (GPA values, credit hours, percentages).

## Build stages
Work on ONE stage at a time. After each stage: stop, tell me exactly how to verify it, and wait
for my approval.
1. Skeleton: `backend/src/` structure, `pyproject.toml` via uv (reuse the existing `.venv`),
   `src/config.py`, and the frontend API-contract report
2. Data prep: list `backend/sources/`, build KAU Q&A (rules above), extract PDFs, print reports
3. Retrieval: ingest into ChromaDB; verify Hit@3 before moving on
4. Generation: provider-agnostic interface, all four models swappable
5. API: /ask, /feedback, /universities, plus CORS for the frontend's origin
6. Evaluation: run the dataset through multiple generators and log results

## Conventions
- Type hints on all functions. Small modules, one responsibility each.
- English docstrings and comments.
- Ask before adding any new dependency.
- Do not rewrite working code from an earlier stage unless I ask.