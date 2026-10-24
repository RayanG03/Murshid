"""Free OCR for any KAU/KSU PDF page via Gemini vision (rate-limit friendly).

src.data_prep.ocr_vision_check is hardcoded to 4 KAU PDFs and stops at
data/extracted/_diagnostics/ — it never publishes anywhere. This module
generalizes it to every KAU/KSU PDF and finishes the job: transcribed pages
land directly in sources/<UNI>/transcribed/<stem>/page_NN.txt, which is
exactly where the kau_pdf / ksu_pdf ingestors read from.

Per document:
1. render 300 DPI page PNGs into data/extracted/_diagnostics/<stem>/
   (cached — existing PNGs are kept)
2. Gemini-transcribe each page into page_NN.gemini.txt (cached; retry with
   exponential backoff; settings.gemini_request_delay_seconds throttle)
3. publish every transcribed page as sources/<UNI>/transcribed/<stem>/
   page_NN.txt — this overwrites the broken text-layer extraction for that
   document, which is the point.

Rate limits: every successfully transcribed page is cached, so a run killed
by a 429 can simply be re-run (only missing pages are fetched). If 429s keep
coming, raise GEMINI_REQUEST_DELAY_SECONDS in backend/.env (e.g. 8) and/or
continue the next day when the free-tier daily quota resets.

Usage (from backend/):
  uv run python -X utf8 -m src.data_prep.ocr_pages --list
  uv run python -X utf8 -m src.data_prep.ocr_pages KSU المكافئات
  uv run python -X utf8 -m src.data_prep.ocr_pages KAU "لائحة الدراسة" "تقويم"
  uv run python -X utf8 -m src.data_prep.ocr_pages KAU --all
  uv run python -X utf8 -m src.data_prep.ocr_pages KSU --all

--all skips دليل التحويل الداخلي (its verified legacy transcription already
exists in the diagnostics folder); naming it explicitly forces it.
"""

import base64
import sys
import time
from pathlib import Path

import fitz
import httpx

LEGACY_KAU_STEM = "دليل التحويل الداخلي"
DPI = 300

# Same literal-transcription contract as ocr_vision_check: verbatim copy,
# digits exactly as shown, [ILLEGIBLE] instead of guessing.
GEMINI_PROMPT = (
    "انسخ النص العربي الظاهر في هذه الصورة حرفيًا كما هو، سطرًا بسطر، من الأعلى "
    "إلى الأسفل ومن اليمين إلى اليسار، دون أي تلخيص أو إعادة صياغة أو تصحيح "
    "إملائي. انسخ جميع الأرقام (التواريخ، النسب المئوية، المعدلات، الساعات، "
    "الأرقام الجامعية) بدقة كما تظهر تمامًا. لا تضف أي نص غير موجود في الصورة "
    "ولا تشرح أو تحلل. إذا وُجدت منطقة غير واضحة لا تستطيع قراءتها بثقة "
    "فاكتب [ILLEGIBLE] مكانها بدلًا من التخمين. ابدأ الإجابة بالنص مباشرة."
)


def diag_dir(data_dir: Path, stem: str) -> Path:
    """Cache root for rendered PNGs and per-page Gemini transcriptions."""
    return data_dir / "extracted" / "_diagnostics" / stem


def discover_pdfs(sources_dir: Path, uni: str) -> list[Path]:
    """Top-level PDFs of one university (same discovery as the ingestors)."""
    return sorted((sources_dir / uni).glob("*.pdf"))


def resolve_docs(
    sources_dir: Path, uni: str, filters: list[str], include_all: bool
) -> list[Path]:
    """Pick the PDFs to OCR from --all or substring name filters."""
    pdfs = discover_pdfs(sources_dir, uni)
    if include_all:
        legacy = [p for p in pdfs if p.stem == LEGACY_KAU_STEM]
        if legacy:
            print(
                f"[skip] {uni}/{LEGACY_KAU_STEM}.pdf — verified legacy "
                "transcription exists (name it explicitly to force)"
            )
        return [p for p in pdfs if p.stem != LEGACY_KAU_STEM]
    if filters:
        pdfs = [p for p in pdfs if any(f in p.name for f in filters)]
    if not pdfs:
        raise SystemExit("No PDFs matched. Run with --list to see names.")
    return pdfs


def render_pages(pdf_path: Path, out_dir: Path) -> int:
    """Render every page to 300 DPI PNG (cached); return the page count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        matrix = fitz.Matrix(DPI / 72, DPI / 72)
        for i, page in enumerate(doc, start=1):
            out = out_dir / f"page_{i:02d}.png"
            if not out.exists():
                page.get_pixmap(matrix=matrix).save(out)
        return len(doc)
    finally:
        doc.close()


def gemini_transcribe(settings, image: bytes, label: str) -> str:
    """One Gemini vision call with retry + exponential backoff (loud fail)."""
    url = (
        f"{settings.gemini_base_url}/models/"
        f"{settings.gemini_model_id}:generateContent"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": GEMINI_PROMPT},
                {"inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image).decode("ascii"),
                }},
            ]
        }],
        "generationConfig": {"temperature": 0.0},
    }
    attempts = max(2, settings.llm_max_retries)
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            resp = httpx.post(
                url,
                headers={
                    "x-goog-api-key": settings.gemini_api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=settings.llm_timeout_seconds,
            )
            if resp.status_code == 200:
                body = resp.json()
                text = body["candidates"][0]["content"]["parts"][0]["text"]
                if text.strip():
                    return text
                last_err = "HTTP 200 but empty candidate text"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code == 503:
                    attempts = max(attempts, 6)
        except Exception as exc:  # noqa: BLE001 — mirror ocr_vision_check
            last_err = f"{type(exc).__name__}: {exc}"
        backoff = min(
            settings.gemini_retry_backoff_base_seconds * (2 ** (attempt - 1)),
            90.0,
        )
        print(
            f"[gemini] {label} attempt {attempt}/{attempts} FAILED "
            f"({last_err.splitlines()[0][:120]}) — backoff {backoff:.0f}s"
        )
        if attempt < attempts:
            time.sleep(backoff)
    raise SystemExit(
        f"GEMINI FAILED after {attempts} attempts on {label}: {last_err}\n"
        "Re-run the same command later — finished pages are cached."
    )


def transcribe_doc(
    settings, doc_label: str, out_dir: Path
) -> tuple[int, int]:
    """Gemini every missing page; return (api_calls_this_run, cached)."""
    calls = cached = 0
    for png in sorted(out_dir.glob("page_*.png")):
        out_txt = out_dir / (png.stem + ".gemini.txt")
        if out_txt.exists():
            cached += 1
            continue
        calls += 1
        text = gemini_transcribe(
            settings, png.read_bytes(), f"{doc_label}/{png.stem}"
        )
        out_txt.write_text(text, encoding="utf-8")
        print(f"[gemini] {doc_label}/{png.stem}: ok ({len(text)} chars)")
        time.sleep(settings.gemini_request_delay_seconds)
    return calls, cached


def publish_pages(out_dir: Path, transcribed_dir: Path) -> int:
    """Copy page_NN.gemini.txt results into the committed transcribed/ dir."""
    gem = sorted(out_dir.glob("page_*.gemini.txt"))
    if not gem:
        return 0
    transcribed_dir.mkdir(parents=True, exist_ok=True)
    for g in gem:
        target = transcribed_dir / (g.name[: -len(".gemini.txt")] + ".txt")
        target.write_text(g.read_text(encoding="utf-8"), encoding="utf-8")
    return len(gem)


def process_doc(settings, uni: str, pdf_path: Path) -> None:
    """Render -> transcribe -> publish one PDF, with a per-doc summary."""
    doc_label = f"{uni}/{pdf_path.name}"
    out_dir = diag_dir(settings.data_dir, pdf_path.stem)
    npages = render_pages(pdf_path, out_dir)
    calls, cached = transcribe_doc(settings, doc_label, out_dir)
    published = publish_pages(
        out_dir, settings.sources_dir / uni / "transcribed" / pdf_path.stem
    )
    transcribed_dir = settings.sources_dir / uni / "transcribed" / pdf_path.stem
    on_disk = len(list(transcribed_dir.glob("page_*.txt")))
    if on_disk != npages:
        print(
            f"[warn] {doc_label}: {on_disk} page files in transcribed/ but "
            f"the PDF has {npages} pages — stale files may need cleanup"
        )
    print(
        f"[done] {doc_label}: pages={npages} ocr_now={calls} "
        f"cached={cached} published={published} -> {transcribed_dir}"
    )


def list_docs(sources_dir: Path) -> None:
    """Show every KAU/KSU PDF with its transcription coverage."""
    for uni in ("KAU", "KSU"):
        print(f"\n=== {uni} ===")
        pdfs = discover_pdfs(sources_dir, uni)
        if not pdfs:
            print("  (no PDFs)")
            continue
        for pdf_path in pdfs:
            doc = fitz.open(pdf_path)
            try:
                npages = len(doc)
            finally:
                doc.close()
            transcribed_dir = (
                sources_dir / uni / "transcribed" / pdf_path.stem
            )
            done = len(list(transcribed_dir.glob("page_*.txt")))
            note = ""
            if pdf_path.stem == LEGACY_KAU_STEM:
                note = " (legacy transcription in diagnostics/)"
            print(
                f"  {pdf_path.name}: pdf_pages={npages} "
                f"transcribed={done}{note}"
            )


def main() -> None:
    """Dispatch: --list or OCR the selected documents."""
    from src.config import get_settings

    settings = get_settings()
    args = sys.argv[1:]
    if not args or args[0] == "--list":
        list_docs(settings.sources_dir)
        return
    uni = args[0]
    if uni not in {"KAU", "KSU"}:
        raise SystemExit("first argument must be KAU or KSU (or --list)")
    if not settings.gemini_api_key:
        raise SystemExit(
            "GEMINI_API_KEY is empty in backend/.env — add it and re-run."
        )
    include_all = "--all" in args
    filters = [a for a in args[1:] if not a.startswith("--")]
    docs = resolve_docs(settings.sources_dir, uni, filters, include_all)
    touched: set[str] = set()
    for pdf_path in docs:
        process_doc(settings, uni, pdf_path)
        touched.add(uni)
    for u in sorted(touched):
        print(
            f"\nNext step: uv run python -X utf8 -m src.ingest.run_{u.lower()} "
            f"--only pdf"
        )
    print(
        "Then re-check the refusal gates: kau_ref_snapshot --check, "
        "gate_verify, ksu_gate_eval"
    )


if __name__ == "__main__":
    main()
