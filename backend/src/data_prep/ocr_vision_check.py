"""Step E diagnostics: Tesseract OCR vs Gemini vision transcription.

Stages (run in order; each is idempotent and writes only to
backend/data/extracted/_diagnostics/):
- render: rasterize each page of the 4 corrupted KAU PDFs at 300 DPI
- ocr:    Tesseract (ara) per page, per-word confidences kept
- gemini: Gemini gemini-flash-latest transcription per page, with retry +
          exponential backoff + per-request progress logs (needs GEMINI_API_KEY)
- compare: cross-check numbers/lines between the two, classify regions as
           HIGH CONFIDENCE / FLAGGED / UNRESOLVED, write markdown report

Run from backend/:
  uv run python -X utf8 -m src.data_prep.ocr_vision_check render|ocr|gemini|compare
"""

import base64
import csv
import io
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import httpx

CORRUPTED_PDFS = [
    "تقويم 26-27.pdf",
    "دليل التحويل الداخلي.pdf",
    "دليل القبول لبرامج البكالوريوس والدبلوم.pdf",
    "دليل القبول للتحويل الخارجي إلى الجامعة.pdf",
]

GEMINI_PROMPT = (
    "انسخ النص العربي الظاهر في هذه الصورة حرفيًا كما هو، سطرًا بسطر، من الأعلى "
    "إلى الأسفل ومن اليمين إلى اليسار، دون أي تلخيص أو إعادة صياغة أو تصحيح "
    "إملائي. انسخ جميع الأرقام (التواريخ، النسب المئوية، المعدلات، الساعات، "
    "الأرقام الجامعية) بدقة كما تظهر تمامًا. لا تضف أي نص غير موجود في الصورة "
    "ولا تشرح أو تحلل. إذا وُجدت منطقة غير واضحة لا تستطيع قراءتها بثقة "
    "فاكتب [ILLEGIBLE] مكانها بدلًا من التخمين. ابدأ الإجابة بالنص مباشرة."
)

NUM_RE = re.compile(r"\d+(?:[.,/:\-]\d+)*")
LINE_AGREE_RATIO = 0.75
STRUCTURAL_TOLERANCE_FRAC = 0.2


def diag_root() -> Path:
    from src.config import get_settings
    return get_settings().data_dir / "extracted" / "_diagnostics"


def pdf_dir(stem: str) -> Path:
    return diag_root() / stem


@dataclass
class PageOcr:
    lines: list[str] = field(default_factory=list)
    word_confidences: list[float] = field(default_factory=list)
    avg_line_conf: dict[int, float] = field(default_factory=dict)


def stage_render() -> None:
    """Render all pages of the 4 PDFs to 300 DPI PNGs."""
    from src.config import get_settings
    settings = get_settings()
    matrix = fitz.Matrix(300 / 72, 300 / 72)
    for name in CORRUPTED_PDFS:
        pdf_path = settings.sources_dir / "KAU" / name
        out_dir = pdf_dir(pdf_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            out = out_dir / f"page_{i:02d}.png"
            if not out.exists():
                page.get_pixmap(matrix=matrix).save(out)
            print(f"[render] {name} page {i}/{len(doc)} -> {out.name}")
        doc.close()


def _parse_tsv(tsv_text: str) -> PageOcr:
    """Reconstruct lines and confidences from tesseract TSV output."""
    result = PageOcr()
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    lines: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for row in reader:
        if row["level"] != "5":
            continue
        text = (row["text"] or "").strip()
        if not text:
            continue
        try:
            conf = float(row["conf"])
        except ValueError:
            conf = -1.0
        key = (row["block_num"], row["par_num"], row["line_num"])
        lines.setdefault(key, []).append((text, conf))
        result.word_confidences.append(conf)
    for idx, key in enumerate(sorted(lines, key=lambda k: tuple(map(int, k)))):
        words = lines[key]
        result.lines.append(" ".join(w for w, _ in words))
        result.avg_line_conf[idx] = (
            sum(c for _, c in words) / len(words) if words else -1.0
        )
    return result


def stage_ocr() -> None:
    """Run Tesseract (ara) on every rendered page (image via stdin)."""
    from src.config import get_settings
    tess = str(get_settings().tesseract_path)
    for name in CORRUPTED_PDFS:
        out_dir = pdf_dir(Path(name).stem)
        pages = sorted(out_dir.glob("page_*.png"))
        for png in pages:
            proc = subprocess.run(
                [tess, "stdin", "stdout", "-l", "ara", "tsv"],
                input=png.read_bytes(), capture_output=True, timeout=300,
            )
            if proc.returncode != 0:
                raise SystemExit(
                    f"tesseract failed on {png.name} of {name}: "
                    f"{proc.stderr.decode('utf-8', 'replace')[:300]}"
                )
            tsv = proc.stdout.decode("utf-8-sig", "replace")
            ocr = _parse_tsv(tsv)
            txt_out = out_dir / (png.stem + ".tesseract.txt")
            txt_out.write_text("\n".join(ocr.lines), encoding="utf-8")
            low = sum(
                1 for c in ocr.word_confidences
                if c < get_settings().ocr_low_confidence_threshold
            )
            print(f"[ocr] {name} {png.stem}: lines={len(ocr.lines)} "
                  f"words={len(ocr.word_confidences)} low_conf_words={low}")


def _gemini_call(
    settings, image: bytes, page_label: str
) -> str:
    """One Gemini generateContent call with retry + exponential backoff."""
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
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                print(f"[gemini] {page_label} attempt {attempt}: ok "
                      f"({len(text)} chars)")
                return text
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code == 503:
                attempts_this = max(attempts, 6)
                if attempt < attempts_this:
                    attempts = attempts_this
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        backoff = min(
            settings.gemini_retry_backoff_base_seconds * (2 ** (attempt - 1)),
            90.0,
        )
        print(f"[gemini] {page_label} attempt {attempt}/{attempts} FAILED "
              f"({last_err.splitlines()[0][:120]}) — backoff {backoff:.0f}s")
        if attempt < attempts:
            time.sleep(backoff)
    raise SystemExit(f"GEMINI FAILED after {attempts} retries on {page_label}: "
                     f"{last_err}. Stopping (no silent skips).")


def stage_gemini() -> None:
    """Transcribe every rendered page with Gemini; no silent skips."""
    from src.config import get_settings
    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is empty in backend/.env — add it "
                         "and re-run: uv run python -X utf8 -m "
                         "src.data_prep.ocr_vision_check gemini")
    calls = 0
    for name in CORRUPTED_PDFS:
        out_dir = pdf_dir(Path(name).stem)
        for png in sorted(out_dir.glob("page_*.png")):
            out_txt = out_dir / (png.stem + ".gemini.txt")
            if out_txt.exists():
                print(f"[gemini] {png.stem}: cached, skipping")
                continue
            calls += 1
            text = _gemini_call(settings, png.read_bytes(), f"{name}/{png.stem}")
            out_txt.write_text(text, encoding="utf-8")
            time.sleep(settings.gemini_request_delay_seconds)
    print(f"[gemini] total API calls this run: {calls}")


def _normalize_line(line: str) -> str:
    lowered = re.sub(r"[\u0640\s\u064B-\u065F]+", "", line)
    return re.sub(r"[^\u0621-\u064A0-9]", "", lowered)


def _compare_page(png: Path) -> dict:
    """Cross-check tesseract vs gemini output for one page."""
    tess_txt = (png.parent / (png.stem + ".tesseract.txt")).read_text("utf-8")
    gem_txt = (png.parent / (png.stem + ".gemini.txt")).read_text("utf-8")
    import difflib

    t_nums = NUM_RE.findall(tess_txt)
    g_nums = NUM_RE.findall(gem_txt)
    from collections import Counter
    t_count, g_count = Counter(t_nums), Counter(g_nums)
    nums_high = sorted((t_count & g_count).elements())
    nums_flagged = []
    for n in (t_count - g_count).elements():
        nums_flagged.append(("tesseract-only", n))
    for n in (g_count - t_count).elements():
        nums_flagged.append(("gemini-only", n))

    t_lines = [l for l in tess_txt.splitlines() if l.strip()]
    g_lines = [l for l in gem_txt.splitlines() if l.strip()]
    g_norm = [(l, _normalize_line(l)) for l in g_lines]
    flagged_lines, agreed_lines = [], 0
    for tl in t_lines:
        tn = _normalize_line(tl)
        if not tn:
            continue
        best, best_r = None, 0.0
        for gl, gn in g_norm:
            if not gn:
                continue
            r = difflib.SequenceMatcher(None, tn, gn).ratio()
            if r > best_r:
                best, best_r = gl, r
        if best is not None and best_r >= LINE_AGREE_RATIO:
            agreed_lines += 1
        else:
            flagged_lines.append((tl, best if best is not None else "<none>",
                                  round(best_r, 2)))

    unresolved = gem_txt.count("[ILLEGIBLE]")
    structural = abs(len(t_lines) - len(g_lines)) > max(
        2, STRUCTURAL_TOLERANCE_FRAC * max(len(t_lines), len(g_lines))
    )
    return {
        "png": png,
        "nums_high": nums_high,
        "nums_flagged": nums_flagged,
        "flagged_lines": flagged_lines,
        "agreed_lines": agreed_lines,
        "t_lines": len(t_lines),
        "g_lines": len(g_lines),
        "illegible": unresolved,
        "structural": structural,
    }


def stage_compare() -> None:
    """Write the full markdown cross-check report + console summary."""
    lines_out: list[str] = ["# OCR vs Gemini cross-check report", ""]
    for name in CORRUPTED_PDFS:
        out_dir = pdf_dir(Path(name).stem)
        pages = sorted(out_dir.glob("page_*.png"))
        f_high = f_flag = f_unres = 0
        lines_out.append(f"\n## {name} ({len(pages)} pages)\n")
        for png in pages:
            if not (png.parent / (png.stem + ".gemini.txt")).exists():
                continue
            r = _compare_page(png)
            f_high += len(r["nums_high"]) + r["agreed_lines"]
            f_flag += len(r["nums_flagged"]) + len(r["flagged_lines"])
            f_unres += r["illegible"]
            lines_out.append(f"### {png.name}")
            lines_out.append(f"- image: `{png}`")
            lines_out.append(
                f"- lines: tesseract={r['t_lines']} gemini={r['g_lines']} "
                f"agreed={r['agreed_lines']}"
                + ("  **STRUCTURAL MISMATCH**" if r["structural"] else "")
            )
            if r["illegible"]:
                lines_out.append(f"- UNRESOLVED: {r['illegible']} [ILLEGIBLE] marks")
            if r["nums_flagged"]:
                lines_out.append("- FLAGGED numbers:")
                for side, n in r["nums_flagged"]:
                    lines_out.append(f"  - {side}: `{n}`")
            if r["flagged_lines"]:
                lines_out.append("- FLAGGED lines (tesseract | gemini | ratio):")
                for tl, gl, ratio in r["flagged_lines"]:
                    lines_out.append(f"  - `{tl[:120]}` | `{gl[:120]}` | {ratio}")
            lines_out.append("")
        print(f"[compare] {name}: high={f_high} flagged={f_flag} "
              f"unresolved={f_unres} pages={len(pages)}")
    report = diag_root() / "crosscheck_report.md"
    report.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"[compare] full report -> {report}")


def main() -> None:
    """Dispatch to the requested stage."""
    if len(sys.argv) < 2 or sys.argv[1] not in {
        "render", "ocr", "gemini", "compare",
    }:
        raise SystemExit("usage: ... render|ocr|gemini|compare")
    {"render": stage_render, "ocr": stage_ocr,
     "gemini": stage_gemini, "compare": stage_compare}[sys.argv[1]]()


if __name__ == "__main__":
    main()
