# KSU PDF transcriptions — layout and rules

This folder makes the KSU PDF corpus rebuildable from a fresh clone. Every
PDF in `backend/sources/KSU/` gets one subfolder named after the PDF file
stem, containing one UTF-8 text file per page:

```
backend/sources/KSU/transcribed/
└── <pdf-stem>/
    ├── page_01.txt
    ├── page_02.txt
    └── ...
```

Rules:

- **Line-based layout.** Unlike the KAU folder, blank lines are NOT
  meaningful here — the KSU ingestor works line-by-line, so each page file
  holds only its non-empty lines, one per line. Cover/TOC pages may be
  empty files; keep the numbering intact.
- **All pages are committed** (front matter included). The ingestor itself
  skips the 2024 regulation's 6 front-matter pages at ingest time.
- **Committed wins.** The ingestor (`src.ingest.ksu_pdf`) prefers
  `transcribed/<stem>/` for every document and falls back to reading the
  PDF text layer directly when this folder has no pages for it — so
  re-ingesting works even before every document is transcribed here.
- **Naming is fixed.** Files must be `page_NN.txt` (the page number is
  parsed from the filename; pages are sorted numerically).

## Producing the pages

1. **Text-layer PDFs** (born-digital):

   ```powershell
   cd backend
   uv run python -X utf8 -m src.data_prep.extract_ksu_pages
   ```

   The report marks each document `text-layer OK` or `OCR CANDIDATE`
   (no Arabic text, garbled lam-alef, or many near-empty pages).

2. **Scanned PDFs** (`OCR CANDIDATE`): OCR each page with any tool and save
   the text as `page_NN.txt` in the document's folder here. Free options:
   - **Surya** (open-source, local, strong Arabic): `uv add surya-ocr`, then
     recognize each page image and paste the text lines.
   - **Mistral OCR** (paid, ~$4/1k pages): send the whole PDF once, split the
     per-page markdown into `page_NN.txt` files.
   - **Gemini free tier**: one whole-PDF request per document (not per page —
     per-page requests are what exhausted the rate limits last time).

   Re-run the extractor with `--overwrite` only for text-layer refreshes;
   OCR output should be written by the OCR tooling directly.

3. **Ingest** (rebuild the `ksu_pdf` collection):

   ```powershell
   uv run python -X utf8 -m src.ingest.run_ksu --only pdf
   ```

   The ingest log prints one line per document (mode, text source, page and
   chunk counts) and warns loudly about any document with no text at all.
   After a big transcription change, re-check the KSU gates:
   `src.ksu_gate_eval`, `src.gate_verify`.
