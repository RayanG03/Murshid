# KAU PDF transcriptions — layout and rules

This folder makes the KAU PDF corpus rebuildable from a fresh clone. Every
PDF in `backend/sources/KAU/` gets one subfolder named after the PDF file
stem, containing one UTF-8 text file per page:

```
backend/sources/KAU/transcribed/
└── <pdf-stem>/
    ├── page_01.txt
    ├── page_02.txt
    └── ...
```

Rules:

- **Committed wins.** The ingestor (`src.ingest.kau_pdf`) prefers
  `transcribed/<stem>/` for every document. Only `دليل التحويل الداخلي` has
  a legacy fallback (the human-verified Gemini transcription under
  `backend/data/extracted/_diagnostics/`, which is gitignored) used when this
  folder has no pages for it.
- **Blank lines matter.** The ingestor groups paragraph blocks on blank
  lines — keep them when editing or OCR-ing pages.
- **Naming is fixed.** Files must be `page_NN.txt` (the page number is parsed
  from the filename; pages are sorted numerically).

## Producing the pages

1. **Text-layer PDFs** (born-digital — most likely for recent documents):

   ```powershell
   cd backend
   uv run python -X utf8 -m src.data_prep.extract_kau_pages
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

3. **Ingest** (rebuild the `kau_pdf` collection):

   ```powershell
   uv run python -X utf8 -m src.ingest.run_kau --only pdf
   ```

   The ingest log lists every skipped document that still has no
   transcription. After a big transcription change, re-check the gates:
   `src.kau_ref_snapshot --check`, `src.gate_verify`, `src.retrieval_eval`.

Note: `دليل التحويل الداخلي` is skipped by the extractor by default — its
verified transcription already exists; pass `--all` to force.
