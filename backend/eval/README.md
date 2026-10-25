# Evaluation set

`questions.csv` is a held-out test set: questions are written the way
students type them (dialect, paraphrases), NOT copied from the indexed
Q&A pairs, plus questions that MUST be refused (off-topic or not covered
by the corpus). It is a seed set — review it and extend it (aim for 100+
rows, from real student questions where possible).

Columns:
- `expect`: `answer` or `refuse`
- `expect_any`: `|`-separated keywords; a retrieval HIT means at least one
  of them appears in the evidence sent to the model
- `history_question` / `history_answer`: optional previous turn (follow-up
  rows; only evaluated with `--generate`, the rewrite needs an LLM)
- `note`: why the row exists

`kau_r01` (bus timetable) is a known gate pass: the bus-service chunks
reach the model but the timetable itself is an image, so only the LLM can
refuse it — in gate-only runs it shows up as "answered but should refuse".

Run from `backend/` (see `src/eval_suite.py` for all options):

    python -m uv run python -X utf8 -m src.eval_suite --matrix
    python -m uv run python -X utf8 -m src.eval_suite --university ksu --generate
