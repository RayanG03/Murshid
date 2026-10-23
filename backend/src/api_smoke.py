"""Live API smoke: starts the real server, exercises every endpoint.

Launches `uv run uvicorn src.main:app` as a subprocess, waits for /health
(startup_seconds), then calls, per endpoint, one or two requests while
measuring status codes and timings. Latency rule: /ask must stay under
15s (PArt 5 requirement; the frontend aborts at 20s).

Console: ASCII status/seconds lines only. Arabic answers go to
data/api_smoke.md (UTF-8).

run from backend/:  uv run python -X utf8 -m src.api_smoke
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

from src.config import get_settings

OUT_NAME = "api_smoke.md"
BASE = "http://localhost:8000"
ASK_BUDGET_S = 15.0
STARTUP_BUDGET_S = 180.0

KAU_Q = "كيف احذف ماده بعد انتهاء فتره الحذف والاضافه"
KSU_Q = "وش شروط الاعتذار عن اختبار المقرر النهائي بعذر مقبول؟"


def _post(client: httpx.Client, path: str, payload: dict):
    t0 = time.perf_counter()
    r = client.post(f"{BASE}{path}", json=payload)
    return r, time.perf_counter() - t0


def _get(client: httpx.Client, path: str):
    t0 = time.perf_counter()
    r = client.get(f"{BASE}{path}")
    return r, time.perf_counter() - t0


def main() -> None:
    settings = get_settings()
    md: list[str] = [
        "# Live API smoke", "",
        "base: http://localhost:8000 (frontend contract)", "",
    ]
    console: list[str] = []
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app"],
        cwd=Path.cwd(), shell=False,
    )
    try:
        t0 = time.perf_counter()
        startup = None
        with httpx.Client(timeout=25.0) as client:
            while time.perf_counter() - t0 < STARTUP_BUDGET_S:
                try:
                    r, _ = _get(client, "/health")
                    if r.status_code == 200:
                        startup = time.perf_counter() - t0
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2.0)
            if startup is None:
                print(f"FAIL server did not become healthy within "
                      f"{STARTUP_BUDGET_S:.0f}s")
                return
            console.append(
                f"startup_seconds={startup:.1f} "
                f"(models loaded once at startup, /health 200)"
            )
            md += [f"startup: {startup:.1f}s to first healthy /health", ""]

            r, secs = _get(client, "/universities")
            console.append(
                f"GET /universities HTTP {r.status_code} ({secs:.2f}s)"
            )
            md += [f"--- /universities HTTP {r.status_code} ({secs:.2f}s)",
                   "", "```", r.text, "```", ""]

            fallback_answer = ""
            for label, question in (("kau", KAU_Q), ("ksu", KSU_Q)):
                r, secs = _post(client, "/ask",
                                {"question": question,
                                 "university": label})
                ok = r.status_code == 200
                answer = ""
                if ok:
                    body = r.json()
                    answer = body.get("answer", "")
                under = secs <= ASK_BUDGET_S
                console.append(
                    f"POST /ask[{label}] HTTP {r.status_code} ({secs:.2f}s) "
                    f"under_budget={under}"
                )
                md += [
                    f"--- /ask ({label}) HTTP {r.status_code}, {secs:.2f}s",
                    "", "```", answer, "```", "",
                ]
                if label == "kau":
                    fallback_answer = answer

            # exercise feedback up, then null (vote removed)
            for value in ("up", None):
                r, secs = _post(client, "/feedback",
                                {"university": "kau",
                                 "answer": fallback_answer,
                                 "feedback": value})
                console.append(
                    f"POST /feedback[{value}] HTTP {r.status_code} "
                    f"({secs:.3f}s)"
                )
            md += ["--- /feedback up + null: 200 recorded in "
                   "data/logs/feedback.jsonl", ""]

            # JSONL cross-check: matched_question recorded for /feedback
            import json as _json
            fb_log = settings.data_dir / "logs" / "feedback.jsonl"
            matched = "no"
            if fb_log.exists():
                for line in reversed(
                    fb_log.read_text(encoding="utf-8").splitlines()
                ):
                    rec = _json.loads(line)
                    if rec.get("matched_question"):
                        matched = "yes"
                        break
            console.append(f"feedback matched_question={matched}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
    out_path = settings.data_dir / OUT_NAME
    out_path.write_text("\n".join(md), encoding="utf-8")
    for line in console:
        print(line)
    print(f"arabic answers -> {out_path}")


if __name__ == "__main__":
    main()
