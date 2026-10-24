"""Failover drill + first-request latency (read-only task except this file).

Starts the REAL app the way uvicorn does (startup loads the models once)
in a subprocess, once per scenario, passing key overrides as PROCESS
ENVIRONMENT VARIABLES (pydantic-settings: env beats .env; the .env file
on disk is never touched). Using the REAL providers:

  1. Normal chain                    (expect groq)
  2. GROQ key invalid in-process     (expect failover to mistral)
  3. GROQ + MISTRAL keys invalid     (expect gemini)
  4. All three keys invalid          (expect a clear 502 listing outcomes)

Per ask (one KAU + one KSU question) it prints: which provider answered,
attempts (provider, outcome, seconds) parsed from the server's
"[failover]" progress logs, total seconds, and whether the 14s generation
budget held (stage seconds from data/logs/ask.jsonl). Scenario 1
additionally compares the FIRST /ask after startup against later ones
(cold-start penalty; the frontend aborts at 20s).

Console: numbers only (ASCII). Arabic answers go to
data/failover_drill.md (UTF-8).

run from backend/:  uv run python -X utf8 data/failover_drill.py
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.config import get_settings  # noqa: E402

BASE = "http://localhost:8000"
OUT_NAME = "failover_drill.md"
LOGCONF_NAME = ".drill_logconf.json"
ASK_TIMEOUT_S = 25.0          # mirrors the frontend abort (20s) + margin
STARTUP_BUDGET_S = 180.0
GENERATION_BUDGET_S = 14.0

# uvicorn's default root logging swallows app-only INFO lines; our
# happy-path "[failover] ... attempt 1: ok (…s)" progress lines are INFO,
# so the drill forces a dictConfig with root=INFO for the server process.
LOGCONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(levelname)s %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
    },
}

FAIL_OK = re.compile(
    r"\[failover\] (\w+) answer\[.*? attempt (\d+): ok \((\d+) chars, ([\d.]+)s\)"
)
FAIL_BAD = re.compile(
    r"\[failover\] (\w+) answer\[.*? attempt (\d+) FAILED \(([\w.]+), ([\d.]+)s\)"
)
REJECT_AUTH = re.compile(r"\[failover\] (\w+) REJECTED AUTH")
AUTH_SKIPPED = re.compile(r"(\w+)=(auth_skipped)")

SCENARIOS: list[tuple[str, dict[str, str]]] = [
    ("1-normal-chain", {}),
    ("2-groq-invalid-key", {"GROQ_API_KEY": "drill-invalid-groq-key"}),
    (
        "3-groq-mistral-invalid",
        {
            "GROQ_API_KEY": "drill-invalid-groq-key",
            "MISTRAL_API_KEY": "drill-invalid-mistral-key",
        },
    ),
    (
        "4-all-three-invalid",
        {
            "GROQ_API_KEY": "drill-invalid-groq-key",
            "MISTRAL_API_KEY": "drill-invalid-mistral-key",
            "GEMINI_API_KEY": "drill-invalid-gemini-key",
        },
    ),
]


class LogTailer:
    """Reads the subprocess output in a thread; sliceable by position."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._lines: list[str] = []
        self._lock = threading.Lock()

        def _pump() -> None:
            for raw in iter(self._proc.stdout.readline, ""):
                if not raw:
                    break
                with self._lock:
                    self._lines.append(raw.rstrip())

        self._thread = threading.Thread(target=_pump, daemon=True)
        self._thread.start()

    def since(self, pos: int) -> tuple[list[str], int]:
        with self._lock:
            return list(self._lines[pos:]), len(self._lines)


def _attempts_of(new_lines: list[str]) -> list[tuple[str, str, float]]:
    """(provider, outcome+seconds) tuples, in order, from failover lines."""
    attempts: list[tuple[str, str, float]] = []
    for line in new_lines:
        m = FAIL_OK.search(line)
        if m:
            attempts.append((m.group(1), "ok", float(m.group(4))))
            continue
        m = FAIL_BAD.search(line)
        if m:
            attempts.append((m.group(1), m.group(3), float(m.group(4))))
    return attempts


def _auth_notes_of(new_lines: list[str]) -> list[str]:
    notes: list[str] = []
    for line in new_lines:
        m = REJECT_AUTH.search(line)
        if m:
            notes.append(f"{m.group(1)}:rejected_auth")
        m = AUTH_SKIPPED.search(line)
        if m:
            notes.append(f"{m.group(1)}:{m.group(2)}")
    return notes


def _last_ask_record(
    university: str, question: str, scenario_start_utc: datetime,
) -> dict | None:
    """Latest ask.jsonl entry for this question, from THIS scenario only."""
    path = BACKEND_DIR / "data" / "logs" / "ask.jsonl"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("question") != question or (
            rec.get("university") != university
        ):
            continue
        # Reject records from an earlier scenario/server process: stale
        # per-stage seconds would otherwise leak into this row.
        try:
            rec_ts = datetime.fromisoformat(rec["timestamp"])
        except (KeyError, ValueError):
            continue
        if (
            rec_ts.tzinfo is not None
            and rec_ts >= scenario_start_utc  # zero margin: scenario-scoped
        ):
            return rec
        return None
    return None


def _wait_healthy(client: httpx.Client) -> float | None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < STARTUP_BUDGET_S:
        try:
            if client.get(f"{BASE}/health").status_code == 200:
                return time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    return None


def _budget_held(rep: dict) -> bool:
    """Held = generation stage fit in 14s; failed requests: fast verdict."""
    gen = rep.get("generation_seconds")
    if gen is not None:
        return gen <= GENERATION_BUDGET_S + 0.05
    return rep["http_seconds"] < GENERATION_BUDGET_S


def main() -> None:
    get_settings()
    questions = {
        "kau": "كيف احذف ماده بعد انتهاء فتره الحذف والاضافه",
        "ksu": "وش شروط الاعتذار عن اختبار المقرر النهائي بعذر مقبول؟",
    }
    md: list[str] = ["# Failover drill (real providers)", ""]

    logconf_path = BACKEND_DIR / "data" / LOGCONF_NAME
    logconf_path.write_text(
        json.dumps(LOGCONF, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    for name, overrides in SCENARIOS:
        env = os.environ.copy()
        env.update(overrides)
        print(f"=== {name} (key overrides: {sorted(overrides) or 'none'})")
        md += [
            f"## {name}", "", f"key overrides: {sorted(overrides) or 'none'}",
            "",
        ]
        scenario_start_utc = datetime.now(timezone.utc)
        # Child stdout is a pipe: without this, its text-mode codec on
        # Windows is cp1252 and Arabic log records raise UnicodeEncodeError
        # (logging then prints its "Message: '<template>'" fallbacks, which
        # the attempts parser can never match).
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "src.main:app",
                "--log-config", str(logconf_path),
            ],
            cwd=str(BACKEND_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", shell=False,
        )
        tailer = LogTailer(proc)
        try:
            with httpx.Client(timeout=ASK_TIMEOUT_S) as client:
                startup = _wait_healthy(client)
                if startup is None:
                    print("FAIL server never became healthy")
                    md.append("server never became healthy")
                    continue
                print(f"startup_seconds={startup:.1f}")
                md += [f"startup: {startup:.1f}s (models loaded once)", ""]
                _lines, log_pos = tailer.since(0)  # startup noise discarded
                rows: list[dict] = []
                order = (
                    ("kau", "ksu", "kau", "ksu")
                    if name == "1-normal-chain" else ("kau", "ksu")
                )
                for i, uni in enumerate(order):
                    t0 = time.perf_counter()
                    resp = client.post(
                        f"{BASE}/ask",
                        json={"question": questions[uni], "university": uni},
                    )
                    elapsed = time.perf_counter() - t0
                    new_lines, log_pos = tailer.since(log_pos)
                    attempts = _attempts_of(new_lines)
                    auth_notes = _auth_notes_of(new_lines)
                    rec = _last_ask_record(
                        uni, questions[uni], scenario_start_utc
                    )
                    stage_seconds = (rec or {}).get("seconds")
                    generation = (stage_seconds or {}).get("generation")
                    try:
                        body = resp.json()
                        answer = (
                            body.get("answer", "")
                            if resp.status_code == 200
                            else json.dumps(body, ensure_ascii=False)
                        )
                    except Exception:
                        answer = resp.text[:400]
                    row = {
                        "uni": uni,
                        "cold": (
                            "first-cold" if i == 0
                            else "repeat-warm" if i >= 2 else "second"
                        ),
                        "status": resp.status_code,
                        "http_seconds": round(elapsed, 2),
                        "generation": generation,
                        "attempts": attempts,
                        "auth_notes": auth_notes,
                        "answer": answer,
                        "stage_seconds": stage_seconds,
                    }
                    rows.append(row)
                    held = _budget_held(row)
                    generation_txt = (
                        f"{generation}s" if generation is not None else "n/a"
                    )
                    print(
                        f"[{uni}/{row['cold']}] HTTP {resp.status_code} "
                        f"http={row['http_seconds']}s "
                        f"generation={generation_txt} "
                        f"budget_14s={'held' if held else 'EXCEEDED'}"
                    )
                    print(f"  attempts={attempts}")
                    if auth_notes:
                        print(f"  auth_notes={auth_notes}")
                    md += [
                        f"### {row['uni']} ({row['cold']})",
                        f"- HTTP {row['status']}, {row['http_seconds']}s, "
                        f"generation={generation_txt}, "
                        f"budget_14s={'held' if held else 'EXCEEDED'}",
                        f"- attempts: {attempts or 'n/a'}",
                        f"- stage_seconds: {stage_seconds}",
                        "", "```", answer, "```", "",
                    ]
                if name == "1-normal-chain" and len(rows) >= 3:
                    first, warm = rows[0], rows[2]
                    penalty = round(
                        first["http_seconds"] - warm["http_seconds"], 2
                    )
                    print("--- cold start comparison (scenario 1)")
                    print(
                        f"  first_kau={first['http_seconds']}s "
                        f"second_ksu={rows[1]['http_seconds']}s "
                        f"warm_kau={warm['http_seconds']}s "
                        f"cold_penalty={penalty}s"
                    )
                    md += [
                        "### cold-start comparison (scenario 1)",
                        f"- first /ask (kau): {first['http_seconds']}s",
                        f"- second /ask (ksu): {rows[1]['http_seconds']}s",
                        f"- warm repeat /ask (kau): {warm['http_seconds']}s",
                        f"- cold-start penalty over warm repeat: {penalty}s",
                        "",
                    ]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        print()

    out_path = BACKEND_DIR / "data" / OUT_NAME
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"arabic answers -> {out_path}")


if __name__ == "__main__":
    main()
