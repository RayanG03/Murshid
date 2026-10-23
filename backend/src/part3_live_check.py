"""Part 3 live check: Groq and Mistral model existence + one Arabic request.

For each provider: GET /models and confirm the configured model id is
listed (abort loudly if not, no substitutions), then one short Arabic
chat completion. Console shows status / seconds / finish_reason only;
Arabic text goes to data/part3_live.md (UTF-8).

run from backend/:  uv run python -X utf8 -m src.part3_live_check
"""

import time

import httpx

from src.config import get_settings

OUT_NAME = "part3_live.md"

PROMPT = "اكتب جملة واحدة قصيرة بالعربية للترحيب بالطلاب الجدد."


def _check_provider(
    settings, name: str, base: str, model_id: str, api_key: str
) -> tuple[list[str], list[str]]:
    """GET /models + one Arabic completion.

    Returns (console_lines, markdown_lines). The console lines are ASCII
    only; the markdown carries the Arabic completion text.
    """
    console: list[str] = []
    md: list[str] = [f"## {name}", ""]
    headers = {"Authorization": f"Bearer {api_key}"}

    t0 = time.perf_counter()
    resp = httpx.get(
        f"{base.rstrip('/')}/models", headers=headers,
        timeout=settings.per_attempt_timeout_s,
    )
    listing_seconds = time.perf_counter() - t0
    listed = False
    if resp.status_code == 200:
        listed = any(
            m.get("id", "") == model_id for m in resp.json().get("data", [])
        )
    console.append(
        f"{name}: GET /models HTTP {resp.status_code} "
        f"({listing_seconds:.2f}s) model listed={listed}"
    )
    md.append(f"models endpoint: HTTP {resp.status_code}; "
              f"model listed: {listed}")
    md.append("")
    if not listed:
        console.append(
            f"STOP: {model_id} is NOT listed for {name} — aborting, "
            "no substitution attempted."
        )
        return console, md

    t0 = time.perf_counter()
    seconds = 0.0
    finish = ""
    text = ""
    truncated = ""
    status = 0
    for attempt in range(3):
        resp = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 0.2,
                "max_completion_tokens": 512,
            },
            timeout=settings.per_attempt_timeout_s,
        )
        status = resp.status_code
        seconds = time.perf_counter() - t0
        if status == 429 and attempt < 2:
            time.sleep(5.0 * (attempt + 1))
            continue
        if status == 200:
            choice = resp.json()["choices"][0]
            finish = str(choice.get("finish_reason", ""))
            text = choice.get("message", {}).get("content") or ""
            truncated = " (TRUNCATED)" if finish == "length" else ""
        break
    console.append(
        f"{name}: completion HTTP {status} ({seconds:.2f}s) "
        f"finish_reason={finish}{truncated}"
    )
    md += [
        f"completion: HTTP {status}, {seconds:.2f}s, "
        f"finish_reason={finish}{truncated}",
        "",
        "```",
        text,
        "```",
        "",
    ]
    return console, md


def main() -> None:
    """Live-check both OpenAI-compatible providers; md report to data/."""
    s = get_settings()
    targets = (
        ("groq", s.groq_base_url, s.groq_model_id, s.groq_api_key),
        ("mistral", s.mistral_base_url, s.mistral_model_id, s.mistral_api_key),
    )
    console_all: list[str] = []
    md_all: list[str] = ["# Part 3 live provider check", ""]
    for name, base, model_id, api_key in targets:
        console_lines, md_lines = _check_provider(
            s, name, base, model_id, api_key
        )
        console_all += console_lines
        md_all += md_lines
    out_path = s.data_dir / OUT_NAME
    out_path.write_text("\n".join(md_all), encoding="utf-8")
    for line in console_all:
        print(line)
    print(f"arabic text -> {out_path}")


if __name__ == "__main__":
    main()
