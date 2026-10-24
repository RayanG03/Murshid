"""KSU failover drill: scenarios 2 (500 failover) and 3 (timeout failover).

Runs the REAL pipeline (retrieval gate -> rerank gate -> failover chain)
for KSU with fakes for the slow parts (retriever/reranker sleep to mimic
real CPU stage cost; httpx.post faked per scenario) — no models, no
network. The request deadline is anchored at drill request start exactly
like src/main.py does, so each printed total is the true /ask-equivalent
wall-clock time. PASS requires every total < 20s (frontend abort limit).

Scenarios:
  2. groq returns HTTP 500 -> fail over -> mistral answers.
  3. groq raises a read timeout -> fail over -> mistral answers.

run from backend/:
  uv run python -X utf8 -m src.failover_drill
"""

import time
from collections.abc import Callable
from typing import Any

import src.generators as gens
from src.config import Settings, get_settings
from src.pipeline import answer_question
from src.retrieval import Retrieved

RERANK_SLEEP_S = 2.0  # mimics the real bge-reranker stage on CPU
ASK_LIMIT_S = 20.0  # frontend abort limit

QA_HITS: list[Retrieved] = [
    Retrieved(
        source_type="ksu_qa",
        score=0.91,
        document="س: وش شروط الاعتذار عن اختبار المقرر النهائي بعذر مقبول؟",
        metadata={"type": "qa", "q_id": 1, "answer": "إجابة مصدرية."},
    ),
    Retrieved(
        source_type="ksu_qa",
        score=0.88,
        document="س: متى يبدأ التسجيل للفصل الدراسي الأول؟",
        metadata={"type": "qa", "q_id": 2, "answer": "إجابة مصدرية."},
    ),
]


class _FakeRetriever:
    """Dense hits above the gate; sleeps to consume deadline budget."""

    def retrieve_split(
        self, query: str, k: int | None = None, university: str = "kau",
    ) -> tuple[list[Retrieved], list[Retrieved]]:
        del query, k, university
        time.sleep(0.2)  # retrieval stage cost
        return QA_HITS, []


class _FakeReranker:
    """Scores the pool above the KSU threshold after a realistic sleep."""

    def predict(self, pairs: Any, batch_size: int = 8,
                show_progress_bar: bool = False) -> list[float]:
        del pairs, batch_size, show_progress_bar
        time.sleep(RERANK_SLEEP_S)
        return [0.90, 0.85]


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload) if status_code != 200 else ""
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self._payload


def _ok(content: str) -> dict[str, Any]:
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": content}}
        ]
    }


def _install_post(responder: Callable[[str], tuple[int, dict[str, Any]]]) -> None:
    """Fake gens.httpx.post: responder(url) -> (status, payload)."""

    def _fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        del kwargs
        status, payload = responder(url)
        return _FakeResponse(payload, status)

    gens.httpx.post = _fake_post


def _drill_settings() -> Settings:
    """Production-like settings; keys are dummies, URLs are unreachable."""
    return Settings(
        groq_api_key="DRILL_GROQ_KEY",
        mistral_api_key="DRILL_MISTRAL_KEY",
        gemini_api_key="DRILL_GEMINI_KEY",
        groq_base_url="https://api.groq.invalid/openai/v1",
        mistral_base_url="https://api.mistral.invalid/v1",
        gemini_base_url="https://generativelanguage.invalid/v1beta",
        llm_retry_backoff_base_seconds=0.01,
    )


def _run_scenario(name: str) -> float:
    """Run one KSU drill; return total request seconds (assert < 20s)."""
    settings = _drill_settings()
    chain = gens.get_generation_chain(["groq", "mistral", "gemini"], settings)
    deadline = time.monotonic() + get_settings().request_deadline_s
    t0 = time.perf_counter()
    result = answer_question(
        "وش شروط الاعتذار عن اختبار المقرر النهائي بعذر مقبول؟",
        _FakeRetriever(),  # type: ignore[arg-type]
        chain,
        university="ksu",
        reranker=_FakeReranker(),  # type: ignore[arg-type]
        deadline=deadline,
    )
    total = time.perf_counter() - t0
    verdict = "PASS" if total < ASK_LIMIT_S else "FAIL"
    print(
        f"scenario {name}: refused={result.refused} "
        f"provider={result.provider} model={result.model} "
        f"total={total:.2f}s limit={ASK_LIMIT_S:.0f}s -> {verdict}"
    )
    return total


def main() -> None:
    """Run scenarios 2 and 3 for KSU; every total must be under 20s."""
    real_post = gens.httpx.post
    try:
        # Scenario 2: groq 500 -> in-provider retry then fail over ->
        # mistral answers; gemini never reached.
        _install_post(lambda url: (
            (500, {"error": "boom"}) if "groq" in url
            else (200, _ok("إجابة ميسترال بعد فشل جروق."))
        ))
        total_2 = _run_scenario("2 (groq 500 -> mistral)")

        # Scenario 3: groq read timeout -> mistral answers.
        def _timeout_responder(url: str) -> tuple[int, dict[str, Any]]:
            if "groq" in url:
                raise gens.httpx.ReadTimeout("groq read timed out")
            return (200, _ok("إجابة ميسترال بعد مهلة جروق."))

        _install_post(_timeout_responder)
        total_3 = _run_scenario("3 (groq timeout -> mistral)")
    finally:
        gens.httpx.post = real_post

    worst = max(total_2, total_3)
    print(
        f"worst total={worst:.2f}s limit={ASK_LIMIT_S:.0f}s -> "
        f"{'PASS' if worst < ASK_LIMIT_S else 'FAIL'}"
    )
    if worst >= ASK_LIMIT_S:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
