"""Tests for the 2026-10-25 improvements (no models, no network)."""

import pytest
from fastapi.testclient import TestClient

from src.answer_cache import AnswerCache, cache_key
from src.chroma_store import EmptyIngestError, ensure_chunks
from src.config import get_settings
from src.dialect import normalize_dialect
from src.grounding import numbers_in, ungrounded_numbers
from src.ksu_gate import KsuGateResult, filter_ranked, rerank_and_gate
from src.lexical import Bm25Index, normalize_token, tokenize
from src.main import create_app
from src.pipeline import CANNOT_CONFIRM, answer_question
from src.prompts import build_user_prompt
from src.retrieval import Retrieved

from tests.test_api_layer import (
    _FakeChainGenerator,
    _FakeReranker,
    _FakeRetriever,
    _configure,
)


def _r(src: str, doc: str, score: float = 0.9, **md) -> Retrieved:
    return Retrieved(source_type=src, score=score, document=doc, metadata=md)


# ------------------------------------------------------------ ingest guard


def test_ensure_chunks_refuses_zero() -> None:
    with pytest.raises(EmptyIngestError):
        ensure_chunks("kau_pdf", 0, "hint")
    ensure_chunks("kau_pdf", 3)  # no error



# ---------------------------------------------------------- prompt / rewrite


def test_standalone_question_line_only_when_different() -> None:
    same = build_user_prompt("س؟", "qa", "doc", standalone_question="س؟")
    assert same == build_user_prompt("س؟", "qa", "doc")
    diff = build_user_prompt("وهل فيه مواعيد؟", "qa", "doc",
                             standalone_question="ما مواعيد القبول؟")
    assert "ما مواعيد القبول؟" in diff and "وهل فيه مواعيد؟" in diff


# ------------------------------------------------------------ rerank context


def test_filter_ranked_drops_noise_keeps_top() -> None:
    ranked = [_r("ksu_pdf", "a"), _r("ksu_qa", "b"), _r("ksu_txt", "c")]
    gate = KsuGateResult(True, 0.9, 3, ranked, [0.9, 0.3, 0.05], 0.0)
    kept = filter_ranked(gate, min_score=0.10, relative=0.20)
    assert [r.document for r in kept] == ["a", "b"]
    gate_low = KsuGateResult(True, 0.4, 1, ranked[:1], [0.4], 0.0)
    assert filter_ranked(gate_low, 0.5, 0.2) == ranked[:1]


def test_extra_candidates_join_pool_deduplicated() -> None:
    dense = [_r("ksu_pdf", "a", 0.9)]
    extra = [_r("ksu_pdf", "a", 7.0), _r("ksu_txt", "kw", 5.0)]
    gate = rerank_and_gate("q", [], dense, _FakeReranker([0.2, 0.8]),
                           pool_size=6, threshold=0.4, extra_candidates=extra)
    assert gate.pool_size == 2
    assert gate.ranked[0].document == "kw" and gate.passed


def test_kau_rerank_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "kau_gate_mode", "rerank")
    retriever = _FakeRetriever(qa=[0.70], docs=[0.70])  # cosine gate would refuse
    gen = _FakeChainGenerator()
    res = answer_question("سؤال", retriever, gen, university="kau",
                          reranker=_FakeReranker([0.9, 0.6]))
    assert res.refused is False and gen.calls == 1
    assert res.gate == "rerank-pass"
    refused = answer_question("سؤال", retriever, gen, university="kau",
                              reranker=_FakeReranker([0.1, 0.05]))
    assert refused.refused is True and refused.answer == CANNOT_CONFIRM


def test_kau_cosine_mode_still_available(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_settings().kau_gate_mode == "rerank"  # default since 2026-10-25
    monkeypatch.setattr(get_settings(), "kau_gate_mode", "cosine")
    retriever = _FakeRetriever(qa=[0.70], docs=[0.70])
    res = answer_question("سؤال", retriever, _FakeChainGenerator(),
                          university="kau", reranker=_FakeReranker([0.99, 0.99]))
    assert res.refused is True  # calibrated cosine floors still decide


# ---------------------------------------------------------------- grounding


def test_numbers_canonical_forms() -> None:
    assert numbers_in("المعدل ٤٫٥٠ أو 4,5 والساعات 08") == {"4.5", "8"}


def test_ungrounded_numbers() -> None:
    ctx = "يشترط معدل 4.50 وإكمال ثلاث ساعات في 1448/1447"
    ans = "1. المعدل 4.5\n2. ثلاث ساعات = 3\nالعام 1448 [المصدر: سؤال رقم 77]"
    assert ungrounded_numbers(ans, ctx) == []
    assert ungrounded_numbers("المعدل 3.75 والمدة 14 يوما", ctx) == ["14", "3.75"]


def test_numeric_guard_refuse_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "numeric_guard_mode", "refuse")

    class _Inventing(_FakeChainGenerator):
        def _invoke_once(self, system, prompt, timeout_seconds):
            return "المعدل المطلوب 3.99 من 5.", False

    retriever = _FakeRetriever(qa=[0.90])
    res = answer_question("سؤال", retriever, _Inventing(), university="kau")
    assert res.refused is True and res.answer == CANNOT_CONFIRM
    assert "3.99" in res.ungrounded_numbers


# -------------------------------------------------------- dialect / lexical


def test_dialect_whole_words_only() -> None:
    assert normalize_dialect("وش شروط التحويل") == "ما شروط التحويل"
    assert normalize_dialect("ابي احذف ماده") == "أريد احذف ماده"
    assert normalize_dialect("الوشاح") == "الوشاح"


def test_lexical_normalization_and_bm25() -> None:
    assert normalize_token("بالجامعة") == "جامعه"
    assert "وش" not in tokenize("وش شروط الحرمان")
    index = Bm25Index(
        ["الحرمان من دخول الاختبار عند الغياب 25%", "مكافأة الطالب الشهرية",
         "شروط التحويل الداخلي"],
        [{}, {}, {}],
    )
    hits = index.search("نسبة الغياب للحرمان", k=2)
    assert hits and hits[0][0] == 0


# --------------------------------------------------------------------- API


def test_answer_cache_ttl_and_lru() -> None:
    cache = AnswerCache(size=1, ttl_s=60)
    cache.put(cache_key("kau", " س  ؟"), {"answer": "a"})
    assert cache.get(cache_key("kau", "س ؟")) == {"answer": "a"}
    cache.put(cache_key("kau", "ب"), {"answer": "b"})
    assert cache.get(cache_key("kau", "س ؟")) is None  # evicted
    assert AnswerCache(0, 60).get(cache_key("kau", "x")) is None


def test_api_cache_sources_and_empty_kb(tmp_path) -> None:
    from src.api_logging import JsonlLogger

    app = create_app(load_resources=False)
    with TestClient(app) as client:
        app.state.ask_logger = JsonlLogger(tmp_path, "ask")
        gen = _FakeChainGenerator()
        _configure(app.state, retriever=_FakeRetriever(qa=[0.90]), chain=[gen])
        body = {"question": "سؤال مكرر", "university": "kau"}
        first = client.post("/ask", json=body).json()
        second = client.post("/ask", json=body).json()
        assert first == second and gen.calls == 1  # served from cache
        assert len(first["sources"]) == 1
        assert first["sources"][0].startswith("سؤال موثّق رقم")

        app.state.collection_counts = {"kau": {"kau_qa_verified": 0}, "ksu": {}}
        res = client.post("/ask", json={"question": "جديد", "university": "kau"})
        assert res.status_code == 503 and "empty" in res.json()["error"]
        details = client.get("/health/details").json()
        assert details["status"] == "degraded"
        assert client.get("/health").json() == {"status": "ok"}


# --------------------------------------------------------------- eval suite


def test_eval_suite_gate_only_runs_with_fakes(capsys) -> None:
    from src.eval_suite import QUESTIONS, load_rows, run, summarize

    rows = load_rows(QUESTIONS, "ksu")
    assert len(rows) >= 20
    records = run(rows, _FakeRetriever(qa=[0.9], docs=[0.9], qa_type="ksu_qa",
                                       doc_type="ksu_pdf"),
                  _FakeReranker([0.9, 0.2]), generate=False)
    assert any(r["status"].startswith("skipped") for r in records)  # follow-up row
    metrics = summarize(records, "fake")["ksu"]
    assert metrics["refuse_correct"].endswith("/7")


def test_eval_keyword_match_ignores_spelling_variants() -> None:
    from src.eval_suite import _contains_any

    assert _contains_any("كيف اطلع بطاقه جامعيه بدل فاقد", ["البطاقة الجامعية"])
    assert not _contains_any("مكافأة شهرية", ["الحرمان"])


# ------------------------------------------------- rewrite drift safety net


def test_refused_rewrite_retries_with_raw_question() -> None:
    """New topic after a CS-admission turn: the rewrite drags the old topic
    in and is refused; the raw question must get a second chance."""

    class _QueryAwareRetriever(_FakeRetriever):
        def __init__(self) -> None:
            super().__init__(qa=[0.9], qa_type="ksu_qa", doc_type="ksu_pdf")
            self.queries: list[str] = []

        def retrieve_split(self, query, k=None, university="kau"):
            self.queries.append(query)
            return super().retrieve_split(query, k, university)

    class _QueryAwareReranker:
        def predict(self, pairs, batch_size=8, show_progress_bar=False):
            return [0.05 if "علوم الحاسب" in q else 0.9 for q, _d in pairs]

    class _Scripted(_FakeChainGenerator):
        def _invoke_once(self, system, prompt, timeout_seconds):
            if "سؤال المتابعة" in prompt:
                return "كم المعدل الذي يسبب الإنذار في كلية علوم الحاسب؟", False
            return "جواب من اللائحة.", False

    retriever = _QueryAwareRetriever()
    res = answer_question(
        "كم المعدل اللي يجي عليه انذار", retriever, _Scripted(),
        university="ksu", reranker=_QueryAwareReranker(),
        history=[("ما شروط القبول في كلية علوم الحاسب؟", "الشروط ...")],
    )
    assert len(retriever.queries) == 2
    assert "علوم الحاسب" in retriever.queries[0]
    assert "علوم الحاسب" not in retriever.queries[1]
    assert res.refused is False and "raw-question retry" in res.gate


def test_rewrite_prompt_forbids_topic_injection() -> None:
    from src.query_rewrite import build_rewrite_prompts

    system, _user = build_rewrite_prompts([("س", "ج")], "سؤال")
    assert "موضوعاً جديداً" in system and "لا تضف" in system


# ------------------------------------------------ catalogue-page chunking


def test_catalogue_page_one_chunk_per_college() -> None:
    from src.ingest.kau_pdf import doc_page_chunks

    page = (
        "كلية الهندسة\n\n- • الهندسة المدنية\n- • الهندسة الكيميائية\n\n"
        "كلية الحاسبات وتقنية المعلومات\n\n- • علوم الحاسبات\n- • نظم المعلومات\n\n"
        "كلية العمارة والتخطيط\n\n- • العمارة\n- • الجيوماتكس"
    )
    chunks = doc_page_chunks(page, "دليل القبول", 5, target=700, max_chars=1200)
    assert len(chunks) == 3
    cs = [c for c in chunks if "علوم الحاسبات" in c][0]
    assert "كلية الحاسبات وتقنية المعلومات" in cs and "الهندسة" not in cs
    assert cs.startswith("دليل القبول — صفحة 5\n")


def test_non_catalogue_page_keeps_historical_chunking() -> None:
    from src.ingest.kau_pdf import catalogue_sections

    blocks = ["مقدمة الدليل", "نص عادي طويل عن شروط التحويل.", "| جدول | 4.5 |"]
    assert catalogue_sections(blocks) is None


# ------------------------------------------------- general fallback (opt-in)


_SEARCH_RESPONSE = {
    "candidates": [{
        "content": {"parts": [{"text": "تُقدَّم طلبات المنح عبر بوابة القبول."}]},
        "groundingMetadata": {
            "webSearchQueries": ["منح جامعة الملك عبدالعزيز site:kau.edu.sa"],
            "groundingChunks": [
                {"web": {"uri": "https://vertexaisearch.cloud.google.com/x1",
                         "title": "admission.kau.edu.sa"}},
                {"web": {"uri": "https://vertexaisearch.cloud.google.com/x2",
                         "title": "studyshoot.com"}},
            ],
        },
    }]
}


def test_parse_search_keeps_only_official_sources() -> None:
    from src.general_fallback import parse_search_response

    text, official = parse_search_response(_SEARCH_RESPONSE, ("kau.edu.sa",))
    assert "بوابة القبول" in text
    assert [w["title"] for w in official] == ["admission.kau.edu.sa"]
    unofficial = {"candidates": [{**_SEARCH_RESPONSE["candidates"][0],
                                  "groundingMetadata": {"groundingChunks": [
                                      {"web": {"uri": "u", "title": "studyshoot.com"}}]}}]}
    assert parse_search_response(unofficial, ("kau.edu.sa",)) is None
    no_answer = {"candidates": [{"content": {"parts": [{"text": "NO_ANSWER"}]}}]}
    assert parse_search_response(no_answer, ("kau.edu.sa",)) is None


def test_fallback_off_by_default_keeps_refusal() -> None:
    assert get_settings().general_fallback_mode == "off"
    res = answer_question("من هو رئيس الجامعة", _FakeRetriever(qa=[0.1]),
                          _FakeChainGenerator(), university="kau",
                          reranker=_FakeReranker([0.01]))
    assert res.refused is True and res.answer == CANNOT_CONFIRM


def test_web_fallback_labels_answer_with_links(monkeypatch) -> None:
    import httpx as _httpx

    settings = get_settings()
    monkeypatch.setattr(settings, "general_fallback_mode", "web_then_llm")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "web_search_provider", "gemini")

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return _SEARCH_RESPONSE

    calls = []
    monkeypatch.setattr(_httpx, "post", lambda *a, **k: calls.append(k) or _Resp())
    res = answer_question("كيف أقدم على المنح؟", _FakeRetriever(qa=[0.1]),
                          _FakeChainGenerator(), university="kau",
                          reranker=_FakeReranker([0.01]))
    assert res.refused is False and "+fallback:web" in res.gate
    assert res.answer.startswith("🔎") and "admission.kau.edu.sa" in res.answer
    assert "studyshoot" not in res.answer
    assert calls[0]["json"]["tools"] == [{"google_search": {}}]


def test_llm_fallback_labeled_and_declines_regulations(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "general_fallback_mode", "llm")

    class _General(_FakeChainGenerator):
        def _invoke_once(self, system, prompt, timeout_seconds):
            if "الحرمان" in prompt:
                return "NO_ANSWER", False
            return "تقع الجامعة في جدة.", False

    ok = answer_question("وين تقع الجامعة", _FakeRetriever(qa=[0.1]), _General(),
                         university="kau", reranker=_FakeReranker([0.01]))
    assert ok.refused is False and ok.answer.startswith("⚠️")
    reg = answer_question("كم نسبة الحرمان", _FakeRetriever(qa=[0.1]), _General(),
                          university="kau", reranker=_FakeReranker([0.01]))
    assert reg.refused is True and reg.answer == CANNOT_CONFIRM


def test_ksu_web_pages_join_txt_collection() -> None:
    from src.ingest.ksu_txt import build_ksu_web_chunks

    s = get_settings()
    chunks = build_ksu_web_chunks(s.sources_dir / "KSU" / "web", s.chunk_max_chars,
                                  lambda t: len(t.split()) * 2, 450)
    assert chunks and all(c.metadata["type"] == "web" for c in chunks)
    assert any("رئيس الجامعة" in c.display for c in chunks)


def test_dependent_follow_up_kept_verbatim_is_anchored() -> None:
    from src.query_rewrite import looks_dependent

    assert looks_dependent("لخّص الإجابة بنقاط")
    assert looks_dependent("هل هناك مواعيد محددة؟")
    assert looks_dependent("أعطني مثالاً")
    assert not looks_dependent("كم المعدل اللي يجي عليه انذار")
    assert not looks_dependent("أين أجد الجدول الدراسي؟")

    class _Echo(_FakeChainGenerator):
        def _invoke_once(self, system, prompt, timeout_seconds):
            if "سؤال المتابعة" in prompt:
                return "لخّص الإجابة بنقاط", False  # model kept it verbatim
            return "جواب.", False

    class _Rec(_FakeRetriever):
        def __init__(self):
            super().__init__(qa=[0.9])
            self.queries = []

        def retrieve_split(self, query, k=None, university="kau"):
            self.queries.append(query)
            return super().retrieve_split(query, k, university)

    r = _Rec()
    answer_question("لخّص الإجابة بنقاط", r, _Echo(), university="kau",
                    reranker=_FakeReranker([0.9]),
                    history=[("ما التخصصات المتاحة في كلية الحاسبات؟", "علوم الحاسبات...")])
    assert r.queries[0].startswith("ما التخصصات المتاحة في كلية الحاسبات؟")


def test_tavily_web_fallback_answers_from_official_pages(monkeypatch) -> None:
    import httpx as _httpx

    settings = get_settings()
    monkeypatch.setattr(settings, "general_fallback_mode", "web")
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"results": [
                {"title": "التقويم الأكاديمي", "url": "https://ksu.edu.sa/ar/calendar",
                 "raw_content": "تبدأ الاختبارات النهائية للفصل الأول يوم 11-07-1448هـ"},
                {"title": "منتدى", "url": "https://forum.example.com/x",
                 "raw_content": "الاختبارات 1-1-1449"},
            ]}

    sent = []
    monkeypatch.setattr(_httpx, "post", lambda url, **k: sent.append((url, k)) or _Resp())

    class _Reader(_FakeChainGenerator):
        def _invoke_once(self, system, prompt, timeout_seconds):
            assert "forum.example.com" not in prompt  # non-official dropped
            return "تبدأ الاختبارات النهائية يوم 11-07-1448هـ.", False

    res = answer_question("متى الاختبارات النهائية؟",
                          _FakeRetriever(qa=[0.1], qa_type="ksu_qa", doc_type="ksu_pdf"),
                          _Reader(), university="ksu", reranker=_FakeReranker([0.01]))
    assert res.refused is False and res.answer.startswith("🔎")
    assert "https://ksu.edu.sa/ar/calendar" in res.answer
    assert sent[0][1]["json"]["include_domains"] == ["ksu.edu.sa"]


def test_tavily_answer_with_invented_number_is_rejected(monkeypatch) -> None:
    from src.general_fallback import answer_from_pages

    pages = [{"title": "t", "url": "https://kau.edu.sa/p", "text": "الاختبارات يوم 11-07-1448"}]

    class _Inventor(_FakeChainGenerator):
        def _invoke_once(self, system, prompt, timeout_seconds):
            return "الاختبارات يوم 20-08-1448.", False

    assert answer_from_pages("متى؟", "kau", pages, [_Inventor()], None, get_settings()) is None


def test_web_answer_lists_only_used_pages() -> None:
    from src.general_fallback import answer_from_pages, split_used_snippets

    assert split_used_snippets("الجواب.\nالمقتطفات: 1، ٣", 3) == ("الجواب.", [1, 3])
    assert split_used_snippets("الجواب.", 2) == ("الجواب.", [])
    pages = [
        {"title": "تقويم 2026-2027", "url": "https://dar.ksu.edu.sa/ar/current",
         "text": "الاختبارات النهائية 13/12/2026"},
        {"title": "تقويم 1446", "url": "https://cfy.ksu.edu.sa/old.pdf",
         "text": "الاختبارات النهائية 01/01/2025"},
    ]

    class _Cites(_FakeChainGenerator):
        def _invoke_once(self, system, prompt, timeout_seconds):
            return "تبدأ يوم 13/12/2026 (العام 2026-2027).\nالمقتطفات: 1", False

    ans = answer_from_pages("متى؟", "ksu", pages, [_Cites()], None, get_settings())
    assert ans is not None and "dar.ksu.edu.sa" in ans.text
    assert "old.pdf" not in ans.text and "المقتطفات" not in ans.text


def test_warm_up_runs_search_and_rerank_per_university() -> None:
    from types import SimpleNamespace

    from src.main import _warm_up

    class _CountingReranker(_FakeReranker):
        def __init__(self):
            super().__init__([0.5] * 6)
            self.calls = 0

        def predict(self, pairs, batch_size=8, show_progress_bar=False):
            self.calls += 1
            return super().predict(pairs)

    reranker = _CountingReranker()
    app = SimpleNamespace(state=SimpleNamespace(
        retriever=_FakeRetriever(qa=[0.9], docs=[0.8]), reranker=reranker))
    _warm_up(app, get_settings())
    assert reranker.calls == 2  # kau + ksu

    broken = SimpleNamespace(state=SimpleNamespace(retriever=None, reranker=None))
    _warm_up(broken, get_settings())  # never raises


def test_model_refusal_also_triggers_web_fallback(monkeypatch) -> None:
    import httpx as _httpx

    settings = get_settings()
    monkeypatch.setattr(settings, "general_fallback_mode", "web")
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"results": [{"title": "كليات المزاحمية",
                                 "url": "https://ksu.edu.sa/en/colleges-muzahimya-branch",
                                 "raw_content": "كلية العلوم والدراسات الإنسانية بالمزاحمية"}]}

    monkeypatch.setattr(_httpx, "post", lambda url, **k: _Resp())

    class _Gen(_FakeChainGenerator):
        def _invoke_once(self, system, prompt, timeout_seconds):
            if "مقتطفات من الموقع الرسمي" in prompt:
                return "من كليات المزاحمية: كلية العلوم والدراسات الإنسانية.\nالمقتطفات: 1", False
            return CANNOT_CONFIRM, False  # context passed the gate, no answer in it

    res = answer_question("وشو الكليات الي في المزاحمية",
                          _FakeRetriever(qa=[0.9], qa_type="ksu_qa", doc_type="ksu_pdf"),
                          _Gen(), university="ksu", reranker=_FakeReranker([0.9]))
    assert res.answer.startswith("🔎") and "colleges-muzahimya-branch" in res.answer
    assert "+fallback:web" in res.gate


def test_dialect_student_vocabulary_and_typos() -> None:
    assert normalize_dialect("ابي اأجل الترم كيف") == "أريد أؤجل الفصل الدراسي كيف"
    assert normalize_dialect("كم مكافئة التفوق") == "كم مكافأة التفوق"
    assert normalize_dialect("بعد كم انزار ينفصل") == "بعد كم إنذار ينفصل"


def test_tavily_query_uses_normalized_search_and_university_name(monkeypatch) -> None:
    import httpx as _httpx

    settings = get_settings()
    monkeypatch.setattr(settings, "general_fallback_mode", "web")
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")
    sent = []

    class _Empty:
        status_code = 200
        text = ""

        def json(self):
            return {"results": []}

    monkeypatch.setattr(_httpx, "post", lambda url, **k: sent.append(k["json"]) or _Empty())
    answer_question("وشو الكليات الي في المزاحمية",
                    _FakeRetriever(qa=[0.1], qa_type="ksu_qa", doc_type="ksu_pdf"),
                    _FakeChainGenerator(), university="ksu", reranker=_FakeReranker([0.01]))
    assert sent and sent[0]["query"] == "ما الكليات الي في المزاحمية جامعة الملك سعود"
    assert sent[0]["max_results"] == 5


def test_select_passages_prefers_relevant_parts_of_long_pages() -> None:
    from src.general_fallback import select_passages

    filler = "\n\n".join(f"فقرة عامة رقم {i} عن أخبار الجامعة والفعاليات." for i in range(40))
    page = {"title": "فرع المزاحمية", "url": "https://mz.ksu.edu.sa/ar", "snippet": "",
            "text": filler + "\n\nكليات فرع المزاحمية: كلية الهندسة التطبيقية، "
                    "كلية إدارة الأعمال، كلية العلوم والدراسات الإنسانية."}
    other = {"title": "برنامج التسويق", "url": "https://mz.ksu.edu.sa/ar/marketing",
             "snippet": "برنامج التسويق التطبيقي في كلية إدارة الأعمال بالمزاحمية.", "text": ""}
    chosen = select_passages("ما الكليات في المزاحمية", [page, other], 700, 3)
    texts = " ".join(c["text"] for c in chosen)
    assert "كلية العلوم والدراسات الإنسانية" in texts  # deep in the page, still found
    assert chosen[0]["text"].startswith("برنامج التسويق")  # Tavily snippet kept first


# ------------------------------------------------------------ status page


def test_parse_duration_variants() -> None:
    from src.usage_tracker import parse_duration

    assert abs(parse_duration("2m59.56s") - 179.56) < 1e-6
    assert parse_duration("7.66s") == 7.66
    assert parse_duration("250ms") == 0.25
    assert parse_duration("1h2m") == 3720
    assert parse_duration("12") == 12.0
    assert parse_duration("") is None


def test_tracker_groq_headers_and_counters(tmp_path) -> None:
    from src.usage_tracker import UsageTracker, groq_view

    t = UsageTracker(tmp_path / "usage.json")
    headers = {"x-ratelimit-remaining-requests": "990", "x-ratelimit-limit-requests": "1000",
               "x-ratelimit-remaining-tokens": "7500", "x-ratelimit-limit-tokens": "8000",
               "x-ratelimit-reset-requests": "2m0s", "x-ratelimit-reset-tokens": "7.5s",
               "authorization": "Bearer SECRET"}
    t.record_response("groq", "gpt-oss", headers, prompt_tokens=1200, completion_tokens=300)
    t.record_error("gemini", 429, "quota", {"retry-after": "30"})
    snap = t.snapshot()
    g = snap["providers"]["groq"]
    assert "authorization" not in g["rate_headers"]  # only rate-limit headers kept
    view = groq_view(g["rate_headers"], g["rate_headers_at"])
    assert view["requests_remaining_day"] == 990 and view["tokens_limit_minute"] == 8000
    assert abs(view["requests_reset_at"] - g["rate_headers_at"] - 120) < 1e-6
    assert snap["today"]["groq"]["prompt_tokens"] == 1200
    assert snap["providers"]["gemini"]["last_error"]["retry_after_s"] == 30
    assert "SECRET" not in (tmp_path / "usage.json").read_text(encoding="utf-8")
    again = UsageTracker(tmp_path / "usage.json").snapshot()  # survives restart
    assert again["today"]["groq"]["requests"] == 1


def test_usage_endpoint_and_status_page() -> None:
    app = create_app(load_resources=False)
    with TestClient(app) as client:
        page = client.get("/status")
        assert page.status_code == 200 and "رصيد المزودين" in page.text
        data = client.get("/usage").json()
        ids = [p["id"] for p in data["providers"]]
        assert ids[:3] == ["groq", "mistral", "gemini"] and "tavily" in ids
        gem = [p for p in data["providers"] if p["id"] == "gemini"][0]
        assert "reset_at" not in gem and gem["note"]  # no computed numbers
        assert data["live"] is False


def test_generator_feeds_tracker(monkeypatch) -> None:
    import httpx as _httpx
    from src import generators as gens
    from src.usage_tracker import get_tracker

    class _R:
        status_code = 200
        text = ""
        headers = {"x-ratelimit-remaining-requests": "500"}

        def json(self):
            return {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2}}

    monkeypatch.setattr(_httpx, "post", lambda *a, **k: _R())
    before = get_tracker().snapshot()["today"].get("groq", {}).get("prompt_tokens", 0)
    gens.GroqGenerator(get_settings())._invoke_once("s", "p", 5.0)
    after = get_tracker().snapshot()
    assert after["today"]["groq"]["prompt_tokens"] == before + 7
    assert after["providers"]["groq"]["rate_headers"]["x-ratelimit-remaining-requests"] == "500"



def test_live_check_uses_provider_headers_without_counting(monkeypatch) -> None:
    import httpx as _httpx

    from src.status_page import build_usage_report
    from src.usage_tracker import UsageTracker

    class _Models:
        status_code = 200
        text = ""
        headers = {"x-ratelimit-remaining-requests": "997", "x-ratelimit-limit-requests": "1000"}

        def json(self):
            return {"data": []}

    class _Tav:
        status_code = 200

        def json(self):
            return {"key": {"usage": 10, "limit": 1000}}

    monkeypatch.setattr(_httpx, "get", lambda url, **k: _Tav() if "tavily" in url else _Models())
    monkeypatch.setattr(_httpx, "post", lambda url, **k: _Models())
    s = get_settings()
    monkeypatch.setattr(s, "groq_api_key", "g")
    monkeypatch.setattr(s, "tavily_api_key", "t")
    tracker = UsageTracker(None)
    report = build_usage_report(s, tracker, live=True)
    groq = [p for p in report["providers"] if p["id"] == "groq"][0]
    assert groq["check_source"] == "models"
    assert groq["groq"]["requests_remaining_day"] == 997
    assert groq["device_today"] == {}  # live checks never inflate counters
    tav = [p for p in report["providers"] if p["id"] == "tavily"][0]
    assert tav["account"]["key"]["usage"] == 10
