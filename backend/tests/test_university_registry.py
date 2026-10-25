"""University registry tests: routing, prompts, citation shapes, gates.

Pure unit tests: fake collection/embedding objects, no ChromaDB, no models.
"""

import pytest

from src.chroma_store import (
    COL_KAU_PDF,
    COL_KAU_QA,
    COL_KAU_WEB,
    COL_KSU_PDF,
    COL_KSU_QA,
    COL_KSU_TXT,
    get_existing_collection,
)
from src.config import (
    UNIVERSITY_COLLECTIONS,
    get_university_gates,
    is_university_available,
)
from src.context_assembly import (
    AssembledContext,
    Retrieved,
    _doc_line,
    render_context,
)
from src.prompts import (
    DOC_BLOCK_TITLE,
    QA_BLOCK_TITLE,
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_prompt,
    qa_block_title,
)
from src.retrieval import Retriever

KAU_PDF_MD = {"type": "pdf", "doc": "دليل التحويل الداخلي", "page": 7}
KAU_WEB_MD = {
    "type": "web",
    "page_title": "الخدمات المالية",
    "section": "الرسوم",
    "source_url": "https://kau.edu.sa/x",
}


class _FakeCollection:
    name: str
    count_value: int

    def __init__(self, name: str, count_value: int) -> None:
        self.name = name
        self.count_value = count_value

    def count(self) -> int:
        return self.count_value

    def query(self, **_: int) -> dict:
        """Minimal query response with a single result."""
        return {
            "documents": [["نص"]],
            "metadatas": [[{"q_id": 3, "answer": "جواب"}]],
            "distances": [[0.05]],
        }


class _FakeClient:
    def __init__(self, names: list[str]) -> None:
        self.names = set(names)
        self.get_collection_calls = 0

    def get_collection(self, name: str) -> _FakeCollection:
        self.get_collection_calls += 1
        if name not in self.names:
            raise KeyError(name)
        return _FakeCollection(name, 2)

    def create_collection(self, name: str) -> _FakeCollection:
        raise AssertionError(f"create side effect requested for {name!r}")


class _FakeEmbedder:
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in queries]


def _hit(md: dict) -> Retrieved:
    return Retrieved(source_type="kau_pdf", score=0.9, document="نص", metadata=md)


def test_registry_collections_match_chroma_constants() -> None:
    assert UNIVERSITY_COLLECTIONS["kau"] == (COL_KAU_QA, COL_KAU_WEB, COL_KAU_PDF)
    assert UNIVERSITY_COLLECTIONS["ksu"] == (COL_KSU_QA, COL_KSU_TXT, COL_KSU_PDF)


def test_get_existing_collection_never_creates() -> None:
    client = _FakeClient(["kau_qa_verified"])
    assert get_existing_collection(client, "kau_qa_verified") is not None
    missing = get_existing_collection(client, "ksu_txt")
    assert missing is None
    assert len(client.names) == 1  # no collection was created


def test_retrieve_split_routes_by_university() -> None:
    client = _FakeClient(
        ["kau_qa_verified", "kau_web", "kau_pdf", "ksu_qa", "ksu_txt", "ksu_pdf"]
    )
    retriever = Retriever(client, _FakeEmbedder())
    qa, docs = retriever.retrieve_split("سؤال", k=3, university="kau")
    assert qa and qa[0].source_type == COL_KAU_QA
    assert all(d.source_type in (COL_KAU_WEB, COL_KAU_PDF) for d in docs)
    qa, docs = retriever.retrieve_split("سؤال", k=3, university="ksu")
    assert qa and qa[0].source_type == COL_KSU_QA
    assert all(d.source_type in (COL_KSU_TXT, COL_KSU_PDF) for d in docs)


def test_retrieve_split_missing_collections_unavailable() -> None:
    retriever = Retriever(_FakeClient([]), _FakeEmbedder())
    qa, docs = retriever.retrieve_split("سؤال", university="kau")
    assert qa == [] and docs == []


def test_retrieve_split_unknown_university_raises() -> None:
    retriever = Retriever(_FakeClient([]), _FakeEmbedder())
    with pytest.raises(ValueError):
        retriever.retrieve_split("سؤال", university="uqu")


def test_kau_system_prompt_is_byte_identical_to_constant() -> None:
    assert build_system_prompt("kau") == SYSTEM_PROMPT
    assert SYSTEM_PROMPT.count("جامعة الملك عبدالعزيز") == 1
    assert "دليل التحويل الداخلي — صفحة 7" in SYSTEM_PROMPT


def test_ksu_system_prompt_swaps_university_and_example() -> None:
    ksu = build_system_prompt("ksu")
    assert "جامعة الملك سعود" in ksu
    assert "جامعة الملك عبدالعزيز" not in ksu
    assert "اللائحة التنظيمية للشؤون الدراسية والأكاديمية للمرحلة الجامعية" in ksu
    assert "دليل التحويل الداخلي" not in ksu
    # Non-"موثّق" Q&A wording on KSU, tag cites "سؤال رقم N".
    assert "(أسئلة وإجابات و/أو مقتطفات من" in ksu
    assert "(أسئلة وإجابات موثّقة" not in ksu
    assert "سؤال رقم 12" in ksu
    assert "سؤال موثّق" not in ksu
    # KAU wording keeps the موثّق forms.
    assert "(أسئلة وإجابات موثّقة و/أو مقتطفات من" in SYSTEM_PROMPT
    assert "سؤال موثّق رقم 12" in SYSTEM_PROMPT
    # Rules 3-5 stay identical; rules 1-2 are university-specific by design.
    for rule in ("3.", "4.", "5."):
        idx = SYSTEM_PROMPT.find(f"\n{rule} ")
        assert SYSTEM_PROMPT[idx:SYSTEM_PROMPT.find("\n", idx)] in ksu
    with pytest.raises(ValueError):
        build_system_prompt("uqu")


def test_qa_block_title_per_university() -> None:
    assert qa_block_title("kau") == QA_BLOCK_TITLE
    assert "جامعة الملك سعود" in qa_block_title("ksu")
    with pytest.raises(ValueError):
        qa_block_title("uqu")


def test_build_user_prompt_kau_unchanged() -> None:
    prompt = build_user_prompt("س?", "qa block", "doc block")
    assert prompt == (
        "السؤال: س?\n\nأسئلة وإجابات موثّقة من جامعة الملك عبدالعزيز:\nqa block\n\n"
        "مقتطفات من اللوائح والأدلة الرسمية:\ndoc block\n\n"
        "أجب وفق القواعد المذكورة، مع ذكر المصدر."
    )


def test_build_user_prompt_ksu_swaps_title() -> None:
    prompt = build_user_prompt("س?", "qa block", "doc block", university="ksu")
    assert "أسئلة وإجابات من قاعدة أسئلة جامعة الملك سعود:\nqa block" in prompt
    assert DOC_BLOCK_TITLE in prompt
    with pytest.raises(ValueError):
        build_user_prompt("س?", "qa", "doc", university="uqu")


def test_doc_line_kau_pdf_unchanged() -> None:
    assert _doc_line(_hit(KAU_PDF_MD)) == "- نص\n  [المصدر: دليل التحويل الداخلي — صفحة 7]"


def test_doc_line_kau_web_unchanged() -> None:
    assert _doc_line(_hit(KAU_WEB_MD)) == (
        "- نص\n  [المصدر: الخدمات المالية > الرسوم (https://kau.edu.sa/x)]"
    )


def test_doc_line_ksu_txt_shape() -> None:
    md = {
        "type": "txt",
        "main_section": "الطلاب",
        "service": "طلب إعادة تصحيح أوراق إجابات الاختبارات",
    }
    assert _doc_line(_hit(md), university="ksu") == (
        "- نص\n  [المصدر: الطلاب > طلب إعادة تصحيح أوراق إجابات الاختبارات]"
    )


def test_doc_line_ksu_pdf_shape_printed_page() -> None:
    from src.ingest.ksu_pdf import DOC_TITLE

    md = {"type": "pdf", "doc": DOC_TITLE, "unit": "المادة الرابعة عشر", "page": 20}
    assert _doc_line(_hit(md), university="ksu") == (
        f"- نص\n  [المصدر: {DOC_TITLE} — المادة الرابعة عشر (صفحة 14)]"
    )


def test_qa_line_tags_per_university() -> None:
    from src.context_assembly import _qa_line

    item = Retrieved(
        source_type=COL_KSU_QA, score=0.9, document="س",
        metadata={"q_id": 3, "answer": "ج"},
    )
    assert _qa_line(item, university="kau") == (
        "- س: س\n  ج: ج\n  [المصدر: سؤال موثّق رقم 3]"
    )
    assert _qa_line(item, university="ksu") == "- س: س\n  ج: ج\n  [المصدر: سؤال رقم 3]"


def test_render_context_swaps_block_title_per_university() -> None:
    ctx = AssembledContext(
        qa_items=[_hit(KAU_WEB_MD)], doc_items=[_hit(KAU_PDF_MD)]
    )
    kau_text = render_context(ctx, university="kau")
    assert kau_text.startswith(QA_BLOCK_TITLE)
    assert qa_block_title("ksu") not in kau_text
    ksu_text = render_context(ctx, university="ksu")
    assert ksu_text.startswith(qa_block_title("ksu"))
    assert QA_BLOCK_TITLE not in ksu_text


def test_get_university_gates_kau_exact() -> None:
    gates = get_university_gates("kau")
    # Values from the 2026-10-25 recalibration on the full transcribed
    # corpus (src/config.py); this test still pinned the old quartet.
    assert gates == {
        "floor_qa": 0.83,
        "floor_docs": 0.80,
        "strong_leader": 0.832,
        "gap": 0.012,
    }


def test_get_university_gates_ksu_rerank() -> None:
    from src.config import get_settings

    gates = get_university_gates("ksu")
    settings = get_settings()
    assert gates["rerank_threshold"] == settings.ksu_rerank_refuse_threshold
    assert gates["rerank_pool_size"] == float(settings.ksu_rerank_pool_size)
    with pytest.raises(ValueError):
        get_university_gates("uqu")


def test_university_available_when_registered_and_gated() -> None:
    assert is_university_available("kau") is True
    assert is_university_available("ksu") is True


def test_university_unknown_name_unavailable() -> None:
    assert is_university_available("uqu") is False


def test_university_availability_follows_gates_not_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once gates exist for a registered name it flips to available
    automatically; there is no hardcoded university check in the guard."""

    def _fake_gates(university: str) -> dict[str, float]:
        import src.config as config

        if university in config.UNIVERSITY_COLLECTIONS:
            return {
                "floor_qa": 0.0,
                "floor_docs": 0.0,
                "strong_leader": 0.0,
                "gap": 0.0,
            }
        raise ValueError(university)

    import src.config as config

    monkeypatch.setattr(config, "get_university_gates", _fake_gates)
    assert config.is_university_available("ksu") is True
    assert config.is_university_available("kau") is True
    assert config.is_university_available("uqu") is False
