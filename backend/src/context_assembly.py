"""Context assembly: relevance floors + labeled context blocks.

Retrieval sides (QA vs docs) are filtered independently; anything below its
floor is dropped. If nothing clears either floor, no context is assembled and
the pipeline must refuse to answer (known-gaps principle wired in here).
"""

from dataclasses import dataclass, field

from src.prompts import DOC_BLOCK_TITLE, qa_block_title
from src.retrieval import Retrieved


@dataclass
class AssembledContext:
    """Filtered retrieval results plus the final prompt-ready blocks."""

    qa_items: list[Retrieved] = field(default_factory=list)
    doc_items: list[Retrieved] = field(default_factory=list)
    qa_gate: str = "excluded"
    doc_gate: str = "excluded"

    @property
    def is_empty(self) -> bool:
        return not self.qa_items and not self.doc_items


def _qa_line(item: Retrieved, university: str = "kau") -> str:
    q_id = item.metadata.get("q_id", "?")
    tag = (
        f"[المصدر: سؤال موثّق رقم {q_id}]"
        if university == "kau"
        else f"[المصدر: سؤال رقم {q_id}]"
    )
    return (
        f"- س: {item.document}\n"
        f"  ج: {item.metadata.get('answer', '')}\n"
        f"  {tag}"
    )


def _kau_doc_source(md: dict) -> str:
    """KAU citation line (unchanged historical wording)."""
    if md.get("type") == "pdf":
        src = f"{md.get('doc', 'وثيقة رسمية')} — صفحة {md.get('page', '?')}"
        return src
    src = str(md.get("page_title", ""))
    section = md.get("section", "")
    if section:
        src += f" > {section}"
    if md.get("source_url"):
        src += f" ({md['source_url']})"
    return src


def _ksu_doc_source(md: dict) -> str:
    """KSU citation line for the ksu_txt/ksu_pdf metadata shapes."""
    if md.get("type") == "pdf":
        doc = md.get("doc", "وثيقة رسمية")
        unit = md.get("unit", "")
        # metadata.page is the PDF page; the printed page is 6 pages behind.
        printed = int(md.get("page", 6)) - 6
        return f"{doc} — {unit} (صفحة {printed})"
    type_ = md.get("type")
    if type_ == "txt":
        src = str(md.get("main_section", ""))
        service = md.get("service", "")
        if service:
            src += f" > {service}"
        return src
    # Unknown KSU shape: fall back to the KAU line so nothing renders empty.
    return _kau_doc_source(md)


def _doc_line(item: Retrieved, university: str = "kau") -> str:
    md = item.metadata
    src = _ksu_doc_source(md) if university == "ksu" else _kau_doc_source(md)
    return f"- {item.document}\n  [المصدر: {src}]"


def gate_side(
    results: list[Retrieved],
    floor: float,
    strong_leader: float,
    gap: float,
    max_items: int,
) -> tuple[list[Retrieved], str]:
    """Floor-filter then confidence-gate one retrieval side.

    Returns (kept_items, reason) where reason is one of:
    - "below-floor": nothing cleared the floor
    - "strong-leader": top hit scored >= strong_leader
    - "clear-gap": top hit separated from its runner-up by >= gap
    - "ambiguous-flat": survivors exist but no confident leader -> excluded
    """
    survivors = [r for r in results if r.score >= floor]
    if not survivors:
        return [], "below-floor"
    top = survivors[0].score
    if top >= strong_leader:
        return survivors[:max_items], "strong-leader"
    if len(survivors) >= 2:
        anchor = survivors[min(2, len(survivors)) - 1].score
        if top - anchor >= gap:
            return survivors[:max_items], "clear-gap"
    return [], "ambiguous-flat"


def assemble_context(
    qa_results: list[Retrieved],
    doc_results: list[Retrieved],
    floor_qa: float,
    floor_docs: float,
    max_qa: int,
    max_docs: int,
    strong_leader: float = 0.869,
    gap: float = 0.04,
) -> AssembledContext:
    """Floor + confidence-gate each side independently, then cap counts."""
    qa_items, qa_reason = gate_side(
        qa_results, floor_qa, strong_leader, gap, max_qa
    )
    doc_items, doc_reason = gate_side(
        doc_results, floor_docs, strong_leader, gap, max_docs
    )
    return AssembledContext(
        qa_items=qa_items, doc_items=doc_items,
        qa_gate=qa_reason, doc_gate=doc_reason,
    )


def render_context(context: AssembledContext, university: str = "kau") -> str:
    """Render the labeled context blocks (empty string when nothing passed)."""
    if context.is_empty:
        return ""
    blocks: list[str] = []
    if context.qa_items:
        blocks.append(
            qa_block_title(university)
            + "\n"
            + "\n".join(_qa_line(i, university) for i in context.qa_items)
        )
    if context.doc_items:
        blocks.append(
            DOC_BLOCK_TITLE
            + "\n"
            + "\n".join(_doc_line(i, university) for i in context.doc_items)
        )
    return "\n\n".join(blocks)
