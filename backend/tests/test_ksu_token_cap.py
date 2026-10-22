"""KSU chunk token-cap tests (step 0): budget, headings, losslessness.

No model is loaded: a cheap fake token counter (chars // 3, min 1) supplies
token_len to the builder functions, which keeps them model-free while still
forcing real splits at sentence/line boundaries.

Losslessness model: parts may drop only the block joiner at a part boundary
(heading repetitions are excluded); every atomic unit of the source body
(a pdf table row, a txt sentence/paragraph segment, with its punctuation)
must appear whole somewhere, in source order, exactly once. Parts are walked
against a cursor over the atom list to prove no atom is ever cut, duplicated
or reordered.
"""

import re
from collections.abc import Callable

import pytest

from src.config import get_settings
from src.ingest.ksu_pdf import (
    DOC_TITLE,
    build_pdf_chunks,
    build_units,
    extract_pages,
)
from src.ingest.ksu_txt import (
    NUMBERED_RE,
    build_txt_chunks,
    parse_sections,
)
from src.ingest.token_cap import _boundary_indices

settings = get_settings()


def _fake_token_len() -> Callable[[str], int]:
    """Cheap deterministic counter: ~3 chars per token (no model)."""
    return lambda text: max(1, len(text) // 3)


def _txt_heading(display: str) -> str:
    """Heading block of a stored txt chunk (FAQ items carry two lines)."""
    lines = display.split("\n")
    if len(lines) > 2 and lines[1].rstrip().endswith("؟"):
        return "\n".join(lines[:2])
    return lines[0]


def _txt_body(display: str) -> str:
    return display[len(_txt_heading(display)) + 1:]


def test_pdf_chunks_within_token_budget_and_headed() -> None:
    """Every pdf part fits ksu_chunk_max_tokens and repeats its unit heading."""
    path = settings.sources_dir / "KSU" / "اللائحة التنظيمية للشؤون الدراسية24.pdf"
    if not path.exists():
        pytest.skip("KSU regulation PDF not present")
    budget = settings.ksu_chunk_max_tokens
    chunks = build_pdf_chunks(path, settings.chunk_max_chars,
                              token_len=_fake_token_len(), max_tokens=budget)
    assert chunks
    for chunk in chunks:
        heading = f"{DOC_TITLE} — {chunk.unit}"
        assert chunk.display.startswith(heading + "\n"), chunk.page
        assert _fake_token_len()(chunk.display) <= budget


def test_pdf_parts_concatenate_to_source_lines() -> None:
    """Part bodies are contiguous line runs of the unit; no row cut mid-line."""
    path = settings.sources_dir / "KSU" / "اللائحة التنظيمية للشؤون الدراسية24.pdf"
    if not path.exists():
        pytest.skip("KSU regulation PDF not present")
    budget = settings.ksu_chunk_max_tokens
    chunks = build_pdf_chunks(path, settings.chunk_max_chars,
                              token_len=_fake_token_len(), max_tokens=budget)
    units = {
        unit: "\n".join(lines).strip()
        for unit, _page, lines in build_units(extract_pages(path))
        if "\n".join(lines).strip()
    }
    assert set(c.unit for c in chunks) == set(units)
    for unit, body in units.items():
        parts = [c.display.split("\n", 1)[1] for c in chunks if c.unit == unit]
        joined = "".join(parts)
        assert "".join(joined.split()) == "".join(body.split()), unit
        source_lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
        cursor = 0
        for part_body in parts:
            part_lines = [ln.strip() for ln in part_body.split("\n") if ln.strip()]
            for line in part_lines:
                assert cursor < len(source_lines), unit
                assert line == source_lines[cursor], (unit, line, cursor)
                cursor += 1
        assert cursor == len(source_lines), unit


def txt_atoms(body: str) -> list[str]:
    """Sentence/paragraph atoms of a txt body (separator chars stay behind)."""
    cuts = [
        *sorted(set(_boundary_indices(body, "sentences"))), len(body)
    ]
    atoms: list[str] = []
    pos = 0
    for cut in cuts:
        atom = body[pos:cut].strip()
        if atom:
            atoms.append(atom)
        pos = cut
    return atoms


def test_txt_chunks_within_token_budget_and_headed() -> None:
    """Every txt part fits ksu_chunk_max_tokens and repeats its heading."""
    path = settings.sources_dir / "KSU" / "university documents.txt"
    if not path.exists():
        pytest.skip("KSU text dump not present")
    budget = settings.ksu_chunk_max_tokens
    chunks = build_txt_chunks(path, settings.chunk_max_chars,
                              token_len=_fake_token_len(), max_tokens=budget)
    assert chunks
    for chunk in chunks:
        md = chunk.metadata
        heading = _txt_heading(chunk.display)
        assert heading.split(" > ")[0] == md["main_section"]
        assert heading == md["main_section"] or heading.startswith(
            f"{md['main_section']} > {md['service']}"
        )
        assert _fake_token_len()(chunk.display) <= budget


def test_txt_parts_concatenate_to_source() -> None:
    """Part bodies reconstruct the section body (heading repetitions aside)."""
    path = settings.sources_dir / "KSU" / "university documents.txt"
    if not path.exists():
        pytest.skip("KSU text dump not present")
    budget = settings.ksu_chunk_max_tokens
    text = path.read_text(encoding="utf-8-sig")
    chunks = build_txt_chunks(path, settings.chunk_max_chars,
                              token_len=_fake_token_len(), max_tokens=budget)
    grouped: dict[tuple[str, str], list[str]] = {}
    for chunk in chunks:
        md = chunk.metadata
        grouped.setdefault((md["main_section"], md["service"]), []).append(
            chunk.display
        )
    sections = {
        (s.main_section, s.service): s.body for s in parse_sections(text) if s.body
    }
    assert set(grouped) == set(sections)

    def heading_indices(atoms: list[str]) -> set[int]:
        """Atoms that live only in chunk headings (FAQ block + questions).

        The FAQ question text is stored in chunk headings, not in bodies.
        Numbered question lines atomize as marker ("1.") + question ("...؟")
        when the marker ends with sentence punctuation, so both must go.
        """
        drops: set[int] = set()
        for i, atom in enumerate(atoms):
            if re.fullmatch(r"\(([^)]*شائع[^)]*)\)", atom):
                drops.add(i)
            elif (
                NUMBERED_RE.match(atom) and not NUMBERED_RE.match(atom).group(2).strip()
            ):
                nxt = atoms[i + 1] if i + 1 < len(atoms) else ""
                if nxt.rstrip().endswith("؟"):
                    drops.add(i)
                    drops.add(i + 1)
            elif NUMBERED_RE.match(atom) and atom.rstrip().endswith("؟"):
                drops.add(i)
        return drops

    for key, body in sections.items():
        atoms_all = txt_atoms(body)
        drops = heading_indices(atoms_all)
        atoms = [a for i, a in enumerate(atoms_all) if i not in drops]
        cursor = 0
        for display in grouped[key]:
            body_part = _txt_body(display)
            for piece in body_part.split("\n\n"):
                ps = "".join(piece.split())
                if not ps:
                    continue
                assembled = ""
                while cursor < len(atoms):
                    assembled += atoms[cursor]
                    cursor += 1
                    if "".join(assembled.split()) == ps:
                        break
                else:
                    pytest.fail(f"atom stream short for section {key}: {ps[:60]}")
                assert "".join(assembled.split()) == ps, key
        assert cursor == len(atoms), key
