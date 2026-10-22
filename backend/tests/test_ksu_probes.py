"""Tests for src.ksu_probes (pure functions only, no retrieval/I-O)."""

import pytest

from src.ksu_probes import (
    DOMAIN_ADJACENT_WRONG_KSU,
    OFF_TOPIC_KSU,
    RELEVANT_KSU,
    _gate_pass,
    _side_pass,
    split_tune_test,
)


def test_probe_category_counts() -> None:
    assert len(RELEVANT_KSU) == 15
    assert len(DOMAIN_ADJACENT_WRONG_KSU) == 12
    assert len(OFF_TOPIC_KSU) == 6


def test_probe_categories_disjoint() -> None:
    relevant = set(RELEVANT_KSU)
    daw = set(DOMAIN_ADJACENT_WRONG_KSU)
    off = set(OFF_TOPIC_KSU)
    assert not relevant & daw
    assert not relevant & off
    assert not daw & off


def test_split_tune_test_deterministic() -> None:
    tune_a, test_a = split_tune_test(RELEVANT_KSU)
    tune_b, test_b = split_tune_test(RELEVANT_KSU)
    assert tune_a == tune_b
    assert test_a == test_b


def test_split_tune_test_partition() -> None:
    tune, test = split_tune_test(RELEVANT_KSU)
    assert len(tune) == 8
    assert len(test) == 7
    assert set(tune) | set(test) == set(RELEVANT_KSU)
    assert not set(tune) & set(test)


def test_split_tune_test_even_and_odd_counts() -> None:
    tune, test = split_tune_test(DOMAIN_ADJACENT_WRONG_KSU)
    assert (len(tune), len(test)) == (6, 6)
    tune, test = split_tune_test(OFF_TOPIC_KSU)
    assert (len(tune), len(test)) == (3, 3)


@pytest.mark.parametrize(
    ("scores", "floor", "strong", "gap", "expected_pass", "expected_reason"),
    [
        ([0.50, 0.49], 0.83, 0.87, 0.04, False, "below-floor"),
        ([0.88, 0.86], 0.83, 0.87, 0.04, True, "strong-leader"),
        (
        [0.85, 0.80, 0.79],
        0.79,
        0.87,
        0.04,
        True,
        "clear-gap",
    ),
        ([0.85, 0.84, 0.83], 0.83, 0.87, 0.04, False, "ambiguous-flat"),
        ([], 0.83, 0.87, 0.04, False, "below-floor"),
    ],
)
def test_gate_pass_semantics(
    scores: list[float],
    floor: float,
    strong: float,
    gap: float,
    expected_pass: bool,
    expected_reason: str,
) -> None:
    passed, reason = _gate_pass(scores, floor, strong, gap)
    assert passed is expected_pass
    assert reason == expected_reason


def test_side_pass_answers_when_one_side_strong() -> None:
    gates = {"floor_qa": 0.83, "floor_docs": 0.79, "strong_leader": 0.869, "gap": 0.04}
    answered, why, _state = _side_pass(
        {"qa": [0.88, 0.86], "docs": []}, gates
    )
    assert answered is True
    assert why == "pass"


def test_side_pass_refuses_when_both_sides_below_floor() -> None:
    gates = {"floor_qa": 0.83, "floor_docs": 0.79, "strong_leader": 0.869, "gap": 0.04}
    answered, why, _state = _side_pass(
        {"qa": [0.82, 0.80], "docs": [0.78]}, gates
    )
    assert answered is False
    assert "qa:below-floor" in why
    assert "doc:below-floor" in why
