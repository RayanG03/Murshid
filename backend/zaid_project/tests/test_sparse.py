"""اختبارات البحث اللفظي بأوزان bge-m3 (token_weights و lexical_score في embedding.py).

لا تحمّل النموذج: تختبر تجميع الأوزان وحساب الدرجة فقط.

التشغيل:
    python -m pytest tests/test_sparse.py
"""

from embedding import lexical_score, token_weights


def test_token_weights_keeps_highest_weight_of_repeated_token() -> None:
    assert token_weights([5, 7, 5], [0.2, 0.4, 0.3], skip_ids=set()) == {"5": 0.3, "7": 0.4}


def test_token_weights_drops_special_and_zero_tokens() -> None:
    weights = token_weights([0, 5, 7, 2], [0.9, 0.2, 0.0, 0.8], skip_ids={0, 2})
    assert weights == {"5": 0.2}


def test_lexical_score_sums_shared_tokens_only() -> None:
    query = {"5": 0.5, "7": 0.2}
    document = {"5": 0.4, "9": 1.0}
    assert abs(lexical_score(query, document) - 0.2) < 1e-9


def test_lexical_score_is_zero_without_shared_tokens() -> None:
    assert lexical_score({"5": 0.5}, {"9": 1.0}) == 0.0
