"""주입 임베더도 검색과 인덱스 DB 접근 전에 전체 벡터 계약을 지킨다."""

from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate import evidence, policy_index
from reply_gate.evidence import INTENT_STAGE
from reply_gate.llm import EmbeddingResult
from reply_gate.testing import LexicalEmbeddingClient
from tests.test_evidence import INQUIRY_ID, _client, _collector, _intent


@pytest.mark.parametrize("kind", ["count", "dimension", "nan", "inf", "zero"])
def test_인덱싱은_잘못된_후반_벡터도_DB접근전에_거부한다(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = policy_index.load_policy_documents()
    count = sum(len(doc.chunks) for doc in docs)
    vectors = [[1.0] + [0.0] * 1535 for _ in range(count)]
    if kind == "count":
        vectors.pop()
    elif kind == "dimension":
        vectors[-1].pop()
    elif kind == "zero":
        vectors[-1] = [0.0] * 1536
    else:
        vectors[-1][0] = float(kind)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    monkeypatch.setattr(
        embedder, "embed", lambda **kwargs: EmbeddingResult(vectors=vectors, total_tokens=0)
    )

    def db_access(*args: object) -> None:
        raise AssertionError("벡터 전체 검증 전에는 DB를 만지면 안 된다")

    monkeypatch.setattr(policy_index, "register_vector", db_access)
    with pytest.raises(ValueError):
        policy_index.index_policy_documents(
            conn=cast(psycopg.Connection[DictRow], None), documents=docs, embedder=embedder
        )


@pytest.mark.parametrize("vectors", [[], [[0.0] * 1536], [[float("nan")] + [0.0] * 1535]])
def test_문의_임베딩_오류는_검색이나_무근거_인계로_위장하지_않는다(
    vectors: list[list[float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    collector = _collector(
        _client({INTENT_STAGE: [_intent("policy")]}), query_rewrite_enabled=False
    )
    monkeypatch.setattr(
        collector._embedder,
        "embed",
        lambda **kwargs: EmbeddingResult(vectors=vectors, total_tokens=0),
    )

    def unexpected(**kwargs: Any) -> None:
        raise AssertionError("잘못된 벡터로 검색하면 안 된다")

    monkeypatch.setattr(evidence, "search_policy_rankings", unexpected)
    with pytest.raises(ValueError):
        collector.collect(
            inquiry_id=INQUIRY_ID,
            content="환불 기간",
            order_no=None,
            app_conn=cast(psycopg.Connection[DictRow], None),
            readonly_conn=cast(psycopg.Connection[DictRow], None),
        )
