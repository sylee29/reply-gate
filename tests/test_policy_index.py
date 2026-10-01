"""정책 문서 파싱·적재·검색 테스트.

적재/검색은 DB 가 필요하지만 **임베딩 API 키는 필요 없다** — 결정론 임베딩 대역으로
배관(청킹 → 적재 → 유사도 정렬 → 임계값 필터)을 끝까지 돌린다. 임베딩 품질 자체는
실제 키가 있어야 확인할 수 있고 그건 골든셋 평가의 몫이다.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Self, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.abc import Params, Query
from psycopg.rows import DictRow

from reply_gate.db import connect
from reply_gate.evidence import adopt_policy_hits
from reply_gate.llm import EmbeddingClient
from reply_gate.policy_index import (
    DEFAULT_POLICY_DIR,
    PlantedKind,
    PolicyIndexProvenanceError,
    PolicySearchHit,
    index_policy_documents,
    load_policy_documents,
    parse_policy_document,
    search_policy_chunks,
)
from reply_gate.testing import LexicalEmbeddingClient


def _count(cur: psycopg.Cursor[DictRow], sql: str) -> int:
    """단일 카운트 조회. fetchone() 이 None 이면 그 자체가 실패다."""
    cur.execute(sql)
    row = cur.fetchone()
    assert row is not None
    return int(row["n"])


SAMPLE = """\
---
slug: sample
title: 샘플 정책
---

<!-- planted: none -->
## 9-1 첫 조항

첫 조항 본문이다.

<!-- planted: decoy; note: 전화번호를 뺐다 -->
## 9-2 둘째 조항

둘째 조항 본문이다.
계속 이어지는 줄.
"""


# ── 파싱 (DB 불필요) ────────────────────────────────────────────────────────


def test_조항_단위로_쪼개고_근거_ID_를_붙인다() -> None:
    document = parse_policy_document(SAMPLE)

    assert document.slug == "sample"
    assert [chunk.article for chunk in document.chunks] == ["9-1", "9-2"]
    assert document.chunks[0].evidence_id == "policy:sample:9-1"
    assert document.chunks[1].content == "둘째 조항 본문이다.\n계속 이어지는 줄."


def test_심은_장치를_메타데이터로_읽는다() -> None:
    document = parse_policy_document(SAMPLE)

    assert document.chunks[0].planted is PlantedKind.NONE
    assert document.chunks[1].planted is PlantedKind.DECOY
    assert document.chunks[1].planted_note == "전화번호를 뺐다"


def test_임베딩_텍스트에_문서와_조항_제목이_들어간다() -> None:
    chunk = parse_policy_document(SAMPLE).chunks[0]

    assert "샘플 정책" in chunk.embedding_text
    assert "첫 조항" in chunk.embedding_text


@pytest.mark.parametrize(
    "bad",
    [
        "## 1-1 제목\n본문\n",  # front matter 없음
        "---\nslug: x\n---\n\n## 1-1 제목\n본문\n",  # title 없음
        "---\nslug: x\ntitle: y\n---\n\n본문만 있다\n",  # 조항 없음
        "---\nslug: x\ntitle: y\n---\n\n## 1-1 제목\n\n## 1-2 제목\n본문\n",  # 빈 조항
    ],
)
def test_잘못된_문서는_거부한다(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_policy_document(bad)


# ── 저장소의 실제 정책 문서 (DB 불필요) ─────────────────────────────────────


def test_저장소_정책_문서에_기각_유발_장치가_살아_있다() -> None:
    """미끼·모호·상충 조항은 데모와 지표의 신뢰성 장치다 — 정리 명목으로 사라지면 안 된다."""
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    chunks = [chunk for document in documents for chunk in document.chunks]
    planted = [chunk.planted for chunk in chunks]

    assert len(chunks) >= 20
    assert planted.count(PlantedKind.DECOY) >= 2, "미끼 조항이 없으면 기각 장면이 재현되지 않는다"
    assert planted.count(PlantedKind.AMBIGUOUS) >= 1
    assert planted.count(PlantedKind.CONFLICTING) >= 2, "상충은 최소 2개 조항이 짝을 이룬다"


def test_미끼_조항에는_패턴형_값이_실제로_없다() -> None:
    """미끼가 값을 담고 있으면 모델이 지어낼 이유가 없어져 장치가 무력해진다."""
    import re

    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    decoys = [
        chunk
        for document in documents
        for chunk in document.chunks
        if chunk.planted is PlantedKind.DECOY
    ]
    phone_or_email = re.compile(
        r"(?<![0-9])(1[5-8][0-9]{2}|0[1-9][0-9]?)[-. ]?[0-9]{3,4}[-. ]?[0-9]{4}(?![0-9])|@"
    )

    for chunk in decoys:
        assert not phone_or_email.search(chunk.content), f"{chunk.evidence_id} 에 패턴형 값이 있다"


def test_근거_ID_는_문서_전체에서_유일하다() -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    ids = [chunk.evidence_id for document in documents for chunk in document.chunks]

    assert len(ids) == len(set(ids))
    assert all(evidence_id.startswith("policy:") for evidence_id in ids)


# ── 적재·검색 (DB 필요, 임베딩 키 불필요) ────────────────────────────────────


@pytest.mark.db
def test_적재_후_조항이_전부_들어가고_재실행해도_중복되지_않는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    expected = sum(len(document.chunks) for document in documents)

    first = index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)
    second = index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)

    assert first.upserted == expected
    assert second.upserted == expected
    assert second.deleted == 0
    with app_conn.cursor() as cur:
        assert _count(cur, "SELECT count(*) AS n FROM policy_chunks") == expected
        assert _count(cur, "SELECT count(*) AS n FROM policy_chunks WHERE embedding IS NULL") == 0


@pytest.mark.db
def test_문서에서_사라진_조항은_삭제된다(app_conn: psycopg.Connection[DictRow]) -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)

    trimmed = index_policy_documents(conn=app_conn, documents=documents[:1], embedder=embedder)

    assert trimmed.deleted > 0
    with app_conn.cursor() as cur:
        assert _count(cur, "SELECT count(*) AS n FROM policy_chunks") == len(documents[0].chunks)


@pytest.mark.db
def test_유사도_검색이_관련_조항을_돌려준다(app_conn: psycopg.Connection[DictRow]) -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)

    query = embedder.embed(stage="inquiry", texts=["환불 신청 기간이 어떻게 되나요"]).vectors[0]
    hits = search_policy_chunks(
        conn=app_conn,
        query_vector=query,
        top_k=5,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
    )

    assert hits, "검색 결과가 비면 안 된다"
    assert hits[0].document_slug == "refund"
    assert hits == sorted(hits, key=lambda hit: hit.similarity, reverse=True)


@pytest.mark.db
def test_검색은_컷_전_상위_top_k_후보를_그대로_돌려준다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """자르는 것은 `LIMIT top_k` 하나다 — 컷은 `evidence.adopt_policy_hits` 가 건다.

    검색이 컷을 먼저 걸면 질의 단위 기권 게이트가 **짧아진 슬라이스**로 산포를 재게 되어
    오프라인 격자와 런타임이 다른 수를 낸다(`docs/tracking/decisions/0014`). 그래서 컷
    미만 후보가 여기서 살아 있는 것이 계약이다.
    """
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)
    query = embedder.embed(stage="inquiry", texts=["환불 신청 기간"]).vectors[0]

    hits = search_policy_chunks(
        conn=app_conn,
        query_vector=query,
        top_k=10,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
    )

    assert len(hits) == 10
    # 양성 대조 — 대역 임베딩의 상위 10건에는 제품 컷(0.30) 미만이 실제로 섞여 있다.
    assert [hit for hit in hits if hit.similarity < 0.30]
    # 그리고 그 후보를 버리는 것은 채택 단계다.
    assert all(
        hit.similarity >= 0.30
        for hit in adopt_policy_hits(
            candidates=hits, top_k=10, similarity_threshold=0.30, gate=None
        ).hits
    )


# ── 임베딩 provenance (불변식: 저장된 벡터와 질의는 같은 공간에서 나와야 한다) ──


@pytest.mark.db
def test_적재는_벡터와_함께_그것을_만든_모델과_차원을_적는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)

    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)

    with app_conn.cursor() as cur:
        cur.execute("SELECT DISTINCT embedding_model, embedding_dimensions FROM policy_chunks")
        assert cur.fetchall() == [
            {"embedding_model": embedder.model, "embedding_dimensions": embedder.dimensions}
        ]


def test_출처를_밝히지_못하는_임베더는_적재하지_않는다() -> None:
    """모델 이름이 빈 벡터는 나중에 같은 공간인지 판정할 근거가 없다.

    커넥션 없이 도는 것 자체가 확인이다 — 거부가 DB 왕복이나 임베딩 호출보다 앞에 있다.
    """
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    # 적재기의 선검사를 직접 겨냥한다. 어휘 대역은 빈 출처를 생성자에서 이미 거부한다.
    embedder = cast(EmbeddingClient, SimpleNamespace(dimensions=1536, model="   "))

    with pytest.raises(ValueError, match="임베딩 모델 이름이 비어 있다"):
        index_policy_documents(
            conn=cast(psycopg.Connection[DictRow], None), documents=documents, embedder=embedder
        )


@pytest.mark.db
def test_다른_모델의_질의는_유사도를_내지_않고_거부된다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """차원이 같으면 pgvector 는 코사인을 계산해 준다 — 오류가 아니라 **근거 없음**으로
    위장되는 경로이므로, 계산 전에 코드가 막는다."""
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    indexed = LexicalEmbeddingClient(dimensions=1536, model="stub:model-a")
    index_policy_documents(conn=app_conn, documents=documents, embedder=indexed)
    query = indexed.embed(stage="inquiry", texts=["환불 신청 기간"]).vectors[0]

    with pytest.raises(PolicyIndexProvenanceError) as excinfo:
        search_policy_chunks(
            conn=app_conn,
            query_vector=query,
            top_k=5,
            embedding_model="stub:model-b",
            embedding_dimensions=1536,
        )

    assert "stub:model-a" in str(excinfo.value)
    assert "stub:model-b" in str(excinfo.value)
    assert "index_policies" in str(excinfo.value)


@pytest.mark.db
def test_차원만_다른_질의도_거부된다(app_conn: psycopg.Connection[DictRow]) -> None:
    """모델 이름이 같아도 차원이 다르면 다른 공간이다(`3-large` 는 1536·3072 둘 다 낸다)."""
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536, model="stub:model-a")
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)
    query = embedder.embed(stage="inquiry", texts=["환불 신청 기간"]).vectors[0]

    with pytest.raises(PolicyIndexProvenanceError):
        search_policy_chunks(
            conn=app_conn,
            query_vector=query,
            top_k=5,
            embedding_model="stub:model-a",
            embedding_dimensions=3072,
        )


@pytest.mark.db
def test_두_공간이_섞인_인덱스는_어느_질의로도_거부된다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """적재가 도중에 끊겨 행마다 출처가 다를 수 있다 — 그 상태는 어떤 질의와도 맞지 않는다."""
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536, model="stub:model-a")
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)
    app_conn.execute(
        "UPDATE policy_chunks SET embedding_model = 'stub:model-b' WHERE evidence_id = %s",
        (documents[0].chunks[0].evidence_id,),
    )
    query = embedder.embed(stage="inquiry", texts=["환불 신청 기간"]).vectors[0]

    with pytest.raises(PolicyIndexProvenanceError, match=r"stub:model-a.*stub:model-b"):
        search_policy_chunks(
            conn=app_conn,
            query_vector=query,
            top_k=5,
            embedding_model="stub:model-a",
            embedding_dimensions=1536,
        )


@pytest.mark.db
def test_적재_전_빈_인덱스는_불일치가_아니다(app_conn: psycopg.Connection[DictRow]) -> None:
    """검색할 것이 없는 것과 다른 공간을 비교하는 것은 다르다."""
    app_conn.execute("DELETE FROM policy_chunks")

    hits = search_policy_chunks(
        conn=app_conn,
        query_vector=[0.0] * 1536,
        top_k=5,
        embedding_model="stub:model-a",
        embedding_dimensions=1536,
    )

    assert hits == []


# ── 실제 두 연결의 snapshot 경계: 공유 정책 테이블/sequence는 쓰지 않는다 ──


@contextmanager
def _isolated_policy_index(
    *,
    reindex_after_read: bool = False,
) -> Iterator[
    tuple[psycopg.Connection[DictRow], psycopg.Connection[DictRow], sql.Identifier, list[str]]
]:
    table_name = "test_policy_snapshot_" + uuid4().hex
    table = sql.Identifier("public", table_name)
    events: list[str] = []
    with connect(statement_timeout_ms=5000) as reader, connect(statement_timeout_ms=5000) as writer:
        reader.isolation_level = psycopg.IsolationLevel.READ_COMMITTED
        created = False
        try:
            # Identity/default를 복사하지 않아 shared sequence도 소비하지 않는다.
            writer.execute(sql.SQL("CREATE TABLE {} (LIKE public.policy_chunks)").format(table))
            created = True
            writer.execute(
                sql.SQL(
                    "INSERT INTO {} (id, evidence_id, document_slug, document_title, article, "
                    "article_title, content, embedding, embedding_model, embedding_dimensions, "
                    "created_at) VALUES (1, 'policy:snapshot:1', 'snapshot', '시험', '1', '시험', "
                    "'snapshot-a', %s::vector, 'stub:snapshot-a', 1536, now())"
                ).format(table),
                ("[1," + ",".join("0" for _ in range(1535)) + "]",),
            )
            writer.commit()
            fired = False

            class SnapshotCursor(psycopg.Cursor[DictRow]):
                def execute(
                    self,
                    query: Query,
                    params: Params | None = None,
                    *,
                    prepare: bool | None = None,
                    binary: bool | None = None,
                ) -> Self:
                    nonlocal fired
                    if not isinstance(query, str | bytes | sql.SQL | sql.Composed):
                        assert params is None
                        return super().execute(query, prepare=prepare, binary=binary)
                    text = (
                        query.as_string(self.connection)
                        if isinstance(query, sql.Composable)
                        else query
                    )
                    if isinstance(text, bytes):
                        text = text.decode("utf-8")
                    targeted = "policy_chunks" in text
                    if targeted:
                        query = text.replace("policy_chunks", table.as_string(self.connection))
                        events.append("policy_select")
                    result = super().execute(query, params, prepare=prepare, binary=binary)
                    if targeted and reindex_after_read and not fired:
                        fired = True
                        events.append("first_execute_completed")
                        writer.execute(
                            sql.SQL(
                                "UPDATE {} SET embedding_model = 'stub:snapshot-b', "
                                "content = 'snapshot-b'"
                            ).format(table)
                        )
                        writer.commit()
                        events.append("writer_committed_before_fetch")
                    return result

            reader.cursor_factory = SnapshotCursor
            yield reader, writer, table, events
        finally:
            reader.rollback()
            writer.rollback()
            if created:
                writer.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
                writer.commit()
                row = writer.execute(
                    "SELECT to_regclass(%s) IS NULL AS cleaned", (f"public.{table_name}",)
                ).fetchone()
                assert row is not None and row["cleaned"], (
                    f"독립 시험 테이블 청소 실패: {table_name}"
                )
                writer.rollback()


def _snapshot_search(conn: psycopg.Connection[DictRow], *, top_k: int = 5) -> list[PolicySearchHit]:
    return search_policy_chunks(
        conn=conn,
        query_vector=[1.0] + [0.0] * 1535,
        top_k=top_k,
        embedding_model="stub:snapshot-a",
        embedding_dimensions=1536,
    )


@pytest.mark.db
def test_독립_정책_조회_직후_재색인이_커밋되어도_검사한_스냅샷을_돌려준다() -> None:
    """두 SELECT 사이 commit이면 A 출처 검사를 통과한 B 본문이 나오던 실제 경로다."""
    with _isolated_policy_index(reindex_after_read=True) as (reader, _writer, _table, events):
        hits = _snapshot_search(reader)
        assert [hit.content for hit in hits] == ["snapshot-a"]
        assert events == [
            "policy_select",
            "first_execute_completed",
            "writer_committed_before_fetch",
        ]


@pytest.mark.db
@pytest.mark.parametrize(
    ("model", "dimensions"), [("stub:snapshot-b", 1536), ("stub:snapshot-a", 3072)]
)
def test_독립_정책_출처가_다르면_벡터_차원_오류보다_먼저_거부한다(
    model: str, dimensions: int
) -> None:
    """WHERE 문면 순서만 믿으면 pgvector 연산이 출처 오류를 가릴 수 있다."""
    with (
        _isolated_policy_index() as (reader, _writer, _table, _events),
        pytest.raises(PolicyIndexProvenanceError, match="재색인"),
    ):
        search_policy_chunks(
            conn=reader,
            query_vector=[1.0] + [0.0] * (dimensions - 1),
            top_k=5,
            embedding_model=model,
            embedding_dimensions=dimensions,
        )


@pytest.mark.db
@pytest.mark.parametrize("empty", [True, False])
def test_독립_정책이_비었거나_NULL_벡터뿐이면_빈_후보를_돌려준다(empty: bool) -> None:
    with _isolated_policy_index() as (reader, writer, table, _events):
        statement = "DELETE FROM {}" if empty else "UPDATE {} SET embedding = NULL"
        writer.execute(sql.SQL(statement).format(table))
        writer.commit()
        assert _snapshot_search(reader) == []


@pytest.mark.db
def test_독립_정책의_동점_절단과_혼합_출처_오류를_보존한다() -> None:
    with _isolated_policy_index() as (reader, writer, table, _events):
        writer.execute(
            sql.SQL(
                "UPDATE {} SET id = 3, evidence_id = 'policy:snapshot:3', article = '3'"
            ).format(table)
        )
        for number in (2, 1):
            writer.execute(
                sql.SQL(
                    "INSERT INTO {} SELECT %s, %s, document_slug, document_title, %s, article_title, "
                    "content, embedding, embedding_model, embedding_dimensions, created_at "
                    "FROM {} WHERE id = 3"
                ).format(table, table),
                (number, f"policy:snapshot:{number}", str(number)),
            )
        writer.commit()
        hits = _snapshot_search(reader, top_k=2)
        assert [hit.evidence_id for hit in hits] == ["policy:snapshot:1", "policy:snapshot:2"]
        assert [hit.similarity for hit in hits] == [1.0, 1.0]
        writer.execute(
            sql.SQL("UPDATE {} SET embedding_model = 'stub:snapshot-b' WHERE id = 2").format(table)
        )
        writer.commit()
        with pytest.raises(PolicyIndexProvenanceError, match=r"stub:snapshot-a.*stub:snapshot-b"):
            _snapshot_search(reader)


def test_top_k가_0이면_커넥션을_열지_않고_빈_후보를_돌려준다() -> None:
    assert _snapshot_search(cast(psycopg.Connection[DictRow], None), top_k=0) == []
