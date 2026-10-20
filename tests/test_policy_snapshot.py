"""두 검색 사이 재색인 commit이 들어와도 한 statement의 정책 판만 채택한다."""

from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from reply_gate.db import connect
from reply_gate.evidence import INTENT_STAGE
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.query_rewrite import QUERY_REWRITE_STAGE
from reply_gate.testing import LexicalEmbeddingClient
from tests.test_evidence import INQUIRY_ID, _client, _collector, _intent, _rewrite


@pytest.mark.db
def test_원문과_재작성_검색의_사이에_재색인해도_한_정책판만_쓴다(
    seeded_order_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    del seeded_order_count
    embedder = LexicalEmbeddingClient(dimensions=1536)
    docs = load_policy_documents()
    # 동시 commit이 필요하므로 fixture rollback 대신 이 검사만 소유한 테이블을 쓴다.
    table = "t5_policy_snapshot_" + uuid4().hex
    with connect() as setup:
        setup.execute(
            sql.SQL("CREATE TABLE {} (LIKE policy_chunks INCLUDING ALL)").format(
                sql.Identifier(table)
            )
        )
    original = psycopg.Cursor.execute
    searched = 0

    def execute(cursor: psycopg.Cursor[Any], query: Any, params: Any = None, **kwargs: Any) -> Any:
        nonlocal searched
        text = query.as_string(cursor.connection) if hasattr(query, "as_string") else str(query)
        text = text.replace("policy_chunks", table)
        result = original(cursor, text, params, **kwargs)
        if "compatible_candidates" in text and "distance" in text:
            searched += 1
            if searched == 1:
                with connect() as writer:
                    writer.execute("UPDATE policy_chunks SET content='판B'")
        return result

    try:
        monkeypatch.setattr(psycopg.Cursor, "execute", execute)
        with connect() as setup:
            index_policy_documents(conn=setup, documents=docs, embedder=embedder)
            setup.execute("UPDATE policy_chunks SET content='판A'")
        with connect() as reader:
            transaction = reader.execute("SELECT txid_current() AS id").fetchone()
            client = _client(
                {INTENT_STAGE: [_intent("policy")], QUERY_REWRITE_STAGE: [_rewrite("배송 방법")]}
            )
            result = _collector(
                client, top_k=26, threshold=-1, abstention_gate_enabled=False
            ).collect(
                inquiry_id=INQUIRY_ID,
                content="환불 방법",
                order_no=None,
                app_conn=reader,
                readonly_conn=reader,
            )
            assert result.evidence
            assert all(
                "판A" in item.content and "판B" not in item.content for item in result.evidence
            )
            assert searched == 1
            assert reader.execute("SELECT txid_current() AS id").fetchone() == transaction
    finally:
        monkeypatch.setattr(psycopg.Cursor, "execute", original)
        with connect() as cleanup:
            cleanup.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))
