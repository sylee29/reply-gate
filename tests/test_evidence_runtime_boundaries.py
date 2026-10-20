"""실제 DB 실행 결과와 운영 오류를 업무 판정으로 잘못 접지 않는다."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from decimal import Decimal
from typing import Any

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.contracts import Evidence, EvidenceSource, sql_evidence_id
from reply_gate.evidence import (
    INTENT_STAGE,
    SQL_GENERATION_STAGE,
    SqlEvidenceSnapshot,
    _jsonable,
    _sql_evidence_texts,
    execute_validated_query,
)
from reply_gate.records import load_inquiry, save_inquiry
from reply_gate.sql_guard import validate_sql
from tests.test_evidence import INQUIRY, INQUIRY_ID, _client, _collector, _intent, _sql
from tests.test_records import _processed


@pytest.fixture
def sample_order(ro_conn: psycopg.Connection[DictRow]) -> dict[str, Any]:
    row = ro_conn.execute("SELECT * FROM orders ORDER BY order_no LIMIT 1").fetchone()
    assert row is not None
    return row


@pytest.mark.parametrize(
    "value, expected",
    [(float("nan"), "NaN"), (float("inf"), "Infinity"), (float("-inf"), "-Infinity")],
)
def test_비유한_float는_JSON이_표현하는_명시적_문자열이_된다(value: float, expected: str) -> None:
    assert _jsonable(value) == expected
    assert json.loads(json.dumps(_jsonable(value), allow_nan=False)) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, 1),
        (1.25, 1.25),
        (None, None),
        (True, True),
        (Decimal("1.25"), "1.25"),
        (dt.date(2026, 10, 12), "2026-10-12"),
    ],
)
def test_정상_SQL_값의_JSON_표현은_유지된다(value: object, expected: object) -> None:
    assert _jsonable(value) == expected


@pytest.mark.db
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_비유한_실제_SQL_결과도_근거와_jsonb_스냅샷을_왕복한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
    literal: str,
) -> None:
    order_no = sample_order["order_no"]
    query = validate_sql(
        f"SELECT CAST('{literal}' AS double precision) AS amount FROM orders WHERE order_no = '{order_no}'",
        order_no=order_no,
        max_rows=50,
    )
    rows = execute_validated_query(conn=ro_conn, query=query)
    assert rows == ({"amount": literal},)
    content, evidence_text = _sql_evidence_texts(
        sql=query.sql, rows=rows, pii_safe_output_columns=query.pii_safe_output_columns
    )
    evidence_id = sql_evidence_id(inquiry_id=INQUIRY_ID, sequence=1)
    processed = replace(
        _processed(INQUIRY_ID),
        evidence=(
            Evidence(
                id=evidence_id,
                source=EvidenceSource.SQL,
                content=content,
                evidence_text=evidence_text,
            ),
        ),
        sql_snapshots=(
            SqlEvidenceSnapshot(
                evidence_id=evidence_id, sequence=1, query_sql=query.sql, result_rows=rows
            ),
        ),
    )
    save_inquiry(conn=app_conn, processed=processed)
    restored = load_inquiry(conn=app_conn, inquiry_id=INQUIRY_ID)
    assert restored is not None
    assert restored.sql_snapshots == processed.sql_snapshots
    assert restored.evidence == processed.evidence


@pytest.mark.db
def test_SQL_생성_사이에_끊긴_연결은_운영_오류로_전파된다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_no = sample_order["order_no"]
    query = f"SELECT status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(query), _sql(query)]}
    )
    original = client.complete_json

    def disconnect(**kwargs: Any) -> Any:
        result = original(**kwargs)
        if kwargs["stage"] == SQL_GENERATION_STAGE:
            ro_conn.close()
        return result

    monkeypatch.setattr(client, "complete_json", disconnect)
    with pytest.raises(psycopg.OperationalError):
        _collector(client).collect(
            inquiry_id=INQUIRY_ID,
            content=INQUIRY,
            order_no=order_no,
            app_conn=app_conn,
            readonly_conn=ro_conn,
        )
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 1


@pytest.mark.db
def test_실제_statement_timeout은_SQL을_한번_재생성하고_정상조회한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reply_gate import evidence as evidence_module
    from reply_gate.contracts import EscalationReason

    order_no = sample_order["order_no"]
    sql = f"SELECT status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(sql), _sql(sql)]}
    )
    count = 0
    original = evidence_module.execute_validated_query

    def timed_query(*, conn: psycopg.Connection[DictRow], query: Any) -> tuple[dict[str, Any], ...]:
        nonlocal count
        count += 1
        if count == 1:
            conn.execute("SET statement_timeout = '20ms'")
            conn.execute("SELECT pg_sleep(1)")
        return original(conn=conn, query=query)

    monkeypatch.setattr(evidence_module, "execute_validated_query", timed_query)
    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    assert result.escalation_reason is None
    assert result.escalation_reason is not EscalationReason.SQL_FAILED
    assert len(result.sql_failures) == 1
    assert len(result.evidence) == 1
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 2


def test_SQL_오류_복구중_연결실패도_숨기지_않는다() -> None:
    from typing import cast

    from reply_gate.evidence import _reset_after_error

    class BrokenConnection:
        def rollback(self) -> None:
            raise psycopg.OperationalError("connection closed during recovery")

    with pytest.raises(psycopg.OperationalError):
        _reset_after_error(cast(psycopg.Connection[DictRow], BrokenConnection()))
