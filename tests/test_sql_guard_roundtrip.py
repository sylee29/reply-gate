"""승인 AST와 실제 실행 문자열의 경계 — SQL 직렬화도 검증 대상이다."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from psycopg.rows import DictRow
from sqlglot import exp

from reply_gate.contracts import EscalationReason
from reply_gate.evidence import (
    INTENT_STAGE,
    SQL_GENERATION_STAGE,
    SqlFailureKind,
    execute_validated_query,
)
from reply_gate.sql_guard import SqlGuardRejection, validate_sql
from tests.test_evidence import _client, _collector, _intent, _sql

ORDER_NO = "ORD-20260201-0001"


def _base(order_no: str = ORDER_NO) -> str:
    return f"SELECT o.order_no FROM orders AS o WHERE o.order_no = '{order_no}'"


# 실제로 승인 AST의 두 리터럴 경계가 직렬화 뒤 바뀌어 주문 범위 밖 행을 읽었던 입력.
_BOUNDARY = r" AND o.status <> E'\\' AND o.status <> ' OR 1=1 --'"


@pytest.mark.parametrize(
    "sql",
    [
        _base() + _BOUNDARY,
        _base() + _BOUNDARY.replace("o.status", "o.courier"),
        _base() + _BOUNDARY.replace("<>", "="),
        _base() + _BOUNDARY.replace("<>", ">"),
        f"WITH t AS ({_base() + _BOUNDARY}) SELECT order_no FROM t",
        f"SELECT t.order_no FROM ({_base() + _BOUNDARY}) AS t",
    ],
    ids=["재현원문", "다른열", "동등비교", "대소비교", "CTE내부", "파생테이블내부"],
)
def test_직렬화가_리터럴_경계를_바꾸는_SQL은_실행_전에_거부한다(sql: str) -> None:
    """입력 AST 검사만으로는 실행 문자열의 주문 범위를 보장할 수 없었다."""
    with pytest.raises(SqlGuardRejection):
        validate_sql(sql, order_no=ORDER_NO, max_rows=50)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (ORDER_NO, "ORD-20260201-0002"),
        ("o.order_no =", "o.order_no <>"),
        ("LIMIT 50", "LIMIT 500"),
        ("o.order_no FROM", "o.status AS order_no FROM"),
        ("e'CaseSensitive'", "e'casesensitive'"),
    ],
    ids=["주문값", "비교연산자", "상한", "출력유래", "escape대소문자"],
)
def test_주석_없는_렌더러_의미_변형도_거부한다(
    monkeypatch: pytest.MonkeyPatch, before: str, after: str
) -> None:
    """주석 탐지만 추가해 놓고 의미 불변식을 지킨다고 주장하지 못하게 한다."""
    render = exp.Select.sql

    def altered_sql(self: exp.Select, *args: Any, **kwargs: Any) -> str:
        original = render(self, *args, **kwargs)
        assert before in original
        return original.replace(before, after)

    monkeypatch.setattr(exp.Select, "sql", altered_sql)
    with pytest.raises(SqlGuardRejection):
        validate_sql(_base() + " AND o.status <> E'CaseSensitive'", order_no=ORDER_NO, max_rows=50)


@pytest.mark.parametrize(
    "condition",
    [
        "o.status <> '배송중'",
        "o.status <> '고객''확인'",
        "o.status <> '주석 -- 모양 /* 문자열 */'",
        "o.status <> E'CaseSensitive'",
        r"o.status <> E'줄\n바꿈'",
        r"o.status <> E'010\0559999\0559999'",
        "o.status <> $$배송중$$",
    ],
)
def test_정상_리터럴과_승인한_출력_유래는_보존한다(condition: str) -> None:
    sql = (
        "WITH t AS (SELECT o.order_no, coalesce(o.customer_phone, '미정') AS phone "
        f"FROM orders AS o WHERE o.order_no = '{ORDER_NO}' AND {condition}) "
        "SELECT order_no, phone FROM t"
    )
    query = validate_sql(sql, order_no=ORDER_NO, max_rows=50)
    assert query.pii_safe_output_columns == ("order_no", "phone")
    assert query.order_no == ORDER_NO
    assert query.limit == 50


def test_검사한_문자열을_다시_렌더링하지_않고_반환한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """재파싱 뒤 또 직렬화한 미검증 제3 문자열을 실행하면 같은 경계가 다시 열린다."""
    render = exp.Select.sql
    rendered: list[str] = []

    def counted_sql(self: exp.Select, *args: Any, **kwargs: Any) -> str:
        sql = render(self, *args, **kwargs)
        rendered.append(sql)
        return sql

    monkeypatch.setattr(exp.Select, "sql", counted_sql)
    query = validate_sql(_base(), order_no=ORDER_NO, max_rows=50)
    assert rendered == [query.sql]


@pytest.mark.db
def test_거부된_경계_입력은_sql_failed로_끝나고_정상_재시도만_실행한다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """실제 수집·read-only 실행 경로에서 주문 범위 밖 행이 채택되지 않아야 한다."""
    row = app_conn.execute("SELECT order_no FROM orders ORDER BY order_no LIMIT 1").fetchone()
    assert row is not None
    order_no: str = row["order_no"]
    normal = _base(order_no)
    boundary = normal + _BOUNDARY

    for retry, expected_reason in [(boundary, EscalationReason.SQL_FAILED), (normal, None)]:
        client = _client(
            {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(boundary), _sql(retry)]}
        )
        result = _collector(client).collect(
            inquiry_id="11111111-2222-3333-4444-555555555555",
            content="주문 상태를 알려주세요.",
            order_no=order_no,
            app_conn=app_conn,
            readonly_conn=ro_conn,
        )
        assert result.escalation_reason is expected_reason
        assert len(client.calls_for(SQL_GENERATION_STAGE)) == 2
        assert all(f.kind is SqlFailureKind.GUARD_REJECTED for f in result.sql_failures)
        rows = [row for snapshot in result.sql_snapshots for row in snapshot.result_rows]
        assert sum(row["order_no"] != order_no for row in rows) == 0
        if expected_reason is None:
            assert len(rows) == 1
            assert len(result.sql_failures) == 1
        else:
            assert len(result.sql_failures) == 2
            assert result.sql_snapshots == ()
            assert result.evidence == ()

    query = validate_sql(normal, order_no=order_no, max_rows=50)
    assert execute_validated_query(conn=ro_conn, query=query) == ({"order_no": order_no},)
