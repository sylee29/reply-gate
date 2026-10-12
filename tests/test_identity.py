"""문의자 확인은 의도 분류보다 먼저이며 상품 확인은 PII를 허용하지 않는다."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow
from pydantic import SecretStr, ValidationError

from reply_gate.api import InquiryRequest, InquiryService, ServiceOpener, app, get_service
from reply_gate.contracts import EscalationReason, Evidence, EvidenceSource, RejectReason
from reply_gate.db import apply_schema, connect, readonly_connect
from reply_gate.draft import DRAFT_STAGE
from reply_gate.evaluation import fixture_phone_last4
from reply_gate.evidence import INTENT_STAGE, SQL_GENERATION_STAGE
from reply_gate.pipeline import (
    ConfirmationConflict,
    InquiryPipeline,
    ProcessedInquiry,
    new_inquiry_id,
)
from reply_gate.records import load_inquiry, save_inquiry
from tests.conftest import declared_settings
from tests.test_pipeline import (
    citing_draft,
    collection,
    intent_completion,
    live_pipeline,
    pipeline_with,
    scripted_client,
    sql_completion,
)

_created: list[str] = []


@pytest.fixture(autouse=True)
def _cleanup_records(request: pytest.FixtureRequest) -> Iterator[None]:
    start = len(_created)
    yield
    if request.node.get_closest_marker("db") is not None and _created[start:]:
        with connect() as conn:
            conn.execute("DELETE FROM inquiries WHERE id = ANY(%s::uuid[])", (_created[start:],))
        del _created[start:]


def _save_pending(pending: ProcessedInquiry) -> None:
    with connect() as conn:
        save_inquiry(conn=conn, processed=pending)
    _created.append(pending.inquiry_id)


@contextmanager
def service_client(
    *,
    pipeline: InquiryPipeline,
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
) -> Iterator[TestClient]:
    del app_conn, ro_conn

    @contextmanager
    def opened() -> Iterator[InquiryService]:
        with connect() as conn:
            before = {str(row["id"]) for row in conn.execute("SELECT id FROM inquiries").fetchall()}
            service = InquiryService(
                open_pipeline=lambda: pipeline, app_conn=conn, open_readonly=readonly_connect
            )
            yield service
            rows = conn.execute("SELECT id FROM inquiries").fetchall()
            _created.extend(str(row["id"]) for row in rows if str(row["id"]) not in before)

    def opener() -> ServiceOpener:
        return opened

    app.dependency_overrides[get_service] = opener
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_service, None)


@pytest.mark.parametrize(
    "value", ["123", "12345", "\uff11\uff12\uff13\uff14", "12a4", " 1234", "1234 "]
)
def test_전화_뒷자리는_ASCII_정확히_네자리(value: str) -> None:
    with pytest.raises(ValidationError):
        InquiryRequest(content="문의", phone_last4=value)


def test_입력오류는_민감한_본문을_반사하지_않는다() -> None:
    with TestClient(app) as client:
        response = client.post("/inquiries", json={"content": "", "phone_last4": "4321"})
    assert response.status_code == 422
    assert "4321" not in response.text
    assert "input" not in response.json()["detail"][0]


@pytest.mark.db
def test_전화없이_실재주문은_분류전에_상품확인을_기다린다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    row = app_conn.execute("SELECT order_no, product_name FROM orders LIMIT 1").fetchone()
    assert row is not None
    client = scripted_client({})
    result = live_pipeline(client).run(
        inquiry_id=new_inquiry_id(),
        content="정책 문의",
        order_no=row["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    assert result.status.value == "awaiting_confirmation"
    assert result.answer == f"주문하신 상품이 {row['product_name']} 맞나요?"
    assert result.intent is None
    assert result.attempts == ()
    assert result.evidence == ()
    assert result.input_tokens == result.output_tokens == 0
    assert client.calls == []


@pytest.mark.db
def test_없는주문도_분류전에_동일하게_거부한다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = scripted_client({})
    result = live_pipeline(client).run(
        inquiry_id=new_inquiry_id(),
        content="정책 문의",
        order_no="ORD-20991231-9999",
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    assert result.escalation_reason is not None
    assert result.escalation_reason.value == "order_not_found"
    assert client.calls == []


@pytest.mark.db
def test_확인대기_조회_동일ID_거절_재확인409(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    row = app_conn.execute("SELECT order_no FROM orders LIMIT 1").fetchone()
    assert row is not None
    client = scripted_client({})
    with service_client(pipeline=live_pipeline(client), app_conn=app_conn, ro_conn=ro_conn) as http:
        pending = http.post(
            "/inquiries", json={"content": "원래 문의", "order_no": row["order_no"]}
        ).json()
        assert pending["status"] == "awaiting_confirmation"
        key = pending["inquiry_id"]
        assert http.get(f"/inquiries/{key}").json() == pending
        rejected = http.post(f"/inquiries/{key}/confirm", json={"confirmed": False})
        assert rejected.status_code == 200
        assert rejected.json()["inquiry_id"] == key
        assert rejected.json()["escalation_reason"] == "order_unconfirmed"
        assert rejected.json()["answer"] is None
        assert http.post(f"/inquiries/{key}/confirm", json={"confirmed": True}).status_code == 409
        assert http.post("/inquiries/not-uuid/confirm", json={"confirmed": True}).status_code == 404
        assert (
            http.post(
                f"/inquiries/{new_inquiry_id()}/confirm", json={"confirmed": True}
            ).status_code
            == 404
        )
        assert http.post(f"/inquiries/{key}/confirm", json={"confirmed": "true"}).status_code == 422
    restored = load_inquiry(conn=app_conn, inquiry_id=key)
    assert restored is not None
    assert restored.content == "원래 문의"
    assert restored.verification_method == "product_confirm"
    assert client.calls == []


@pytest.mark.db
@pytest.mark.parametrize(
    "projection",
    [
        "customer_name AS status",
        "customer_phone",
        "lower(customer_email)",
        "shipping_address",
        "*",
        "o.*",
    ],
)
@pytest.mark.parametrize("wrapper", ["direct", "cte", "derived"])
def test_상품확인은_개인정보열_우회를_근거채택전에_거부한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    projection: str,
    wrapper: str,
) -> None:
    row = app_conn.execute("SELECT order_no FROM orders LIMIT 1").fetchone()
    assert row is not None
    order = row["order_no"]
    query = f"SELECT {projection} FROM orders o WHERE o.order_no = '{order}'"
    if wrapper == "cte":
        query = f"WITH private_source AS ({query}) SELECT * FROM private_source"
    elif wrapper == "derived":
        query = f"SELECT * FROM ({query}) AS private_source"
    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("order")],
            SQL_GENERATION_STAGE: [sql_completion(query), sql_completion(query)],
        }
    )
    pipe = live_pipeline(client)
    pending = pipe.run(
        inquiry_id=new_inquiry_id(),
        content="주문 상태",
        order_no=order,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    _save_pending(pending)
    result = pipe.confirm(
        inquiry_id=pending.inquiry_id, confirmed=True, app_conn=app_conn, readonly_conn=ro_conn
    )
    assert result.escalation_reason is EscalationReason.SQL_FAILED
    assert result.evidence == ()


@pytest.mark.db
def test_상품확인은_안전한_SQL을_이어가고_패턴PII_답변은_거부한다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    row = app_conn.execute("SELECT order_no FROM orders LIMIT 1").fetchone()
    assert row is not None
    order = row["order_no"]
    # 정책/상품 등 다른 근거에 패턴 PII가 있어도 상품 확인은 allowlist를 만들지 않는다.
    app_conn.execute("UPDATE orders SET product_name='010-1234-5678' WHERE order_no=%s", (order,))
    # RO는 별도 연결이므로 커밋되지 않은 변경에 기대지 않고 초안에 고정 PII를 준다.
    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("order")],
            SQL_GENERATION_STAGE: [
                sql_completion(f"SELECT status FROM orders WHERE order_no = '{order}'")
            ],
            DRAFT_STAGE: [
                citing_draft(text="연락처 010-1234-5678"),
                citing_draft(text="연락처 010-1234-5678"),
            ],
        }
    )
    pipe = live_pipeline(client)
    pending = pipe.run(
        inquiry_id=new_inquiry_id(),
        content="주문 상태",
        order_no=order,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    _save_pending(pending)
    result = pipe.confirm(
        inquiry_id=pending.inquiry_id, confirmed=True, app_conn=app_conn, readonly_conn=ro_conn
    )
    assert result.escalation_reason is EscalationReason.REJECTED_TWICE
    assert result.evidence
    assert all(RejectReason.PII_DETECTED in item.reject_reasons for item in result.attempts)


@pytest.mark.db
def test_전화_실패5회_잠금_만료_성공초기화와_HMAC만저장(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    row = app_conn.execute("SELECT order_no FROM orders LIMIT 1").fetchone()
    assert row is not None
    order = row["order_no"]
    correct = fixture_phone_last4(order)
    assert correct is not None
    wrong = "0000" if correct != "0000" else "9999"
    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("order")],
            SQL_GENERATION_STAGE: [sql_completion("DELETE FROM orders")] * 2,
        }
    )
    pipe = live_pipeline(client)
    for _ in range(5):
        result = pipe.run(
            inquiry_id=new_inquiry_id(),
            content="문의",
            order_no=order,
            phone_last4=wrong,
            app_conn=app_conn,
            readonly_conn=ro_conn,
        )
        assert result.escalation_reason is EscalationReason.ORDER_NOT_FOUND
    assert client.calls == []
    locked = app_conn.execute(
        "SELECT failures, locked_until > now() AS locked FROM order_verification_attempts WHERE order_no=%s",
        (order,),
    ).fetchone()
    assert locked == {"failures": 5, "locked": True}
    denied = pipe.run(
        inquiry_id=new_inquiry_id(),
        content="문의",
        order_no=order,
        phone_last4=correct,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    assert denied.escalation_reason is EscalationReason.ORDER_NOT_FOUND
    with connect() as state_conn:
        state_conn.execute(
            "UPDATE order_verification_attempts SET locked_until=now()-interval '1 second' "
            "WHERE order_no=%s",
            (order,),
        )
    passed = pipe.run(
        inquiry_id=new_inquiry_id(),
        content="문의",
        order_no=order,
        phone_last4=correct,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    assert passed.verification_method == "phone"
    assert passed.escalation_reason is EscalationReason.SQL_FAILED
    assert passed.phone_last4_hmac is not None and len(passed.phone_last4_hmac) == 64
    save_inquiry(conn=app_conn, processed=passed)
    record = app_conn.execute(
        "SELECT to_jsonb(i) AS data FROM inquiries i WHERE id=%s", (passed.inquiry_id,)
    ).fetchone()
    assert record is not None
    assert "phone_last4" not in record["data"]
    assert record["data"]["phone_last4_hmac"] == passed.phone_last4_hmac
    assert app_conn.execute(
        "SELECT failures FROM order_verification_attempts WHERE order_no=%s", (order,)
    ).fetchone() == {"failures": 0}


@pytest.mark.db
def test_확인기한은_생성시각부터이며_만료도_대기로_조회된다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    row = app_conn.execute("SELECT order_no FROM orders LIMIT 1").fetchone()
    assert row is not None
    pipe = live_pipeline(scripted_client({}))
    with service_client(pipeline=pipe, app_conn=app_conn, ro_conn=ro_conn) as http:
        result = http.post(
            "/inquiries", json={"content": "문의", "order_no": row["order_no"]}
        ).json()
        key = result["inquiry_id"]
        with connect() as state_conn:
            state_conn.execute(
                "UPDATE inquiries SET created_at=now()-interval '31 minutes' WHERE id=%s", (key,)
            )
        assert http.post(f"/inquiries/{key}/confirm", json={"confirmed": True}).status_code == 409
        assert http.get(f"/inquiries/{key}").json()["status"] == "awaiting_confirmation"


@pytest.mark.db
def test_동시_실패5개는_재연결후에도_잠금으로_남는다(seeded_order_count: int) -> None:
    del seeded_order_count
    order = "ORD-20991230-9988"
    barrier = Barrier(5)

    def fail() -> str:
        with connect() as conn, readonly_connect() as ro:
            barrier.wait(timeout=10)
            result = live_pipeline(scripted_client({})).run(
                inquiry_id=new_inquiry_id(),
                content="문의",
                order_no=order,
                phone_last4="9998",
                app_conn=conn,
                readonly_conn=ro,
            )
            assert result.escalation_reason is not None
            return result.escalation_reason.value

    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            assert list(pool.map(lambda _: fail(), range(5))) == ["order_not_found"] * 5
        with connect() as conn:
            row = conn.execute(
                "SELECT failures, locked_until > now() AS locked FROM order_verification_attempts WHERE order_no=%s",
                (order,),
            ).fetchone()
            assert row == {"failures": 5, "locked": True}
        with readonly_connect() as ro, pytest.raises(psycopg.errors.InsufficientPrivilege):
            ro.execute("SELECT * FROM order_verification_attempts")
    finally:
        with connect() as conn:
            conn.execute("DELETE FROM order_verification_attempts WHERE order_no=%s", (order,))


@pytest.mark.db
def test_동시확인은_한번만_처리하고_같은ID를_보존한다(seeded_order_count: int) -> None:
    del seeded_order_count
    with connect() as conn, readonly_connect() as ro:
        row = conn.execute("SELECT order_no FROM orders LIMIT 1").fetchone()
        assert row is not None
        pipe = live_pipeline(scripted_client({}))
        pending = pipe.run(
            inquiry_id=new_inquiry_id(),
            content="원문",
            order_no=row["order_no"],
            app_conn=conn,
            readonly_conn=ro,
        )
        save_inquiry(conn=conn, processed=pending)
    barrier = Barrier(2)

    def confirm() -> str:
        try:
            with connect() as conn, readonly_connect() as ro:
                barrier.wait(timeout=10)
                result = live_pipeline(scripted_client({})).confirm(
                    inquiry_id=pending.inquiry_id,
                    confirmed=False,
                    app_conn=conn,
                    readonly_conn=ro,
                )
                save_inquiry(conn=conn, processed=result, update_pending=True)
            return "processed"
        except ConfirmationConflict:
            return "conflict"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(lambda _: confirm(), range(2))) == ["conflict", "processed"]
        with connect() as conn:
            result = load_inquiry(conn=conn, inquiry_id=pending.inquiry_id)
            assert result is not None
            assert result.escalation_reason is EscalationReason.ORDER_UNCONFIRMED
    finally:
        with connect() as conn:
            conn.execute("DELETE FROM inquiries WHERE id=%s", (pending.inquiry_id,))


@pytest.mark.db
def test_기존제약_교체와_멱등스키마가_L1_NULL쌍을_닫는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    # 구버전의 좁은 상태/사유 제약을 복구해도 재적용이 교체해야 한다.
    app_conn.execute("ALTER TABLE inquiries DROP CONSTRAINT inquiries_status_enum")
    app_conn.execute(
        "ALTER TABLE inquiries ADD CONSTRAINT inquiries_status_enum CHECK(status IN ('answered','escalated'))"
    )
    apply_schema(app_conn)
    apply_schema(app_conn)
    row = app_conn.execute(
        "SELECT pg_get_constraintdef(oid) AS def FROM pg_constraint WHERE conname='inquiries_status_enum'"
    ).fetchone()
    assert row is not None and "awaiting_confirmation" in row["def"]
    key = new_inquiry_id()
    app_conn.execute(
        "INSERT INTO inquiries(id, content, status, answer, latency_ms) VALUES (%s, '문의', 'answered', '답변', 0)",
        (key,),
    )
    with pytest.raises(psycopg.errors.CheckViolation), app_conn.transaction():
        app_conn.execute(
            "INSERT INTO inquiry_attempts(inquiry_id, attempt_no, verdict, draft, "
            "l1_reject_reasons) VALUES (%s, 1, 'pass', '{}', '{}')",
            (key,),
        )


@pytest.mark.db
def test_상품확인은_근거에_있는_정책전화도_L1_허용목록에_올리지_않는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    evidence = Evidence(
        id="policy:support:1",
        source=EvidenceSource.POLICY,
        content="전화 1588-1234",
        evidence_text="전화 1588-1234",
    )

    class Collector:
        def collect(self, **kwargs: Any) -> Any:
            assert kwargs["exclude_personal_columns"] is True
            return collection(evidence=(evidence,))

    client = scripted_client({DRAFT_STAGE: [citing_draft(text="전화 1588-1234")] * 2})
    pipe = pipeline_with(collector=Collector(), client=client)
    row = app_conn.execute("SELECT order_no FROM orders LIMIT 1").fetchone()
    assert row is not None
    pending = pipe.run(
        inquiry_id=new_inquiry_id(),
        content="문의",
        order_no=row["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    _save_pending(pending)
    result = pipe.confirm(
        inquiry_id=pending.inquiry_id, confirmed=True, app_conn=app_conn, readonly_conn=ro_conn
    )
    assert result.escalation_reason is EscalationReason.REJECTED_TWICE
    assert all(RejectReason.PII_DETECTED in item.reject_reasons for item in result.attempts)
    assert result.evidence == (evidence,)  # L2용 근거는 그대로다.


@pytest.mark.db
def test_전화성공은_후속_분류오류에도_실패횟수를_0으로_커밋한다(seeded_order_count: int) -> None:
    del seeded_order_count
    with connect() as conn:
        row = conn.execute("SELECT order_no FROM orders ORDER BY order_no LIMIT 1").fetchone()
        assert row is not None
        order = row["order_no"]
        conn.execute(
            "INSERT INTO order_verification_attempts(order_no,failures) VALUES (%s,2) ON CONFLICT(order_no) DO UPDATE SET failures=2,locked_until=NULL",
            (order,),
        )
    try:
        with pytest.raises(RuntimeError), connect() as conn, readonly_connect() as ro:
            client = scripted_client({INTENT_STAGE: [RuntimeError("모델 대역 실패")]})
            live_pipeline(client).run(
                inquiry_id=new_inquiry_id(),
                content="문의",
                order_no=order,
                phone_last4=fixture_phone_last4(order),
                app_conn=conn,
                readonly_conn=ro,
            )
        with connect() as conn:
            assert conn.execute(
                "SELECT failures FROM order_verification_attempts WHERE order_no=%s", (order,)
            ).fetchone() == {"failures": 0}
    finally:
        with connect() as conn:
            conn.execute("DELETE FROM order_verification_attempts WHERE order_no=%s", (order,))


@pytest.mark.db
def test_확인_외부오류뒤에도_선점은_유지하고_호출자변경은_커밋하지_않는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    row = app_conn.execute("SELECT order_no FROM orders LIMIT 1").fetchone()
    assert row is not None
    pipe = live_pipeline(scripted_client({INTENT_STAGE: [RuntimeError("모델 대역 오류")]}))
    pending = pipe.run(
        inquiry_id=new_inquiry_id(),
        content="문의",
        order_no=row["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    _save_pending(pending)
    sentinel = new_inquiry_id()
    app_conn.execute(
        "INSERT INTO inquiries(id,content,status,answer,latency_ms) VALUES(%s,'미커밋','answered','답변',0)",
        (sentinel,),
    )
    with pytest.raises(RuntimeError):
        pipe.confirm(
            inquiry_id=pending.inquiry_id, confirmed=True, app_conn=app_conn, readonly_conn=ro_conn
        )
    with connect() as conn:
        state = conn.execute(
            "SELECT status, confirmation_claimed_at IS NOT NULL AS claimed FROM inquiries WHERE id=%s",
            (pending.inquiry_id,),
        ).fetchone()
        assert state == {"status": "awaiting_confirmation", "claimed": True}
        assert conn.execute("SELECT id FROM inquiries WHERE id=%s", (sentinel,)).fetchone() is None
    with pytest.raises(ConfirmationConflict):
        pipe.confirm(
            inquiry_id=pending.inquiry_id, confirmed=True, app_conn=app_conn, readonly_conn=ro_conn
        )


@pytest.mark.parametrize(
    "number",
    [
        "ORD-\uff12\uff10\uff12\uff16\uff10\uff12\uff10\uff12-0001",
        "ORD-20260202-\uff10\uff10\uff10\uff11",
    ],
)
def test_주문번호_숫자도_ASCII로_제한한다(number: str) -> None:
    with pytest.raises(ValidationError):
        InquiryRequest(content="문의", order_no=number)


def test_HMAC키는_접수와_확인에만_선검사하고_422를_가리지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = declared_settings(l2_enabled=False, phone_hmac_secret=SecretStr(""))
    monkeypatch.setattr("reply_gate.api.get_settings", lambda: settings)
    with TestClient(app) as client:
        assert client.post("/inquiries", json={"content": "문의"}).status_code == 503
        assert (
            client.post(
                f"/inquiries/{new_inquiry_id()}/confirm", json={"confirmed": True}
            ).status_code
            == 503
        )
        assert (
            client.post("/inquiries", json={"content": "", "phone_last4": "123"}).status_code == 422
        )
        assert client.get("/health").status_code == 200
