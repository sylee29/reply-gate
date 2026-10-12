"""공개 응답의 개인정보는 최종 답변만 허용하며 저장 원본은 그대로 둔다."""

from dataclasses import replace

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.api import InquiryResponse
from reply_gate.contracts import (
    Claim,
    ClaimJudgment,
    EscalationReason,
    Evidence,
    EvidenceContradiction,
    EvidenceSource,
    InquiryStatus,
    JudgeResult,
    Verdict,
)
from reply_gate.evidence import SqlEvidenceSnapshot, _sql_evidence_texts
from reply_gate.pipeline import AttemptRecord, ProcessedInquiry
from reply_gate.records import load_inquiry, save_inquiry
from tests.test_records import POLICY_ID, _processed

ID = "11111111-2222-3333-4444-555555555555"
SQL_ID = f"sql:{ID}:1"
QUERY = "SELECT customer_name AS recipient, shipping_address AS destination, customer_phone AS phone, customer_email AS email, status FROM orders WHERE order_no='ORD-20260315-0001' LIMIT 50"
ROWS = (
    {
        "recipient": "홍길동",
        "destination": "서울시 테스트로 12",
        "phone": "010-1234-5678",
        "email": "user@example.com",
        "status": "배송완료",
    },
)


def privacy_processed() -> ProcessedInquiry:
    content, evidence_text = _sql_evidence_texts(
        sql=QUERY, rows=ROWS, pii_safe_output_columns=tuple(ROWS[0])
    )
    return replace(
        _processed(ID),
        evidence=(
            Evidence(
                id=SQL_ID, source=EvidenceSource.SQL, content=content, evidence_text=evidence_text
            ),
            Evidence(
                id=POLICY_ID,
                source=EvidenceSource.POLICY,
                content="반품 문의 1588-1234 또는 help@example.com",
                evidence_text="반품 문의 1588-1234 또는 help@example.com",
            ),
        ),
        sql_snapshots=(
            SqlEvidenceSnapshot(evidence_id=SQL_ID, sequence=1, query_sql=QUERY, result_rows=ROWS),
        ),
        answer="홍길동 고객님 연락처는 +82 10 1234 5678입니다.",
        claims=(
            Claim(text="홍길동 고객님 연락처는 +82 10 1234 5678입니다.", citation_ids=(SQL_ID,)),
        ),
    )


def test_확정답변이_쓴_값만_근거에_남기고_SQL과_나머지값은_감춘다() -> None:
    processed = privacy_processed()
    original = processed.evidence
    response = InquiryResponse.of(processed)
    texts = " ".join(c.content for c in response.citations)
    assert "홍길동" in texts and "010-1234-5678" in texts
    assert "배송완료" in texts
    for forbidden in (
        "서울시 테스트로 12",
        "user@example.com",
        "1588-1234",
        "help@example.com",
        "SELECT",
        "실행 쿼리:",
    ):
        assert forbidden not in texts
    assert processed.evidence == original
    assert processed.sql_snapshots[0].result_rows == ROWS
    assert [c.id for c in response.citations] == [SQL_ID, POLICY_ID]


@pytest.mark.parametrize("status", [InquiryStatus.ESCALATED, InquiryStatus.AWAITING_CONFIRMATION])
def test_인계와_확인대기는_근거의_개인정보를_전부_가린다(status: InquiryStatus) -> None:
    processed = replace(
        privacy_processed(),
        status=status,
        answer=None if status is InquiryStatus.ESCALATED else "주문하신 상품이 샘플 맞나요?",
        claims=(),
        escalation_reason=EscalationReason.REJECTED_TWICE
        if status is InquiryStatus.ESCALATED
        else None,
    )
    texts = " ".join(c.content for c in InquiryResponse.of(processed).citations)
    for forbidden in (
        "홍길동",
        "서울시 테스트로 12",
        "010-1234-5678",
        "user@example.com",
        "1588-1234",
        "help@example.com",
        "SELECT",
    ):
        assert forbidden not in texts
    assert "배송완료" in texts


def test_상품확인은_답변과_L2설명까지_모든_공개_개인정보를_가린다() -> None:
    secret = "홍길동 010-1234-5678 user@example.com 서울시 테스트로 12"
    attempt = AttemptRecord(
        attempt_no=1,
        verdict=Verdict.PASS,
        reject_reasons=(),
        draft={"claims": []},
        l2_result=JudgeResult(
            verdict=Verdict.PASS,
            claim_judgments=(
                ClaimJudgment(claim_text=secret, verdict=Verdict.PASS, explanation=secret),
            ),
            contradictions=(
                EvidenceContradiction(
                    evidence_id_a=SQL_ID, evidence_id_b=POLICY_ID, explanation=secret
                ),
            ),
        ),
    )
    processed = replace(
        privacy_processed(), verification_method="product_confirm", attempts=(attempt,)
    )
    serialized = InquiryResponse.of(processed).model_dump_json()
    for forbidden in ("홍길동", "서울시 테스트로 12", "010-1234-5678", "user@example.com"):
        assert forbidden not in serialized
    assert processed.attempts[0].l2_result == attempt.l2_result


def test_깨진_저장_SQL은_근거전체를_가리고_예외문자열을_반사하지_않는다() -> None:
    processed = privacy_processed()
    broken = replace(processed.sql_snapshots[0], query_sql="BROKEN 홍길동")
    response = InquiryResponse.of(replace(processed, sql_snapshots=(broken,)))
    assert "홍길동" not in response.citations[0].content
    assert "BROKEN" not in response.citations[0].content


@pytest.mark.db
@pytest.mark.parametrize("alias_collision", [False, True])
def test_jsonb가_열순서를_바꿔도_POST와_GET의_가림_응답은_같다(
    app_conn: psycopg.Connection[DictRow],
    alias_collision: bool,
) -> None:
    processed = privacy_processed()
    if alias_collision:
        snapshot = replace(
            processed.sql_snapshots[0],
            query_sql='SELECT status AS "a@example.com", product_name AS "b@example.com" FROM orders',
            result_rows=({"a@example.com": "배송완료", "b@example.com": "등산가방"},),
        )
        processed = replace(processed, sql_snapshots=(snapshot,))
    before = InquiryResponse.of(processed)
    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)
    assert loaded is not None
    assert InquiryResponse.of(loaded) == before
    assert loaded.evidence == processed.evidence
    assert loaded.sql_snapshots == processed.sql_snapshots


@pytest.mark.parametrize(
    "email_value",
    [
        "user@example.com",
        "b=user@example.com",
        "'user@example.com",
        '"a user"@example.com',
        "user@example.com입니다",
    ],
)
def test_SQL_이메일_값의_literal_문맥을_응답_가림에서도_보존한다(email_value: str) -> None:
    processed = privacy_processed()
    snapshot = replace(
        processed.sql_snapshots[0],
        result_rows=({"email": email_value},),
        query_sql="SELECT customer_email AS email FROM orders",
    )
    response = InquiryResponse.of(
        replace(processed, answer="<" + email_value + ">", sql_snapshots=(snapshot,))
    )
    assert "email= " + email_value in response.citations[0].content


def test_서로다른_개인정보_별칭을_가려도_정상_출력값을_덮어쓰지_않는다() -> None:
    processed = privacy_processed()
    snapshot = replace(
        processed.sql_snapshots[0],
        query_sql='SELECT status AS "a@example.com", product_name AS "b@example.com" FROM orders',
        result_rows=({"a@example.com": "배송완료", "b@example.com": "등산가방"},),
    )
    response = InquiryResponse.of(replace(processed, sql_snapshots=(snapshot,)))
    content = response.citations[0].content
    assert "a@example.com" not in content and "b@example.com" not in content
    assert "배송완료" in content and "등산가방" in content
    assert snapshot.result_rows == ({"a@example.com": "배송완료", "b@example.com": "등산가방"},)
