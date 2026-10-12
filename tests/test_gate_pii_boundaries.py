"""재감사에서 통과한 실제 표기를 고정한다. 새 사례는 고정 측정 픽스처와 분리한다."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reply_gate import gate
from reply_gate.contracts import Evidence, EvidenceSource, GateResult, RejectReason, Verdict
from reply_gate.evidence import _sql_evidence_texts
from reply_gate.sql_guard import validate_sql


def _judge(text: str, evidence: str = "일반 안내입니다.") -> GateResult:
    return gate.evaluate_draft(
        raw_draft={"claims": [{"text": text, "citation_ids": ["policy:audit:1"]}]},
        evidences=[
            Evidence(
                id="policy:audit:1",
                source=EvidenceSource.POLICY,
                content=evidence,
                evidence_text=evidence,
            )
        ],
    )


@pytest.mark.parametrize("number", ["010-1234-5678", "02-1234-5678", "1588-1234", "900101-1234567"])
@pytest.mark.parametrize("separator", ["  ", "--", "__", "..", "( )", "\u00b7", "\u318d", "~"])
def test_새_구분자는_무근거_번호를_막고_양방향_에코를_살린다(number: str, separator: str) -> None:
    """대표번호와 주민번호의 반복 구분자 및 네 번호 계열의 미열거 표기가 샜다."""
    variant = number.replace("-", separator)
    assert RejectReason.PII_DETECTED in _judge(variant).reject_reasons
    assert gate.pii_shaped(variant)
    assert _judge(variant, number).verdict is Verdict.PASS
    assert _judge(number, variant).verdict is Verdict.PASS


@pytest.mark.parametrize("zero", [0x660, 0x6F0, 0x966, 0x9E6, 0xE50])
@pytest.mark.parametrize("number", ["010-1234-5678", "02-1234-5678", "1588-1234", "900101-1234567"])
def test_Unicode_십진수_번호를_정규화하되_긴_숫자의_부분은_승인하지_않는다(
    zero: int, number: str
) -> None:
    variant = "".join(chr(zero + int(ch)) if ch.isdecimal() else ch for ch in number)
    assert RejectReason.PII_DETECTED in _judge(variant).reject_reasons
    assert _judge(number, variant).verdict is Verdict.PASS
    assert _judge(variant, number).verdict is Verdict.PASS
    assert not gate.pii_shaped("99999" + variant.replace("-", "") + "99999")
    assert (
        RejectReason.PII_DETECTED
        in _judge(number, "99999" + variant.replace("-", "") + "99999").reject_reasons
    )


@pytest.mark.parametrize(
    "mark", ["\u115f", "\u1160", "\u3164", "\uffa0", "\u2800", "\b", "\x1b", "\u202e", "\u2066"]
)
def test_번호_내부_비가시문자와_양방향제어를_양쪽에서_같이_접는다(mark: str) -> None:
    variant = f"010-1{mark}234-5678"
    assert RejectReason.PII_DETECTED in _judge(variant).reject_reasons
    assert _judge(variant, "010-1234-5678").verdict is Verdict.PASS
    assert _judge("010-1234-5678", variant).verdict is Verdict.PASS


@pytest.mark.parametrize("separator", ["\n", "\r", "\t"])
def test_별도_줄과_탭의_숫자는_무조건_연결하지_않는다(separator: str) -> None:
    assert not gate.pii_shaped(f"010{separator}1234{separator}5678")


@pytest.mark.parametrize(
    ("variant", "canonical"),
    [
        ("+82 (0)10-1234-5678", "010-1234-5678"),
        ("0082 (0)10-1234-5678", "010-1234-5678"),
        ("+82 11-123-4567", "011-123-4567"),
        ("+82 16-123-4567", "016-123-4567"),
        ("+82 (0)2-1234-5678", "02-1234-5678"),
        ("+1-202-555-0123", "001 202 555 0123"),
        ("+44 20 7946 0958", "0044-20-7946-0958"),
        ("+81 90 1234 5678", "0081-90-1234-5678"),
        ("+1 1588 1234", "+115881234"),
        ("+82 1588 1234", "+8215881234"),
    ],
)
def test_명시_국제형은_무근거를_거부하고_동일번호_에코를_허용한다(
    variant: str, canonical: str
) -> None:
    assert RejectReason.PII_DETECTED in _judge(variant).reject_reasons
    assert RejectReason.PII_DETECTED in _judge(canonical).reject_reasons
    assert _judge(variant, canonical).verdict is Verdict.PASS
    assert _judge(canonical, variant).verdict is Verdict.PASS


def test_부호_없는_숫자를_국가번호로_추정하지_않는다() -> None:
    assert not gate.pii_shaped("82 10 1234 5678")
    assert not gate.pii_shaped("1 202 555 0123")


@pytest.mark.parametrize(
    "address",
    [
        "a'user@example.com",
        "a/user@example.com",
        "a!user@example.com",
        "a#user@example.com",
        "a$user@example.com",
        "a&user@example.com",
        "<a=user@example.com>",
        "a?user@example.com",
        "a^user@example.com",
        "a`user@example.com",
        "a{user}@example.com",
        "a|user@example.com",
        "a~user@example.com",
        "홍user@example.com",
        "홍길동@예시.한국",
        "user@example.com\u034f.evil",
        "user@exam\u034fple.com",
        "user@example.हिन्दी",
        "user@example.com\u0301.evil",
        '"a user"@example.com',
    ],
)
def test_이메일_전체토큰을_보존하여_다른주소의_접미사_승인을_막는다(address: str) -> None:
    """ASCII 접미사와 결합표식 앞까지만 뽑으면 완전 일치 대조도 다른 주소를 승인한다."""
    assert gate.pii_shaped(address)
    assert RejectReason.PII_DETECTED in _judge(address, "user@example.com").reject_reasons
    assert _judge(address, address).verdict is Verdict.PASS


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("user@example.हिन्दी", "user@example.हन्दी"),
        ("a\u034fuser@example.com", "auser@example.com"),
        ("user@example.com\u034f.evil", "user@example.com.evil"),
    ],
)
def test_이메일의_결합표식은_서로다른_주소를_합치지_않는다(left: str, right: str) -> None:
    assert RejectReason.PII_DETECTED in _judge(left, right).reject_reasons
    assert RejectReason.PII_DETECTED in _judge(right, left).reject_reasons


@pytest.mark.parametrize("field", ["customer_email", "email", "contact", "연락처"])
def test_이메일_필드명과_동일한_localpart도_명시된_주소에서는_자르지_않는다(field: str) -> None:
    address = f"<{field}=user@example.com>"
    assert RejectReason.PII_DETECTED in _judge(address, "user@example.com").reject_reasons
    assert _judge(address, address).verdict is Verdict.PASS
    assert _judge("user@example.com", f"{field}=user@example.com").verdict is Verdict.PASS


@pytest.mark.parametrize("alias", ["contact", "연락처", "user_address"])
@pytest.mark.parametrize("address", ["user@example.com", "a=user@example.com"])
def test_실제_SQL_근거의_임의_별칭을_분리하되_이메일값은_한번만_분리한다(
    alias: str, address: str
) -> None:
    """직접 컬럼의 별칭은 렌더가 key=value로 만든다. local part의 다음 '='는 값이다."""
    order_no = "ORD-20260201-0001"
    sql = f"SELECT customer_email AS \"{alias}\" FROM orders WHERE order_no = '{order_no}'"
    query = validate_sql(sql, order_no=order_no, max_rows=50)
    content, evidence_text = _sql_evidence_texts(
        sql=sql, rows=({alias: address},), pii_safe_output_columns=query.pii_safe_output_columns
    )
    result = gate.evaluate_draft(
        raw_draft={"claims": [{"text": f"<{address}>", "citation_ids": ["sql:audit:1"]}]},
        evidences=[
            Evidence(
                id="sql:audit:1",
                source=EvidenceSource.SQL,
                content=content,
                evidence_text=evidence_text,
            )
        ],
    )
    assert result.verdict is Verdict.PASS
    assert address in evidence_text
    if "=" in address:
        assert RejectReason.PII_DETECTED in _judge("user@example.com", evidence_text).reject_reasons


@pytest.mark.parametrize(
    "address", ["'user@example.com", "`user@example.com", "{user}@example.com"]
)
def test_명시_이메일의_첫_특수문자도_localpart에서_잘리지_않는다(address: str) -> None:
    literal = f"<{address}>"
    assert RejectReason.PII_DETECTED in _judge(literal, "user@example.com").reject_reasons
    assert RejectReason.PII_DETECTED in _judge(literal, address[1:]).reject_reasons
    assert _judge(literal, literal).verdict is Verdict.PASS


@pytest.mark.parametrize(("left", "right"), [("'", "'"), ("`", "`"), ("{", "}")])
def test_주소_전체를_감싼_구분자는_localpart의_첫글자가_아니다(left: str, right: str) -> None:
    assert _judge(f"{left}user@example.com{right}", "user@example.com").verdict is Verdict.PASS


@pytest.mark.parametrize("suffix", ["\u3002", "\u061b", "😊", "♥"])
def test_Unicode_문장부호와_이모지는_도메인_꼬리가_아니다(suffix: str) -> None:
    """M 표식을 보존하려고 모든 비ASCII 문자를 도메인에 넣으면 정상 산문이 기각된다."""
    assert _judge("user@example.com" + suffix, "user@example.com").verdict is Verdict.PASS
    assert (
        RejectReason.PII_DETECTED in _judge("😊user@example.com", "user@example.com").reject_reasons
    )


@pytest.mark.parametrize("unit", ["원", "개", "건", "명"])
@pytest.mark.parametrize("amount", ["15,881,234", "16,001,234"])
def test_같은_금액과_수량의_쉼표_제거를_새_대표번호로_기각하지_않는다(
    unit: str, amount: str
) -> None:
    canonical = amount.replace(",", "") + unit
    assert _judge(canonical, amount + unit).verdict is Verdict.PASS
    assert _judge(amount + unit, canonical).verdict is Verdict.PASS
    assert RejectReason.PII_DETECTED in _judge(canonical, "다른 금액 123원").reject_reasons


def test_금액_보정이_임의_쉼표나_전화번호를_지우지_않는다() -> None:
    assert RejectReason.PII_DETECTED in _judge("연락처 1588-1234", "금액 123원").reject_reasons
    assert RejectReason.PII_DETECTED in _judge("15881234원", "15,88,1234원").reject_reasons
    assert _judge("일자 2026-10-12, 주문 ORD-20261012-0001, 수량 12개").verdict is Verdict.PASS


@pytest.mark.parametrize(
    "blank",
    [
        "\u200b",
        "\u2060\u202e",
        "\u115f",
        "\u1160",
        "\u3164",
        "\uffa0",
        "\u2800",
        "\x1b",
        "\ufe0f",
        " \u200b\u3164\n",
    ],
)
def test_가시내용_없는_claim은_스키마위반이다(blank: str) -> None:
    assert RejectReason.SCHEMA_VIOLATION in _judge(blank).reject_reasons
    with pytest.raises(ValueError):
        gate.to_draft({"claims": [{"text": blank, "citation_ids": ["policy:audit:1"]}]})


@pytest.mark.parametrize(
    "text", ["안내", "e\u0301", "확인\u200b", "👍", "👨\u200d👩\u200d👧", "!", "한"]
)
def test_정상_문자와_결합문자와_이모지는_빈_claim이_아니다(text: str) -> None:
    assert _judge(text).verdict is Verdict.PASS


def test_골뱅이_없는_긴_입력은_이메일_정규식을_반복탐색하지_않는다() -> None:
    """시간 SLA 대신 병목이었던 regex 진입을 직접 센다. 전각 @ 대조도 유지한다."""
    original = gate._EMAIL_REGEX
    with patch.object(gate, "_EMAIL_REGEX", wraps=original) as regex:
        assert _judge("a" * 32000).verdict is Verdict.PASS
        regex.sub.assert_not_called()
    assert RejectReason.PII_DETECTED in _judge("user\uff20example.com").reject_reasons
