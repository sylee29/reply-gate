"""SQL 숫자 타입은 PII 면제가 아니며 확정 답변의 정확한 수량만 보존한다."""

from dataclasses import replace

import pytest

from reply_gate.api import InquiryResponse
from reply_gate.contracts import InquiryStatus
from reply_gate.gate import is_quantity_echo
from tests.test_response_privacy import privacy_processed


def _response(value: object, *, answer: str | None, method: str = "phone") -> str:
    processed = privacy_processed()
    snapshot = replace(
        processed.sql_snapshots[0],
        query_sql="SELECT total_price_krw AS amount FROM orders",
        result_rows=({"amount": value},),
    )
    response = InquiryResponse.of(
        replace(
            processed,
            answer=answer,
            status=InquiryStatus.ANSWERED if answer else InquiryStatus.ESCALATED,
            verification_method=method,
            sql_snapshots=(snapshot,),
        )
    )
    assert snapshot.result_rows == ({"amount": value},)
    return response.citations[0].content


@pytest.mark.parametrize("value", [15881234, 9001011234567, 15881234.0, "15881234.00"])
@pytest.mark.parametrize(
    "answer, method",
    [
        ("배송 상태를 확인했습니다.", "phone"),
        (None, "phone"),
        ("금액은 15881234원입니다.", "product_confirm"),
    ],
)
def test_숫자타입이라도_미사용과_인계와_상품확인에서는_가린다(
    value: object, answer: str | None, method: str
) -> None:
    content = _response(value, answer=answer, method=method)
    assert str(value) not in content
    assert "[가림]" in content


@pytest.mark.parametrize("value", [15881234, 15881234.0, "15881234.00", "15,881,234.000"])
def test_확정답변에_실제로_쓴_동일한_정수_수량만_자료형과_무관하게_보존한다(value: object) -> None:
    assert "amount= " + str(value) in _response(value, answer="금액은 15881234원입니다.")


@pytest.mark.parametrize("value", ["15881234.01", 15881235, "1588-1234", "+15881234"])
def test_다른_값과_비정수와_전화표기는_같은_수량으로_추정하지_않는다(value: object) -> None:
    content = _response(value, answer="금액은 15881234원입니다.")
    assert str(value) not in content
    assert "[가림]" in content


@pytest.mark.parametrize(
    "answer",
    [
        "금액은 115881234원입니다.",
        "문의번호 15881234000",
        "15881234@example.com",
    ],
)
def test_답변의_부분숫자와_이메일은_수량_허용근거가_아니다(answer: str) -> None:
    assert "amount= 15881234.00" not in _response("15881234.00", answer=answer)


@pytest.mark.parametrize(
    "value, rendered",
    [
        (None, "(없음)"),
        (True, "True"),
        (False, "False"),
        (1234, "1234"),
        (12.5, "12.5"),
        ("12.50", "12.50"),
        ("NaN", "NaN"),
        ("Infinity", "Infinity"),
        ("-Infinity", "-Infinity"),
    ],
)
def test_정의된_PI가_아닌_보통값과_비유한_표현은_보존한다(value: object, rendered: str) -> None:
    assert "amount= " + rendered in _response(value, answer=None)


@pytest.mark.parametrize(
    "value, answer",
    [
        ("15881234.01", "15881234원"),
        ("-15881234", "15881234원"),
        ("+15881234", "15881234원"),
        ("1588-1234", "15881234원"),
        ("15881234.00", "-15881234원"),
        ("15881234.00", "문의번호 15881234"),
        ("15881234.00", "15881234@example.com"),
        ("15881234.00", "115881234원"),
    ],
)
def test_수량_예외는_완전한_같은_정수와_단위_문맥만_승인한다(value: str, answer: str) -> None:
    assert not is_quantity_echo(value, answer_text=answer)


def test_같은값의_수량이_있어도_별도_개인전화_원열은_수량예외를_받지_않는다() -> None:
    processed = privacy_processed()
    snapshot = replace(
        processed.sql_snapshots[0],
        query_sql="SELECT total_price_krw AS amount, customer_phone AS phone FROM orders",
        result_rows=({"amount": "15881234.00", "phone": "15881234"},),
    )
    response = InquiryResponse.of(
        replace(processed, answer="금액은 15881234원입니다.", sql_snapshots=(snapshot,))
    )
    content = response.citations[0].content
    assert "amount= 15881234.00" in content
    assert "phone= [가림]" in content
    assert snapshot.result_rows == ({"amount": "15881234.00", "phone": "15881234"},)
