"""주문번호 형식 정의의 단위 테스트 — DB 불필요.

`reply_gate.order_ref` 는 주문번호 형식의 **단독 소유자**다. 접수 검증도 시딩도 이 정의만
쓴다는 계약이 여기서 검증된다.
"""

from __future__ import annotations

from datetime import date

import pytest

from reply_gate.order_ref import (
    ORDER_NO_MAX_SEQUENCE,
    ORDER_NO_MIN_SEQUENCE,
    ORDER_NO_REGEX,
    InvalidOrderNoError,
    OrderRef,
    format_order_no,
    is_valid_order_no,
    normalize_order_no,
    parse_order_no,
)

VALID = [
    "ORD-20260202-0001",
    "ORD-20260731-0034",
    "ORD-20240229-0001",  # 2024 는 윤년
    "ORD-19991231-9999",
]

INVALID = [
    ("", "빈 문자열"),
    ("ORD-20260202-0000", "순번 0 은 주문이 아니다"),
    ("ORD-20260202-001", "순번이 3자리"),
    ("ORD-20260202-00012", "순번이 5자리"),
    ("ORD-2026022-0001", "날짜가 7자리"),
    ("ORD-20261302-0001", "13월"),
    ("ORD-20260230-0001", "2월 30일"),
    ("ORD-20260229-0001", "2026 은 평년이라 2월 29일이 없다"),
    ("ord-20260202-0001", "소문자 접두어 (정규화 없이는 무효)"),
    ("ORDER-20260202-0001", "접두어가 다름"),
    ("ORD-20260202-0001 ", "뒤 공백 (정규화 없이는 무효)"),
    (" ORD-20260202-0001", "앞 공백"),
    ("ORD-20260202-0001\n", "개행 — fullmatch 라 통과하면 안 된다"),
    ("ORD-20260202-0001; DROP TABLE orders", "SQL 주입 시도"),
    ("ORD-20260202-0001 OR 1=1", "주입 시도 2"),
    ("ORD_20260202_0001", "구분자가 다름"),
    ("ORD-2026O202-0001", "숫자 자리에 알파벳 O"),
]


@pytest.mark.parametrize("value", VALID)
def test_유효한_주문번호는_허용한다(value: str) -> None:
    assert is_valid_order_no(value) is True


@pytest.mark.parametrize(("value", "why"), INVALID, ids=[why for _, why in INVALID])
def test_잘못된_주문번호는_거부한다(value: str, why: str) -> None:
    assert is_valid_order_no(value) is False, why
    with pytest.raises(InvalidOrderNoError):
        parse_order_no(value)


def test_파싱은_날짜와_순번을_반환한다() -> None:
    ref = parse_order_no("ORD-20260315-0042")
    assert ref == OrderRef(ordered_on=date(2026, 3, 15), sequence=42)


def test_포맷과_파싱이_왕복한다() -> None:
    ordered_on = date(2026, 7, 31)
    order_no = format_order_no(ordered_on=ordered_on, sequence=34)
    assert order_no == "ORD-20260731-0034"
    assert parse_order_no(order_no) == OrderRef(ordered_on=ordered_on, sequence=34)
    assert str(OrderRef(ordered_on=ordered_on, sequence=34)) == order_no


@pytest.mark.parametrize("sequence", [ORDER_NO_MIN_SEQUENCE, 500, ORDER_NO_MAX_SEQUENCE])
def test_포맷은_유효한_순번_범위를_허용한다(sequence: int) -> None:
    assert is_valid_order_no(format_order_no(ordered_on=date(2026, 3, 15), sequence=sequence))


@pytest.mark.parametrize("sequence", [-1, 0, ORDER_NO_MAX_SEQUENCE + 1, 100000])
def test_포맷은_범위밖_순번을_거부한다(sequence: int) -> None:
    with pytest.raises(InvalidOrderNoError):
        format_order_no(ordered_on=date(2026, 3, 15), sequence=sequence)


@pytest.mark.parametrize(
    "raw", ["  ORD-20260315-0042  ", "ord-20260315-0042", "\tOrd-20260315-0042\n"]
)
def test_정규화는_느슨한_입력을_유효하게_만든다(raw: str) -> None:
    """접수 폼의 공백·대소문자만 흡수한다 — 형식이 틀린 값을 고쳐주지는 않는다."""
    assert is_valid_order_no(raw) is False
    assert normalize_order_no(raw) == "ORD-20260315-0042"
    assert is_valid_order_no(normalize_order_no(raw)) is True


def test_정규화도_깨진_값을_살리지_않는다() -> None:
    assert is_valid_order_no(normalize_order_no("  ORD-20260230-0001 ")) is False
    assert is_valid_order_no(normalize_order_no("주문번호 없음")) is False


def test_정규식에는_캡처_그룹이_없다() -> None:
    """정규식 문자열은 Postgres CHECK 제약과 문자 단위로 같아야 한다.

    캡처 그룹이 생기면 두 곳이 어긋나기 시작한다 — `db/schema.sql` 과의 대조 테스트는
    `tests/test_db_schema.py` 에 있다.
    """
    assert "(" not in ORDER_NO_REGEX
    assert ORDER_NO_REGEX.startswith("^") and ORDER_NO_REGEX.endswith("$")
