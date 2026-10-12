"""HTTP 가림은 게이트의 패턴과 정규화만 쓰고 원문 이외의 값을 허용하지 않는다."""

import pytest

from reply_gate import gate


@pytest.mark.parametrize(
    "text, allowed, expected",
    [
        ("문의 010-1234-5678 끝", "", "문의 [가림] 끝"),
        (
            "문의 \uff10\uff11\uff10·\uff11\uff12\uff13\uff14·\uff15\uff16\uff17\uff18 끝",
            "",
            "문의 [가림] 끝",
        ),
        ("문의 0\u200b10-1234-5678 끝", "", "문의 [가림] 끝"),
        ("문의 010-1234-5678 끝", "+82 10 1234 5678", "문의 010-1234-5678 끝"),
        ("문의 010-1234-5678 끝", "9010123456780", "문의 [가림] 끝"),
        ("a_b@example.com, a-b@example.com", "a_b@example.com", "a_b@example.com, [가림]"),
        ("고객 이메일은 user@example.com입니다.", "", "고객 이메일은 [가림]입니다."),
        ("01012345678@example.com", "01012345678@example.com", "01012345678@example.com"),
        ("010 user@example.com 1234 5678", "user@example.com", "010 user@example.com 1234 5678"),
        ("가격은 15,881,234원입니다.", "가격은 15881234원입니다.", "가격은 15,881,234원입니다."),
    ],
)
def test_원문_문맥과_정규화한_정확한_허용값을_보존한다(
    text: str, allowed: str, expected: str
) -> None:
    assert gate.redact_pii(text, allowed_text=allowed) == expected


@pytest.mark.parametrize(
    "allowed, expected",
    [("", "받는 분 [가림], [가림]"), ("홍길동 고객님", "받는 분 홍길동, [가림]")],
)
def test_주문_이름과_주소는_값을_알때만_가린다(allowed: str, expected: str) -> None:
    assert (
        gate.redact_pii(
            "받는 분 홍길동, 서울시 테스트로 12",
            allowed_text=allowed,
            private_values=("홍길동", "서울시 테스트로 12"),
        )
        == expected
    )


def test_긴_무매치_문장은_위치대조를_전혀_실행하지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object) -> None:
        raise AssertionError("PII가 없으면 원문 위치 대조가 필요 없다")

    monkeypatch.setattr(gate, "_source_spans", unexpected)
    text = "가나다 abc 배송 확인 " * 10000
    assert gate.redact_pii(text) == text


def test_반복하는_긴_정상문장_뒤의_번호만_가린다() -> None:
    prefix = "반복 문장 " * 5000
    assert gate.redact_pii(prefix + "010-1234-5678 끝") == prefix + "[가림] 끝"
