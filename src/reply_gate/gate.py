"""L1 게이트 — 근거 없는 답변을 코드로 기각한다.

docs/business-rules.md "L1 게이트 판정 규칙" 을 그대로 옮긴 모듈이다. 제품 정체성이
실제로 구현되는 곳이므로 아래 두 성질은 어떤 편의로도 깨지 않는다.

* **LLM 호출 0회.** 이 모듈은 LLM·네트워크 라이브러리를 import 하지 않는다.
* **100% 결정론.** 시간·난수·환경에 의존하지 않는다 — 같은 입력은 항상 같은 판정과
  같은 사유 순서를 낸다("측정 1 — L1 게이트 단위 정확도(결정론)").

기각 사유는 **하나만 잡고 멈추지 않고 전부 수집한다.** 스키마가 깨져서 하위 검사가
불가능한 부분만 건너뛴다 — 예를 들어 claim 이 dict 가 아니면 그 claim 의 citation 검사는
할 수 없지만, 나머지 claim 의 citation 검사와 초안 전체의 PII 검사는 그대로 수행한다.

구조 검사를 `DRAFT_JSON_SCHEMA` 에 위임하지 않는다: 그건 생성 측 구조화 출력용이고,
생성 측 강제가 L1 검사를 대체하면 게이트가 실제로 무엇을 막는지 증명할 수 없다.

**패턴형 개인정보 정의의 단독 소유자가 이 모듈이다.** 정규식·접기·정규화를 한 벌
(`DEFAULT_PII_PATTERNS`)로 들고, 근거 렌더의 개인정보 필터와 조회 가드가 `pii_shaped`
로 가져다 쓴다 — 층마다 자기 정의를 두면 한쪽만 넓혀져 기준이 갈리고, 실제로 그렇게
뚫렸다(docs/engineering-notes.md "접기 기준이 층마다 달라 근거 필터가 게이트보다 헐거웠다").
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from reply_gate.contracts import Claim, Draft, Evidence, GateResult, RejectReason, Verdict

__all__ = [
    "DEFAULT_PII_PATTERNS",
    "NUMERIC_SEPARATOR_VARIANTS",
    "REASON_ORDER",
    "PiiPattern",
    "evaluate_draft",
    "fold_for_detection",
    "fold_numeric_for_detection",
    "normalize_digits",
    "normalize_email",
    "normalize_phone",
    "pii_shaped",
    "to_draft",
]

_CLAIMS_KEY = "claims"
_TEXT_KEY = "text"
_CITATION_IDS_KEY = "citation_ids"

#: 사유 목록의 고정 순서. 검출 순서에 의존하면 같은 입력이 다른 순서를 내 결정론이 깨진다.
#: 리포트의 사유별 내역도 이 순서를 그대로 쓴다 — 정의는 여기 한 곳뿐이다.
REASON_ORDER: tuple[RejectReason, ...] = (
    RejectReason.SCHEMA_VIOLATION,
    RejectReason.MISSING_CITATION,
    RejectReason.INVALID_CITATION,
    RejectReason.PII_DETECTED,
)

#: 원시 초안에서 텍스트를 긁어모을 때의 최대 중첩 깊이. 자기 참조 구조에서도 멈추게 한다.
_MAX_TEXT_DEPTH = 8

_NON_DIGIT = re.compile(r"[^0-9]")


# ── PII 접기 — 탐지 전 유니코드 정규화 ──────────────────────────────────────

#: 번호 계열이 **구분자로 받는** 표기 변종. **열거된 고정 집합**이고 범주 이름으로
#: 유도하지 않는다 — 대시뿐 아니라 중점·빗금·밑줄도 구분자로 쓰인다:
#: 대시 범주(`Pd`)에 드는 것은 U+2010~U+2015 여섯 자뿐이고, U+2212 는 수학기호(`Sm`),
#: U+2027·U+30FB·U+FF65·`/` 는 구두점(`Po`), `_` 는 연결선(`Pc`) 이다.
#: `category == "Pd"` 로 짜면 **열거된 절반이 그대로 뚫린 채** 문서는 닫혔다고 적게 된다.
#:
#: 눈으로 구분되지 않는 문자는 **이스케이프로 적는다** — 리터럴로 쓰면 무엇이 들어 있는지
#: 읽을 수 없고, 편집 중에 조용히 ASCII 하이픈으로 바뀌어도 아무도 모른다.
NUMERIC_SEPARATOR_VARIANTS: frozenset[str] = frozenset(
    (
        "\u2010",  # HYPHEN
        "\u2011",  # NON-BREAKING HYPHEN
        "\u2012",  # FIGURE DASH
        "\u2013",  # EN DASH
        "\u2014",  # EM DASH
        "\u2015",  # HORIZONTAL BAR
        "\u2212",  # MINUS SIGN
        "\u2027",  # HYPHENATION POINT
        "\u30fb",  # KATAKANA MIDDLE DOT
        "\uff65",  # HALFWIDTH KATAKANA MIDDLE DOT
        "/",  # SOLIDUS
        "_",  # LOW LINE
        "\u00b7",  # MIDDLE DOT
        "\u318d",  # HANGUL LETTER ARAEA (NFKC 뒤에는 U+119E)
        "\u119e",  # HANGUL JUNGSEONG ARAEA
        "~",  # TILDE
    )
)

#: 번호 계열이 **무시하는** 결합 표식의 유니코드 범주 **접두사**. 변이선택자도 전부 든다.
#: ①의 12자와 반대로 **범주로 구현한다** — 대표 문자 몇 개로 한정하면 U+0300 같은 동일
#: 계열이 그대로 남기 때문이다.
#:
#: ⚠ **`Mn` 하나가 아니라 `M` 계열 전부다.** 유니메코드 결합 표식은 `Mn`(nonspacing) ·
#: `Mc`(spacing combining) · `Me`(enclosing) 셋으로 갈린다. `Mn` 만 보면 `U+0903`(Mc)·
#: `U+20E3`(Me)를 숫자 자리 사이에 끼운 번호가 그대로 통과한다 — 실제로 찍어 확인했고,
#: 그것은 ①에서 경고한 "범주가 계열을 덮지 못한다"를 반대쪽에서 여는 것이다.
_COMBINING_MARK_CATEGORY_PREFIX = "M"

#: 폭 없는 서식 문자(U+200B·U+2060 등)의 범주. 공통 접기가 통째로 지운다.
_FORMAT_CATEGORY = "Cf"

# NFKC 이후에도 공백으로 분류되지 않는 빈칸. 문장/행 구분자는 지우지 않는다.
_BLANK_CHARACTERS = frozenset(("\u115f", "\u1160", "\u2800", "\u3164", "\uffa0"))
_QUANTITY_UNIT = r"(?:원|개|건|명)"
_QUANTITY_END = r"(?:입니다|이에요|예요|으로|은|는|을|를|이|가|에|로)?(?=$|\s|[.,!?;:)\]}>])"
_INTEGER_DIGITS = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_GROUPED_QUANTITY = re.compile(
    rf"(?<![0-9,])[0-9]{{1,3}}(?:,[0-9]{{3}})+(?= ?{_QUANTITY_UNIT}{_QUANTITY_END})"
)
_QUANTITY = re.compile(
    rf"(?<![\w+\-.,@/])(?P<sign>\+?)(?P<number>{_INTEGER_DIGITS})(?:\.0+)? ?"
    rf"(?P<unit>{_QUANTITY_UNIT}){_QUANTITY_END}"
)
_INTEGER_VALUE = re.compile(
    rf"(?<![\w+\-.,@/])\+?(?P<number>{_INTEGER_DIGITS})(?:\.0+)?(?![\w.,@/\-])"
)
_INTERNATIONAL_PHONE = re.compile(
    r"(?<![0-9])(?:\+|00)[1-9](?:[-. ()]*[0-9]){6,14}(?![-. ()]*[0-9])"
)


def fold_for_detection(text: str) -> str:
    """탐지 전 **공통** 접기 — 사람 눈에 같은 값이 패턴을 비껴가지 못하게 한다.

    전각 숫자(U+FF10~U+FF19)로 쓴 전화번호는 사람이 읽으면 전화번호지만 ASCII 숫자
    정규식에 걸리지 않는다. NFKC 로 호환 문자를 반각으로 접고, 폭 없는 서식 문자
    (`Cf` 범주 — U+200B 등)를 지운다.

    **이메일도 이 공통 접기를 쓰되 한국어 산문 경계만 별도로 분리한다.** 밑줄과 빗금을
    하이픈으로 접으면 `a_b@…` 와 `a-b@…` 가 **같은 값이 되어**, 근거에 있는 주소를 한
    글자 바꿔 지어낸 주소가 근거 유래로 통과한다 — 우회 하나를 닫으면서 반대 방향 우회를
    여는 것이다. 밑줄은 이 저장소의 사유 코드·근거 ID·컬럼명에 흔해서, 전역으로 접으면
    번호와 무관한 문자열까지 함께 움직인다.

    **초안과 근거 양쪽에 같은 접기를 적용한다.** 한쪽에만 걸면 근거의 같은 값이 다른
    문자열이 되어 정상 에코가 오기각되고, 반대로 근거 쪽만 헐거우면 접어야 PII 가 되는
    값이 allowlist 근거로 살아남는다.
    """
    folded = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in folded if unicodedata.category(ch) != _FORMAT_CATEGORY)


def _fold_numeric_characters(text: str) -> str:
    """번호 문자만 접는다. 수의 소수점/국제전화 구분자 해석은 아직 하지 않는다."""
    return "".join(
        "-"
        if ch in NUMERIC_SEPARATOR_VARIANTS
        else str(unicodedata.decimal(ch))
        if ch.isdecimal()
        else ch
        for ch in fold_for_detection(text)
        if not unicodedata.category(ch).startswith(_COMBINING_MARK_CATEGORY_PREFIX)
        and ch not in _BLANK_CHARACTERS
        and not (unicodedata.category(ch) == "Cc" and not ch.isspace())
    )


def fold_numeric_for_detection(text: str) -> str:
    """번호 계열 접기 — 공통 접기 위에 구분자 변종과 결합 표식을 얹는다.

    NFKC 는 대시류를 ASCII 하이픈으로 접지 않고, 결합 표식은 `Cf` 가 아니라 지워지지도
    않는다. 그래서 en dash 로 쓴 `010-9999-8888` 도, 숫자 **자리 사이**에 U+0301 을 끼운
    `010-9999-8888` 도 구분자 클래스 `[-. ()]` 를 그대로 비껴갔다 — 사람 눈에는 둘 다
    같은 번호로 읽힌다. 실제 우회 문자열은 `tests/test_pii_folding.py` 가 들고 있다.

    **배치가 판정을 가른다** — 결합 표식은 숫자열 안쪽에 끼워야 뚫리고 끝에 붙이면 매치가
    이미 끝난 뒤라 원래도 기각된다. 회귀 검사가 자리 사이 배치를 박는 이유다.

    접기는 **탐지에만** 쓴다. 근거 문면을 갈아 끼우지 않는다 — 바꾸면 정상 에코 계약과
    L2 근거가 함께 흔들린다.
    """
    numeric = _fold_numeric_characters(text)
    # 단위가 붙은 올바른 천 단위 묶음만 접는다. 쉼표를 임의 구분자로 받지 않는다.
    numeric = _GROUPED_QUANTITY.sub(lambda match: match.group().replace(",", ""), numeric)
    # 국제번호 안의 국내형 조각을 별도 번호로 뽑으면 +1 1588 1234와 +115881234의
    # allowlist가 달라진다. 명시적 국제 후보 안에서만 구분자를 지워 경계를 유지한다.
    return _INTERNATIONAL_PHONE.sub(
        lambda match: (
            ("+" if match.group().startswith("+") else "") + normalize_digits(match.group())
        ),
        numeric,
    )


# Unicode local part와 결합표식이 포함된 도메인을 끝까지 보존한다. 이메일에서
# 결합표식을 지우면 다른 주소끼리 같아지므로 번호 접기와 공유하지 않는다.
_EMAIL_UNICODE = r"[^\x00-\x7f\s\u2018\u2019\u201c\u201d]"
_EMAIL_LOCAL = rf"(?:[\w!#$%&'*+/=?^`{{|}}~.-]|{_EMAIL_UNICODE})"
# stdlib re의 \w에는 M 범주가 없다. 대표 표식 몇 개만 넣지 않고 모듈 로드 때
# 범주 전체를 문자 클래스로 한 번 만든다. 문장부호/이모지는 도메인에 붙이지 않는다.
_EMAIL_MARKS = "".join(
    chr(codepoint)
    for codepoint in range(0x110000)
    if unicodedata.category(chr(codepoint)).startswith("M")
)
_EMAIL_LABEL = rf"(?:[^\W_]|-|[{re.escape(_EMAIL_MARKS)}])"
_EMAIL_REGEX = re.compile(
    rf"(?<![\w!#$%&*+/?^|~.-])"
    rf'(?:"(?:[^"\\\r\n]|\\[^\r\n])*"|{_EMAIL_LOCAL}+)'
    rf"@{_EMAIL_LABEL}+(?:\.{_EMAIL_LABEL}+)+"
)
_EMAIL_VALUE_PREFIX = re.compile(
    r"""(?<!\S)[^\s=<>()[\]{}"'`\u2018\u2019\u201c\u201d][^\s=<>]* *=(?=\S)"""
)
_EMAIL_WRAPPED_VALUE = re.compile(r"(?<!\S)(['`{])([^\s<>]+@[^\s<>]+)(['`}])")
_EMAIL_PROSE_LAST_LABEL = re.compile(
    r"([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)"
    r"(입니다|으로|에서|로|은|는|을|를|이|가|에|와|과)"
)


def _email_literal_is_open(prefix: str) -> bool:
    """주소 앞의 열린 구분자를 등장 순서로 추적한다."""
    closing = {"<": ">", "(": ")", "[": "]", "{": "}", "\u201c": "\u201d", "\u2018": "\u2019"}
    quote_closers = {'"', "'", "`", "\u201d", "\u2019"}
    opened: list[str] = []
    for char in prefix:
        if opened and char == opened[-1]:
            opened.pop()
        elif opened and opened[-1] in quote_closers:
            continue
        elif char in closing:
            opened.append(closing[char])
        elif char in ('"', "'", "`"):
            opened.append(char)
    return bool(opened)


def _fold_email_for_detection(text: str) -> str:
    """ASCII 최종 라벨 뒤 고정 한국어 접미사만 산문 경계로 분리한다.

    ponytail: 산문에서 `com입니다`는 주소와 서술어로 읽는다. 실제 혼합 IDN 주소라면
    따옴표·백틱·괄호·key=value로 명시해야 한다. 임의 IDN/한국어의 일반 해독은 하지 않는다.
    명시된 주소와 닫히지 않은 구분자는 줄이지 않으며 근거·초안 모두 같은 경계를 쓴다.
    """
    folded = fold_for_detection(text)
    if "@" not in folded:
        # 후보가 없으면 정규식의 모든 시작점에서 local part를 재탐색할 이유가 없다.
        return ""

    # 근거 렌더의 임의 key=value에서 첫 '='만 필드 경계다. 값 안의 '='는 보존한다.
    # 구분 없는 a=user@…는 필드 문법이고 <a=user@…>는 local part에 '='가 있는 주소다.
    folded = _EMAIL_VALUE_PREFIX.sub(
        lambda match: (
            match.group()
            + (
                ""
                if folded[: match.start()].rstrip().endswith("=")
                or _email_literal_is_open(folded[: match.start()])
                else " "
            )
        ),
        folded,
    )

    # 주소 전체의 구분자만 분리한다. <'user@…>나 a'user@…의 apostrophe는 값이다.
    folded = _EMAIL_WRAPPED_VALUE.sub(
        lambda match: (
            match[1] + " " + match[2] + match[3]
            if match[3] == {"'": "'", "`": "`", "{": "}"}[match[1]]
            else match.group()
        ),
        folded,
    )

    def separate_prose(match: re.Match[str]) -> str:
        address = match.group()
        last_label = _EMAIL_PROSE_LAST_LABEL.fullmatch(address.rsplit(".", 1)[-1])
        if last_label is None:
            return address
        tail = folded[match.end() :]
        if tail and not (tail[0].isspace() or tail[0] in ".,;!?"):
            return address
        if tail.startswith(".") and len(tail) > 1 and not tail[1].isspace():
            return address
        prefix = folded[: match.start()].rstrip()
        if prefix.endswith(("=", ":")):
            return address
        if _email_literal_is_open(prefix):
            return address
        suffix = last_label[2]
        return address[: -len(suffix)] + " " + suffix

    return _EMAIL_REGEX.sub(separate_prose, folded)


# ── PII 정규화 ──────────────────────────────────────────────────────────────


def normalize_digits(value: str) -> str:
    """숫자형 PII 정규화: 구분자·공백을 지우고 숫자열만 남긴다."""
    return _NON_DIGIT.sub("", value)


def normalize_phone(value: str) -> str:
    """전화번호 정규화: 구분자를 지우고 한국 국가번호를 국내 `0` 표기로 바꾼다."""
    digits = normalize_digits(value)
    international = value.startswith(("+", "00"))
    if value.startswith("00"):
        digits = digits[2:]
    if international and digits.startswith("82"):
        national = digits[2:]
        return national if national.startswith("0") else f"0{national}"
    if international:
        return f"+{digits}"
    return digits


def normalize_email(value: str) -> str:
    """이메일 정규화: 앞뒤 공백 제거 후 소문자화."""
    return value.strip().lower()


@dataclass(frozen=True)
class PiiPattern:
    """패턴형 PII 1종 — 탐지 정규식 · 탐지 전 접기 · 대조용 정규화를 함께 가진다.

    **접기가 패턴별인 것이 이 자료형의 요점이다.** 번호 계열은 구분자 변종을 하이픈으로
    접고 결합 표식을 무시해야 하지만, 이메일에 같은 접기를 걸면 서로 다른 주소가 같은
    값이 되어 반대 방향 우회가 열린다. 접기를 전역 함수 하나로 두면 그 둘을 가를 자리가
    없어서, 기본값 없이 **패턴마다 명시**하게 한다.
    """

    name: str
    regex: re.Pattern[str]
    fold: Callable[[str], str]
    normalize: Callable[[str], str]


# 숫자 경계 lookaround 를 붙이는 이유:
#   1) 초안 쪽 — 주문번호처럼 긴 숫자열 한가운데의 일부가 전화번호로 오탐되는 것을 막는다.
#   2) 근거 쪽 — 긴 숫자열의 조각이 allowlist 에 올라 지어낸 번호를 통과시키는 것을 막는다.
#
# **이 집합이 저장소의 패턴형 개인정보 정의 정본이고, 소유자는 이 모듈 하나다.** 근거 렌더의
# 개인정보 필터(`evidence._sql_evidence_texts`)와 조회 가드가 여기서 가져다 쓴다 — 층마다
# 자기 정의를 두면 한쪽만 넓혀져 기준이 갈리고, 접기가 층마다 달라 터진 것이 정확히 그
# 모양이었다. 층별 주입점을 두지 않는 것도 같은 이유다(구조 테스트가 검사한다).
DEFAULT_PII_PATTERNS: tuple[PiiPattern, ...] = (
    PiiPattern(
        name="mobile_phone",
        regex=re.compile(
            r"(?<![0-9])(?:(?:\+82|0082)[-. ()]*(?:0[-. ()]*)?1[016789]|01[016789])"
            r"[-. ()]*[0-9]{3,4}[-. ()]*[0-9]{4}(?![0-9])"
        ),
        fold=fold_numeric_for_detection,
        normalize=normalize_phone,
    ),
    PiiPattern(
        name="landline_phone",
        regex=re.compile(
            r"(?<![0-9])(?:(?:\+82|0082)[-. ()]*(?:0[-. ()]*)?(?:2|[3-9][0-9]?)|0[2-9][0-9]?)"
            r"[-. ()]*[0-9]{3,4}[-. ()]*[0-9]{4}(?![0-9])"
        ),
        fold=fold_numeric_for_detection,
        normalize=normalize_phone,
    ),
    # 15xx/16xx/17xx/18xx 대표번호. 개인 연락처는 아니지만 정책 문서의 미끼 조항이
    # 겨냥하는 값이 바로 이것이다(docs/engineering-notes.md "대표번호(15xx~18xx)를
    # PII 패턴에 넣은 이유") — 고객센터 번호를 비워 둔 조항을 근거로 받고 모델이
    # 일반 지식으로 번호를 채우면 `pii_detected` 로 걸려야 한다. 빼면 미끼가 무력해진다.
    PiiPattern(
        name="service_phone",
        regex=re.compile(r"(?<![+0-9])1[5-8][0-9]{2}[-. ()]*[0-9]{4}(?![0-9])"),
        fold=fold_numeric_for_detection,
        normalize=normalize_digits,
    ),
    PiiPattern(
        name="resident_registration_number",
        regex=re.compile(r"(?<![+0-9])[0-9]{6}[-. ()]*[1-4][0-9]{6}(?![0-9])"),
        fold=fold_numeric_for_detection,
        normalize=normalize_digits,
    ),
    # 명시적 +/00 접두와 7~15자리 숫자만 국제 전화 후보로 본다. 부호 없는 숫자열의
    # 국가를 추정하지 않으며 실제 통신망의 할당/유효성을 검증하는 규칙은 아니다.
    PiiPattern(
        name="international_phone",
        regex=_INTERNATIONAL_PHONE,
        fold=fold_numeric_for_detection,
        normalize=normalize_phone,
    ),
    # 이메일은 공통 접기 뒤 한국어 산문 경계만 분리한다. 번호 접기는 밑줄·빗금을 접어
    # `a_b@…` 와 `a-b@…` 가 같은 allowlist 값이 되고, 근거에 있는 주소를 한 글자 바꿔
    # 지어낸 주소가 통과한다 — 넓히기는 번호 계열 안에서만 한다.
    PiiPattern(
        name="email",
        regex=_EMAIL_REGEX,
        fold=_fold_email_for_detection,
        normalize=normalize_email,
    ),
)


def pii_shaped(text: str) -> bool:
    """텍스트에 패턴형 PII 가 한 건이라도 있으면 True — **탐지 전용 공개 표면**.

    값 대조 없이 "개인정보 모양인가"만 묻는 층이 쓴다: 근거 렌더의 개인정보 필터가
    계산 컬럼의 이름·값을 `evidence_text` 에서 걸러낼 때, 그리고 조회 가드가 고정값을
    승인할지 볼 때. **그 층들이 자기 정규식을 따로 두면 저장소에 개인정보 정의가 셋이
    되고, 한쪽만 넓혀지는 순간 기준이 갈린다** — 접기가 층마다 달라 뚫렸던 경로가
    정확히 그 모양이었다(docs/engineering-notes.md "접기 기준이 층마다 달라 근거
    필터가 게이트보다 헐거웠다").

    패턴별 접기를 그대로 태운다. 접기는 판정에만 쓰고 값을 갈아 끼우지 않는다.
    """
    return any(pattern.regex.search(pattern.fold(text)) for pattern in DEFAULT_PII_PATTERNS)


# ── 진입점 ──────────────────────────────────────────────────────────────────


def evaluate_draft(*, raw_draft: object, evidences: Sequence[Evidence]) -> GateResult:
    """원시 초안을 이번 문의의 수집 근거와 대조해 `pass | reject + 사유 목록` 을 낸다.

    `raw_draft` 는 **무엇이든 들어올 수 있다** — 초안 생성이 형식 불일치 시 원문 문자열을
    그대로 넘기기 때문이다. dict 가 아닌 값에도 예외를 던지지 않고 `schema_violation`
    으로 판정한다.

    **패턴 집합은 인자로 받지 않는다** — 정본은 `DEFAULT_PII_PATTERNS` 하나이고 근거
    필터·조회 가드가 같은 것을 쓴다.
    """
    reasons: set[RejectReason] = set()

    inspection = _inspect_schema(raw_draft)
    if not inspection.ok:
        reasons.add(RejectReason.SCHEMA_VIOLATION)

    known_ids = {evidence.id for evidence in evidences}
    for citation_ids in inspection.citation_lists:
        if not citation_ids:
            reasons.add(RejectReason.MISSING_CITATION)
        # 문자열이 아닌 원소는 이미 schema_violation 이다 — 참조 무결성은 문자열만 본다.
        if any(isinstance(cid, str) and cid not in known_ids for cid in citation_ids):
            reasons.add(RejectReason.INVALID_CITATION)

    # PII 검사는 텍스트만 있으면 할 수 있으므로 스키마가 깨져도 가능한 한 수행한다.
    if _has_unsourced_pii(
        draft_texts=_answer_texts(inspection=inspection, raw_draft=raw_draft),
        evidence_texts=[evidence.evidence_text for evidence in evidences],
        answer_text=inspection.draft.answer_text if inspection.draft is not None else None,
    ):
        reasons.add(RejectReason.PII_DETECTED)

    ordered = tuple(reason for reason in REASON_ORDER if reason in reasons)
    return GateResult(
        verdict=Verdict.REJECT if ordered else Verdict.PASS,
        reject_reasons=ordered,
    )


def to_draft(raw_draft: object) -> Draft:
    """L1 스키마 검사를 통과한 원시 초안을 `Draft` 로 바꾼다.

    구조가 깨진 초안은 `ValueError` 다 — 판정은 `evaluate_draft` 가 하고, 이 함수는
    **통과한 초안에만** 쓴다. citation_ids 가 비어 있는 것은 구조 오류가 아니므로
    (그건 `missing_citation` 전담) 여기서 막지 않는다.
    """
    inspection = _inspect_schema(raw_draft)
    if inspection.draft is None:
        raise ValueError("L1 스키마 검사를 통과하지 못한 초안은 Draft 로 바꿀 수 없다")
    return inspection.draft


# ── 스키마 검사 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SchemaInspection:
    """구조 검사 결과 + 하위 검사가 가능한 부분.

    `citation_lists` 는 citation 검사를 할 수 있는 claim 들의 원시 citation_ids 목록이다
    (구조가 깨진 claim 은 빠진다). `draft` 는 구조가 완전할 때만 채워진다.

    `answer_texts` 는 **답변으로 나갈 수 있는 텍스트**다 — claim 의 `text` 중 문자열인 것
    전부. `draft` 와 달리 구조가 깨져도 채워진다(깨진 claim 옆의 멀쩡한 claim 은 여전히
    답변 후보다). PII 검사 대상이 이것이다.
    """

    ok: bool
    citation_lists: tuple[tuple[object, ...], ...]
    answer_texts: tuple[str, ...]
    draft: Draft | None


def _inspect_schema(raw_draft: object) -> _SchemaInspection:
    """구조(필수 키·타입·claims 비어 있음)까지만 본다.

    **`citation_ids` 최소 개수 제약을 여기에 넣지 않는다.** 넣으면 `missing_citation` 이
    영원히 발화하지 않아 사유 분리가 무너진다(docs/business-rules.md "L1 게이트 판정 규칙").

    스키마에 없는 추가 키는 위반으로 보지 않는다 — 답변 계약은 L2 가 그대로 이어받는
    계약이고, 추가 키는 claim 이 근거를 딛고 섰는지와 무관하기 때문이다.
    """
    if not isinstance(raw_draft, Mapping):
        return _SchemaInspection(ok=False, citation_lists=(), answer_texts=(), draft=None)

    claims = raw_draft.get(_CLAIMS_KEY)
    if not isinstance(claims, list | tuple) or not claims:
        return _SchemaInspection(ok=False, citation_lists=(), answer_texts=(), draft=None)

    ok = True
    citation_lists: list[tuple[object, ...]] = []
    answer_texts: list[str] = []
    parsed: list[Claim] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            ok = False
            continue

        text = claim.get(_TEXT_KEY)
        if isinstance(text, str):
            answer_texts.append(text)
        if not isinstance(text, str) or not any(
            not ch.isspace()
            and ch not in _BLANK_CHARACTERS
            and unicodedata.category(ch) not in ("Cf", "Cc")
            and not unicodedata.category(ch).startswith("M")
            for ch in text
        ):
            ok = False

        citation_ids = claim.get(_CITATION_IDS_KEY)
        if not isinstance(citation_ids, list | tuple):
            ok = False
            continue

        citation_lists.append(tuple(citation_ids))
        if not all(isinstance(cid, str) for cid in citation_ids):
            ok = False
            continue

        if isinstance(text, str):
            parsed.append(Claim(text=text, citation_ids=tuple(citation_ids)))

    return _SchemaInspection(
        ok=ok,
        citation_lists=tuple(citation_lists),
        answer_texts=tuple(answer_texts),
        draft=Draft(claims=tuple(parsed)) if ok else None,
    )


# ── PII 검사 ────────────────────────────────────────────────────────────────


def _answer_texts(*, inspection: _SchemaInspection, raw_draft: object) -> Sequence[str]:
    """PII 검사 대상을 고른다 — **답변으로 나갈 수 있는 텍스트만.**

    `to_draft` 가 살려내는 것은 claim 의 `text` 뿐이고 답변은 그것들의 연결이다
    (`Draft.answer_text`). 개별 claim 과 연결된 답변을 함께 검사한다 — 개별 조각이
    번호가 아니어도 답변의 공백 연결로 전화번호가 될 수 있기 때문이다.
    초안의 나머지 키(최상위 `debug`, claim 안의 `note` 등)는
    답변에 실리지 않으므로 검사하면 **답변에 없는 값으로 정상 초안을 기각하게 된다.**
    초안 JSON 은 LLM 산출이라 이런 키가 언제든 늘어날 수 있고, 헤드라인 지표가
    "정상 초안 오탐률"이므로 그 오탐이 곧 지표 오염이다
    (`src/reply_gate/AGENTS.md` 불변식 7 · docs/business-rules.md "PII 규칙").

    **답변 후보 텍스트를 하나도 식별할 수 없을 때만** 초안 전체를 긁는다 — 원문 문자열이
    통째로 넘어온 경우처럼 무엇이 답변인지 코드가 모르는 상태다. 그런 초안은 어차피
    `schema_violation` 으로 기각되므로 답변이 나가지는 않지만, 게이트는 모르는 쪽에서
    검사하는 방향으로 보수적으로 간다.
    """
    if inspection.draft is not None:
        return (*inspection.answer_texts, inspection.draft.answer_text)
    if inspection.answer_texts:
        return inspection.answer_texts
    return _collect_texts(raw_draft)


def _collect_texts(value: object, *, depth: int = 0) -> list[str]:
    """원시 초안에서 문자열을 전부 긁어온다 (dict 순서대로 — 결정론).

    답변 후보를 식별할 수 없을 때의 폴백 전용이다(`_answer_texts`).

    **`citation_ids` 는 제외한다.** docs/business-rules.md "PII 규칙" 의 검사 대상은
    "초안 텍스트" — 최종 사용자에게
    보여줄 답변이다. 근거 ID 는 답변 문장이 아니라 식별자이고, `sql:<문의 UUID>:<순번>`
    형식이라 UUID 의 16진 숫자 구간이 전화번호 패턴에 우연히 걸린다. 근거 ID 는
    `evidence_text` 에 들어가지 않으므로 allowlist 에도 오르지 않아, 그대로 두면
    **PII 가 전혀 없는 정상 초안이 기각된다** (무작위 UUID 20,000개 중 168개 = 0.84%).
    이 프로젝트의 헤드라인 지표가 "정상 초안 오탐률"이므로 그 오탐이 곧 지표 오염이다.
    참조 무결성은 `invalid_citation` 이 이미 전담하므로 여기서 볼 이유도 없다.
    """
    if depth > _MAX_TEXT_DEPTH:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            text
            for key, item in value.items()
            if key != _CITATION_IDS_KEY
            for text in _collect_texts(item, depth=depth + 1)
        ]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _collect_texts(item, depth=depth + 1)]
    return []


def _without_email_tokens(text: str) -> str:
    """이메일과 같은 접기 버퍼에서 주소 전체를 제외한다. 원문 span을 재사용하지 않는다."""
    folded = _fold_email_for_detection(text)
    # no-@의 빈 반환값은 이메일 후보가 없다는 뜻이지, 근거 전체가 없다는 뜻이 아니다.
    return _EMAIL_REGEX.sub(" ", folded) if folded else text


def _normalized_matches(
    *,
    texts: Sequence[str],
    pattern: PiiPattern,
    sourced_numbers: frozenset[str] = frozenset(),
) -> set[str]:
    """패턴 하나가 텍스트들에서 뽑은 **정규화된 값** 집합. 접기는 그 패턴의 것을 쓴다."""
    values: set[str] = set()
    for text in texts:
        candidate = text if pattern.name == "email" else _without_email_tokens(text)
        if pattern.name != "email" and sourced_numbers:
            # 같은 완전한 정수가 근거에 있는 수량 토큰에서만 '+'를 대조용으로 접는다.
            # 단위 없는 전화까지 전역 allowlist에 넣거나 pii_shaped의 탐지를 끄지 않는다.
            candidate = _QUANTITY.sub(
                lambda match: (
                    match.group()[1:]
                    if match["sign"] == "+" and normalize_digits(match["number"]) in sourced_numbers
                    else match.group()
                ),
                _fold_numeric_characters(candidate),
            )
        folded = pattern.fold(candidate)
        values.update(pattern.normalize(match) for match in pattern.regex.findall(folded))
    return values


def _has_unsourced_pii(
    *, draft_texts: Sequence[str], evidence_texts: Sequence[str], answer_text: str | None
) -> bool:
    """초안의 패턴형 PII 중 **근거에서 유래하지 않은 값**이 있으면 True.

    대조 방식 — 근거 텍스트에도 **같은 패턴·같은 정규화**를 적용해 값 집합을 뽑고,
    정규화된 값끼리 **완전 일치**로 비교한다. 덕분에 근거의 `010-1234-5678` 과 초안의
    `01012345678` 은 통과하고, 지어낸 번호는 기각된다.

    정규화한 숫자열을 근거 텍스트에 '부분 문자열 포함' 으로 대조하지 않는 이유:
    근거에 주문번호 같은 긴 숫자열이 있으면 짧은 전화번호가 우연히 그 안에 들어가
    지어낸 번호가 통과해 버린다(= PII 유출). 완전 일치는 이 오탐 경로를 통째로 없앤다.
    국내형과 한국 국가번호(+82/0082), 공백·괄호 구분자는 같은 값으로 접는다. 그래도 패턴이
    모르는 표기는 정상 에코라도 기각될 수 있는데, L1 실패는 인계(escalation)로 흘러 안전한
    쪽이므로 이 방향의 보수성을 택한다.

    비패턴형 개인정보(이름·주소)는 정규식으로 잡을 수 없어 **L1 의 검사 대상이 아니다** —
    그리고 **L2 의 대상도 아니다**(L2 는 근거와 claim 의 정합만 본다). 어느 층의 검사
    대상도 아니므로 커버리지를 과장하지 않는다.

    **패턴 집합은 주입받지 않는다.** 이 저장소의 정본은 `DEFAULT_PII_PATTERNS` 하나이고,
    근거 필터·조회 가드도 같은 집합을 가져다 쓴다 — 층마다 다른 집합을 받을 수 있으면
    한쪽만 넓혀져 기준이 갈린다.
    """
    # 값의 표기 차이만 대조한다. SQL의 단위 없는 정수도 근거이며 단위 의미는 L2다.
    # 0인 소수부만 정수부와 정확히 비교한다. float 반올림이나 소수점 제거는 하지 않는다.
    # 긴 ID의 조각·비정수·음수·이메일 일부는 독립된 양수 정수로 채택하지 않는다.
    sourced_numbers: set[str] = set()
    for text in evidence_texts:
        numeric = _fold_numeric_characters(_without_email_tokens(text))
        sourced_numbers.update(
            normalize_digits(match["number"])
            for match in _QUANTITY.finditer(numeric)
            if not numeric[: match.start()].rstrip().endswith("-")
        )
        sourced_numbers.update(
            normalize_digits(match["number"])
            for match in _INTEGER_VALUE.finditer(numeric)
            if not numeric[: match.start()].rstrip().endswith("-")
        )
    quantity_references = frozenset(sourced_numbers)
    found: set[str] = set()
    for pattern in DEFAULT_PII_PATTERNS:
        # 이메일 접미사는 열린 구분자 문맥에 의존한다. 정상 초안은 최종 답변만 읽어야
        # '<주소' / 'a@example.com입니다' / '>'처럼 갈린 literal을 산문으로 오인하지 않는다.
        texts = (
            (answer_text,) if pattern.name == "email" and answer_text is not None else draft_texts
        )
        found |= _normalized_matches(
            texts=texts, pattern=pattern, sourced_numbers=quantity_references
        )
    if not found:
        return False

    # allowlist 는 패턴별로 나누지 않고 하나로 합친다 — 근거에서 어느 패턴이 뽑았든
    # 근거에 있었다는 사실은 같기 때문이다(숫자열과 이메일은 정규화 결과가 겹치지 않는다).
    allowed: set[str] = set()
    for pattern in DEFAULT_PII_PATTERNS:
        allowed |= _normalized_matches(
            texts=evidence_texts, pattern=pattern, sourced_numbers=quantity_references
        )

    return bool(found - allowed)
