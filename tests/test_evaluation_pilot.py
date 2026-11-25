"""파일럿의 정답 독립성·가족 분리·미측정 표시와 실제 SQL 결과 대조."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate import evaluation_pilot
from reply_gate.contracts import Claim, Draft, Evidence, EvidenceSource, Verdict
from reply_gate.evaluation_pilot import (
    PILOT_DIR,
    PilotCase,
    SQLOracle,
    check_development,
    load_pilot,
    normalized_rows,
    rows_match,
)
from reply_gate.gate import evaluate_draft
from reply_gate.judge import _parse_judge_result
from reply_gate.sql_guard import validate_sql

CASES = load_pilot()
DEVELOPMENT_SQL = [case for case in CASES if case.split == "development" and case.sql is not None]


def _복사(tmp_path: Path) -> Path:
    destination = tmp_path / "pilot"
    shutil.copytree(PILOT_DIR, destination)
    return destination


def _사례를_다시_쓴다(directory: Path, rows: list[dict[str, Any]]) -> None:
    target = directory / "cases.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_가족_스물넷은_정상과_결함을_같은_묶음에_둔다() -> None:
    assert len(CASES) == 48
    assert Counter(case.category for case in CASES) == {
        "sql": 16,
        "policy": 12,
        "language": 12,
        "identity": 8,
    }
    assert Counter(case.split for case in CASES) == {"development": 32, "reserved": 16}
    for family in {case.family_id for case in CASES}:
        pair = [case for case in CASES if case.family_id == family]
        assert {case.variant for case in pair} == {"good", "bad"}
        assert len({case.split for case in pair}) == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "split", "oracle"])
def test_자료_지문을_다시_적어도_빠진_짝과_분할_유출과_잘못된_정답은_거부한다(
    tmp_path: Path, mutation: str
) -> None:
    """지문 일치는 내용 정합성의 대용이 아니다."""
    directory = _복사(tmp_path)
    rows = [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "split":
        rows[0]["split"] = "reserved"
    else:
        rows[0]["sql"]["expected_rows"] = [{"value": -1}]
    _사례를_다시_쓴다(directory, rows)
    with pytest.raises(ValueError):
        load_pilot(directory)


def test_파일럿_지문이_안_맞으면_실행_전에_거부한다(tmp_path: Path) -> None:
    directory = _복사(tmp_path)
    target = directory / "cases.jsonl"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest"):
        load_pilot(directory)


def test_개발_대조는_보류_사례와_모델을_실행한_척하지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    original = evaluate_draft

    def recording(**kwargs: Any) -> Any:
        calls.append(kwargs["raw_draft"])
        return original(**kwargs)

    monkeypatch.setattr(evaluation_pilot, "evaluate_draft", recording)
    result = check_development(CASES)
    assert len(calls) == result["l1_checked"] == 28
    assert result["development_cases"] == 32
    assert result["reserved_executed"] is False
    assert result["model_calls"] == result["database_calls"] == 0
    assert result["semantic_accuracy"] is None
    assert result["l2_accuracy"] is None
    assert all(row["l2"] is None for row in result["outcomes"])
    assert {row["id"] for row in result["outcomes"]} == {
        case.id for case in CASES if case.split == "development"
    }
    assert result["l1_label_matches"] == 28


def test_모델_입력에_정답과_분할과_검토_메모를_전달하지_않는다() -> None:
    original = next(case for case in CASES if case.category == "policy")
    payload = original.model_dump(mode="json")
    payload["rationale"] = "ANSWER_KEY_SENTINEL_7139"
    payload["required"] = ["ANSWER_KEY_SENTINEL_7139"]
    payload["forbidden"] = ["ANSWER_KEY_SENTINEL_7139"]
    poisoned = PilotCase.model_validate(payload)
    request = poisoned.judge_input()
    assert set(request) == {"question", "draft", "evidences"}
    assert "ANSWER_KEY_SENTINEL_7139" not in json.dumps(request)
    assert request == original.judge_input()


@pytest.mark.parametrize(
    "fields,operation,unit",
    [
        (["quantity"], "field", "KRW"),
        (["total_price_krw"], "field", "count"),
        (["status"], "field", "timestamp"),
        (["total_price_krw", "quantity"], "multiply", "KRW"),
    ],
)
def test_숫자가_맞아도_원본_단위와_계산_의미가_다르면_정답으로_등록하지_못한다(
    fields: list[str], operation: str, unit: str
) -> None:
    with pytest.raises(ValueError):
        SQLOracle.model_validate({"fields": fields, "operation": operation, "unit": unit})


def test_정책_지문_목록을_비워_변경_검사를_우회하지_못한다(tmp_path: Path) -> None:
    directory = _복사(tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["policy_sources"] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="누락"):
        load_pilot(directory)


def test_정책_ID만_맞고_본문이_바뀐_근거를_원문으로_취급하지_않는다(tmp_path: Path) -> None:
    directory = _복사(tmp_path)
    rows = [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
    row = next(row for row in rows if row["id"] == "P01-good")
    row["evidences"][0]["content"] = "원문과 다른 정량 보장입니다."
    _사례를_다시_쓴다(directory, rows)
    with pytest.raises(ValueError, match="snapshot"):
        load_pilot(directory)


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([{"value": True}], [{"value": 1}]),
        ([{"value": "3"}], [{"value": 3}]),
        ([{"value": None}], [{"value": 0}]),
        ([{"wrong_column": 3}], [{"value": 3}]),
        ([], [{"value": 3}]),
        ([{"value": 3}, {"value": 3}], [{"value": 3}]),
        ([{"value": float(2**53)}], [{"value": 2**53 + 1}]),
    ],
)
def test_SQL_정답은_값뿐_아니라_타입과_열과_행수도_구분한다(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    assert not rows_match(actual, expected)
    assert rows_match(expected, expected)


def test_SQL의_같은_JSON_숫자값은_정수와_소수_표기로_오기각하지_않는다() -> None:
    assert rows_match([{"value": 3.0}], [{"value": 3}])
    assert not rows_match([{"value": True}], [{"value": 1.0}])


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_SQL_비유한_관측은_채점_결과로_바꾸지_않는다(number: float) -> None:
    with pytest.raises(ValueError):
        normalized_rows([{"value": number}])


@pytest.mark.db
@pytest.mark.parametrize("case", DEVELOPMENT_SQL, ids=[case.id for case in DEVELOPMENT_SQL])
def test_SQL의_조회_안전성과_업무_정답은_별도로_대조한다(
    case: PilotCase, ro_conn: psycopg.Connection[DictRow]
) -> None:
    """같은 주문의 허용 SELECT도 질문의 정답이 아닐 수 있음을 실제 DB로 구분한다."""
    assert case.sql is not None
    validated = validate_sql(case.sql.query, order_no=case.sql.order_no, max_rows=1)
    rows = ro_conn.execute(validated.sql).fetchall()
    assert rows_match(rows, case.sql.candidate_rows)
    assert rows_match(rows, case.sql.expected_rows) == (case.variant == "good")


J11_CASES = [
    json.loads(line) for line in (PILOT_DIR / "j11_counterfactuals.jsonl").read_text().splitlines()
]


@pytest.mark.parametrize("row", J11_CASES, ids=[row["id"] for row in J11_CASES])
def test_J11_네_상태는_L1과_기대_L2산출의_파서_정합성까지만_검사한다(
    row: dict[str, Any],
) -> None:
    """기대 JSON을 해석할 수 있다는 것이 모델이 그 JSON을 만든다는 뜻은 아니다."""
    evidence = tuple(
        Evidence(
            id=item["id"],
            source=EvidenceSource(item["source"]),
            content=item["content"],
            evidence_text=item.get("evidence_text", item["content"]),
        )
        for item in row["evidences"]
    )
    raw = row["raw_draft"]
    assert evaluate_draft(raw_draft=raw, evidences=evidence).verdict is Verdict.PASS
    draft = Draft(
        claims=tuple(
            Claim(text=c["text"], citation_ids=tuple(c["citation_ids"])) for c in raw["claims"]
        )
    )
    parsed = _parse_judge_result(row["expected_judge"], draft=draft, evidence=evidence)
    assert parsed.verdict.value == row["expected_judge"]["verdict"]
    assert [reason.value for reason in parsed.reject_reasons] == row["expected_judge"][
        "reject_reasons"
    ]
    assert len(parsed.contradictions) == 1
