"""기존 헤드라인과 분리한 평가 파일럿. 기본 대조는 DB·모델을 호출하지 않는다.

정답 자료의 완전성과 결정론 경계만 검사한다. 작성한 기대값의 수용을 실제 모델의
정확도로 바꾸지 않는다. 의미 정답은 docs/evaluation-pilot.md의 독립 검토 대상이다.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from reply_gate.contracts import Evidence, EvidenceSource, RejectReason, Verdict
from reply_gate.gate import evaluate_draft
from reply_gate.policy_index import load_policy_documents
from reply_gate.sql_guard import SqlGuardRejection, validate_sql

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "data" / "pilot" / "v1"
ORDER_FIXTURE = ROOT / "db" / "fixtures" / "orders.jsonl"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PilotEvidence(_StrictModel):
    id: str = Field(min_length=1)
    source: EvidenceSource
    content: str = Field(min_length=1)

    def as_evidence(self) -> Evidence:
        return Evidence(
            id=self.id, source=self.source, content=self.content, evidence_text=self.content
        )


class SQLOracle(_StrictModel):
    operation: Literal["field", "multiply"]
    fields: tuple[str, ...]
    unit: Literal["KRW", "count", "status", "timestamp"]

    @model_validator(mode="after")
    def check_arity(self) -> SQLOracle:
        if len(self.fields) != (1 if self.operation == "field" else 2):
            raise ValueError("oracle 연산과 원본 필드 수가 다르다")
        units = {
            "total_price_krw": "KRW",
            "unit_price_krw": "KRW",
            "quantity": "count",
            "status": "status",
            "ordered_at": "timestamp",
            "shipped_at": "timestamp",
            "delivered_at": "timestamp",
        }
        if set(self.fields) - units.keys():
            raise ValueError("oracle는 명시한 비개인정보 원본 필드만 읽는다")
        if self.operation == "field" and self.unit != units[self.fields[0]]:
            raise ValueError("원본 필드와 단위가 다르다")
        if self.operation == "multiply" and (
            set(self.fields) != {"unit_price_krw", "quantity"} or self.unit != "KRW"
        ):
            raise ValueError("허용한 계산은 단가 곱하기 수량의 KRW 결과뿐이다")
        return self


class SQLProbe(_StrictModel):
    order_no: str
    query: str = Field(min_length=1)
    oracle: SQLOracle
    expected_rows: list[dict[str, JsonValue]]
    candidate_rows: list[dict[str, JsonValue]]


class PilotCase(_StrictModel):
    id: str = Field(pattern=r"^[SPLI]\d{2}-(good|bad)$")
    family_id: str = Field(pattern=r"^[SPLI]\d{2}$")
    category: Literal["sql", "policy", "language", "identity"]
    split: Literal["development", "reserved"]
    variant: Literal["good", "bad"]
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_refs: tuple[str, ...] = Field(min_length=1)
    required: tuple[str, ...] = Field(min_length=1)
    forbidden: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidences: tuple[PilotEvidence, ...] = ()
    raw_draft: JsonValue = None
    expected_l1: Verdict | None
    expected_l1_reasons: tuple[RejectReason, ...] = ()
    expected_l2_reasons: tuple[RejectReason, ...] | None
    sql: SQLProbe | None = None
    identity: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def check_contract(self) -> PilotCase:
        if self.id != f"{self.family_id}-{self.variant}":
            raise ValueError("사례 ID와 가족·변형이 다르다")
        prefix = {"sql": "S", "policy": "P", "language": "L", "identity": "I"}[self.category]
        if not self.family_id.startswith(prefix):
            raise ValueError("가족 ID와 평가 축이 다르다")
        if (self.category == "sql") != (self.sql is not None):
            raise ValueError("SQL 축에만 SQL probe가 필요하다")
        if (self.category == "identity") != (self.identity is not None):
            raise ValueError("확인 축에만 확인 시나리오가 필요하다")
        if self.category == "identity":
            if self.expected_l1 is not None or self.raw_draft is not None:
                raise ValueError("확인 흐름 전체를 L1 초안 판정으로 위장하지 않는다")
        elif self.raw_draft is None or self.expected_l1 is None or not self.evidences:
            raise ValueError("초안 대조에 입력·근거·기대 L1이 필요하다")
        if (self.expected_l1 is Verdict.REJECT) != bool(self.expected_l1_reasons):
            raise ValueError("L1 기각 여부와 기대 사유가 다르다")
        l1 = set(RejectReason) - {
            RejectReason.UNSUPPORTED_CLAIM,
            RejectReason.CONTRADICTORY_EVIDENCE,
        }
        if set(self.expected_l1_reasons) - l1:
            raise ValueError("L1 기대에 의미 판정 사유를 넣지 않는다")
        if self.expected_l2_reasons is not None and set(self.expected_l2_reasons) & l1:
            raise ValueError("L2 기대에 L1 사유를 넣지 않는다")
        if self.variant == "good" and (self.expected_l1_reasons or self.expected_l2_reasons):
            raise ValueError("정상 짝은 게이트 기각을 기대하지 않는다")
        return self

    def judge_input(self) -> dict[str, JsonValue]:
        """정답·분할·평가 메모를 모델 입력으로 넘기지 않는 명시적 경계."""
        return {
            "question": self.question,
            "draft": self.raw_draft,
            "evidences": [
                {"id": item.id, "source": item.source.value, "content": item.content}
                for item in self.evidences
            ],
        }


def _read_jsonl(path: Path) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalized_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, JsonValue]]:
    """실제 DB 시각은 UTC로 맞춘다. 값이나 타입 보정으로 불일치를 숨기지 않는다."""
    normalized: list[dict[str, JsonValue]] = []
    for row in rows:
        result: dict[str, JsonValue] = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    raise ValueError("시간대가 없는 시각은 비교하지 않는다")
                result[key] = value.astimezone(UTC).isoformat()
            elif value is None or type(value) in (str, int, float, bool):
                result[key] = value
            else:
                raise ValueError(f"지원하지 않는 SQL 값 타입: {type(value).__name__}")
        normalized.append(result)
    # NaN/Infinity 관측을 업무상 일치/불일치로 판정하지 않는다.
    json.dumps(normalized, allow_nan=False)
    return normalized


def rows_match(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> bool:
    """JSON 수는 수치로, bool/string/null은 타입을 보존하여 비교한다."""
    left, right = normalized_rows(actual), normalized_rows(expected)
    if len(left) != len(right):
        return False
    for actual_row, expected_row in zip(left, right, strict=True):
        if actual_row.keys() != expected_row.keys():
            return False
        for key, value in actual_row.items():
            wanted = expected_row[key]
            if type(value) in (int, float) and type(wanted) in (int, float):
                if Decimal(str(value)) != Decimal(str(wanted)):
                    return False
            elif type(value) is not type(wanted) or value != wanted:
                return False
    return True


def oracle_rows(probe: SQLProbe, record: Mapping[str, Any]) -> list[dict[str, JsonValue]]:
    """SQL·후보 결과를 읽지 않고 주문의 원본 필드만으로 정답을 구한다."""
    values = [record[field] for field in probe.oracle.fields]
    if probe.oracle.operation == "multiply":
        if any(type(value) is not int for value in values):
            raise ValueError("금액 곱셈 원본은 정수여야 한다")
        result: Any = values[0] * values[1]
    else:
        result = values[0]
    if probe.oracle.unit == "timestamp" and result is not None:
        result = datetime.fromisoformat(result)
    return normalized_rows([{"value": result}])


def load_pilot(directory: Path = PILOT_DIR) -> tuple[PilotCase, ...]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["review_status"] != "pending_independent_review":
        raise ValueError("v1 무료 검사로 독립 검토 승인을 만들지 않는다")
    digest = hashlib.sha256(ORDER_FIXTURE.read_bytes()).hexdigest()
    if manifest["order_fixture_sha256"] != digest:
        raise ValueError("주문 원본 판이 바뀌었다. 정답을 다시 검토해야 한다")
    policy_paths = {str(path.relative_to(ROOT)) for path in (ROOT / "data/policies").glob("*.md")}
    if set(manifest["policy_sources"]) != policy_paths:
        raise ValueError("정책 원본 지문 목록에 누락이나 추가가 있다")
    for relative, expected_hash in manifest["policy_sources"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"정책 원본 판이 바뀌었다: {relative}")
    for name, field in (
        ("cases.jsonl", "cases_sha256"),
        ("j11_counterfactuals.jsonl", "j11_sha256"),
    ):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest[field]:
            raise ValueError(f"파일럿 자료와 manifest 지문이 다르다: {name}")
    cases = tuple(PilotCase.model_validate(row) for row in _read_jsonl(directory / "cases.jsonl"))
    families: dict[str, list[PilotCase]] = defaultdict(list)
    for case in cases:
        families[case.family_id].append(case)
    wanted = {
        f"{prefix}{i:02}"
        for prefix, count in (("S", 8), ("P", 6), ("L", 6), ("I", 4))
        for i in range(1, count + 1)
    }
    if set(families) != wanted or len({case.id for case in cases}) != 48:
        raise ValueError("24가족/48개 고유 사례가 필요하다")
    for family, pair in families.items():
        if len(pair) != 2 or {case.variant for case in pair} != {"good", "bad"}:
            raise ValueError(f"{family}: 정상·결함 짝이 필요하다")
        if len({case.split for case in pair}) != 1:
            raise ValueError(f"{family}: 같은 가족이 개발/보류 양쪽에 있다")
        if pair[0].question != pair[1].question:
            raise ValueError(f"{family}: 짝의 질문이 다르다")
        if pair[0].source_refs != pair[1].source_refs:
            raise ValueError(f"{family}: 짝의 정답 출처가 다르다")
        if (
            pair[0].sql is not None
            and pair[1].sql is not None
            and (
                pair[0].sql.order_no != pair[1].sql.order_no
                or pair[0].sql.oracle != pair[1].sql.oracle
                or not rows_match(pair[0].sql.expected_rows, pair[1].sql.expected_rows)
            )
        ):
            raise ValueError(f"{family}: SQL 짝의 독립 정답이 다르다")
    if Counter(case.split for case in cases) != {"development": 32, "reserved": 16}:
        raise ValueError("개발32/보류16 분할이 다르다")
    records = {row["order_no"]: row for row in _read_jsonl(ORDER_FIXTURE)}
    policy_texts = {
        chunk.evidence_id: chunk.content
        for document in load_policy_documents()
        for chunk in document.chunks
    }
    for case in cases:
        for item in case.evidences:
            if item.id in policy_texts and item.content != policy_texts[item.id]:
                raise ValueError(f"{case.id}: 인용한 정책 원문과 snapshot이 다르다")
        if case.sql is not None and not rows_match(
            oracle_rows(case.sql, records[case.sql.order_no]), case.sql.expected_rows
        ):
            raise ValueError(f"{case.id}: SQL 기대값이 원본 oracle와 다르다")
    return cases


def check_development(cases: Sequence[PilotCase]) -> dict[str, Any]:
    """보류 묶음에는 제품 게이트를 호출하지 않는다. 미측정을 0으로 접지 않는다."""
    results: list[dict[str, Any]] = []
    for case in cases:
        if case.split != "development":
            continue
        row: dict[str, Any] = {"id": case.id, "family_id": case.family_id, "l1": None}
        if case.expected_l1 is not None:
            observed = evaluate_draft(
                raw_draft=case.raw_draft,
                evidences=tuple(item.as_evidence() for item in case.evidences),
            )
            matched = (
                observed.verdict == case.expected_l1
                and observed.reject_reasons == case.expected_l1_reasons
            )
            row["l1"] = {
                "verdict": observed.verdict.value,
                "reasons": [reason.value for reason in observed.reject_reasons],
                "matches_proposed_label": matched,
            }
        if case.sql is not None:
            try:
                validate_sql(case.sql.query, order_no=case.sql.order_no, max_rows=1)
                row["sql_guard"] = "accepted"
            except SqlGuardRejection as exc:
                row["sql_guard"] = "rejected"
                row["sql_guard_reason"] = str(exc)
            row["sql_result"] = None
            row["sql_result_note"] = "실제 SQL 결과는 소유한 격리 DB에서 별도 대조한다"
        if case.category == "identity":
            row["identity_runtime"] = None
            row["identity_note"] = "확인 흐름은 별도 HTTP/DB 검사 대상이다"
        row["l2"] = None
        row["semantic_accuracy"] = None
        results.append(row)
    observed_l1 = [row["l1"] for row in results if row["l1"] is not None]
    return {
        "suite": "evaluation-pilot-v1",
        "review_status": "pending_independent_review",
        "case_version": hashlib.sha256(
            json.dumps(
                [case.model_dump(mode="json") for case in cases],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "model_calls": 0,
        "database_calls": 0,
        "families": len({case.family_id for case in cases}),
        "cases": len(cases),
        "development_cases": len(results),
        "reserved_cases": sum(case.split == "reserved" for case in cases),
        "reserved_executed": False,
        "l1_checked": len(observed_l1),
        "l1_label_matches": sum(item["matches_proposed_label"] for item in observed_l1),
        "l2_accuracy": None,
        "semantic_accuracy": None,
        "outcomes": results,
    }
