"""평가 입력 지문은 파일이 아니라 실행 동안 잠근 실제 DB 행을 재야 한다."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from scripts import evaluate

from reply_gate.config import Settings
from reply_gate.db import connect
from reply_gate.evaluation import measure_pipeline_agreement
from tests.test_evaluation import _NO_CONN, ScriptedPipeline


def _empty_agreement() -> Any:
    return measure_pipeline_agreement(
        cases=[], pipeline=ScriptedPipeline([]), app_conn=_NO_CONN, readonly_conn=_NO_CONN
    )


@pytest.mark.db
def test_평가_중_입력_쓰기를_막고_실제_행_지문을_남긴다(
    seeded_order_count: int, db_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    del seeded_order_count
    args = evaluate.build_parser().parse_args(["--live"])
    monkeypatch.setattr(evaluate, "_clients", lambda **kw: (None, None, "synthetic", "synthetic"))
    monkeypatch.setattr(evaluate, "build_pipeline", lambda **kw: None)
    observed: list[str] = []

    def measure(**kwargs: Any) -> Any:
        for table in ("orders", "policy_chunks"):
            with connect(settings=db_settings) as other:
                try:
                    other.execute("SET LOCAL lock_timeout = '100ms'")
                    with pytest.raises(psycopg.errors.LockNotAvailable):
                        other.execute(f"LOCK TABLE {table} IN ROW EXCLUSIVE MODE")
                    observed.append(table)
                finally:
                    other.rollback()
        return _empty_agreement()

    monkeypatch.setattr(evaluate, "measure_pipeline_agreement", measure)
    agreement, _, _ = evaluate._run_measurement_two(cases=(), args=args, settings=db_settings)
    assert observed == ["orders", "policy_chunks"]
    assert agreement.input_fingerprint["order_data_version"].startswith("orders-db-")
    assert agreement.input_fingerprint["policy_data_version"].startswith("policy-db-")
    with connect(settings=db_settings) as other:
        other.execute("LOCK TABLE orders, policy_chunks IN ROW EXCLUSIVE MODE NOWAIT")
        other.rollback()


@pytest.mark.db
def test_파일을_바꾸지_않은_DB_입력_변경도_평가_지문이_잡는다(
    seeded_order_count: int, db_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    del seeded_order_count
    args = evaluate.build_parser().parse_args(["--live"])
    monkeypatch.setattr(evaluate, "_clients", lambda **kw: (None, None, "synthetic", "synthetic"))
    monkeypatch.setattr(evaluate, "build_pipeline", lambda **kw: None)
    monkeypatch.setattr(evaluate, "measure_pipeline_agreement", lambda **kw: _empty_agreement())
    first, _, _ = evaluate._run_measurement_two(cases=(), args=args, settings=db_settings)
    with connect(settings=db_settings) as other:
        row = other.execute(
            "SELECT order_no, product_name FROM orders ORDER BY order_no LIMIT 1"
        ).fetchone()
        assert row is not None
        other.execute(
            "UPDATE orders SET product_name = %s WHERE order_no = %s",
            ("지문 대조", row["order_no"]),
        )
    try:
        second, _, _ = evaluate._run_measurement_two(cases=(), args=args, settings=db_settings)
        assert (
            first.input_fingerprint["order_data_version"]
            != second.input_fingerprint["order_data_version"]
        )
        assert (
            first.input_fingerprint["policy_data_version"]
            == second.input_fingerprint["policy_data_version"]
        )
    finally:
        with connect(settings=db_settings) as other:
            other.execute(
                "UPDATE orders SET product_name = %s WHERE order_no = %s",
                (row["product_name"], row["order_no"]),
            )


@pytest.mark.db
def test_정책_지문은_적재시각을_무시하고_본문과_벡터_변경은_잡는다(
    app_conn: psycopg.Connection[Any],
) -> None:
    vector = "[" + ",".join(["1"] + ["0"] * 1535) + "]"
    app_conn.execute(
        "INSERT INTO policy_chunks(evidence_id, document_slug, document_title, article, content, embedding, embedding_model, embedding_dimensions) VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s)",
        ("policy:t7:1", "t7", "시험", "1", "본문", vector, "model", 1536),
    )
    first = evaluate._lock_evaluation_inputs(app_conn, timeout_ms=100)
    app_conn.execute(
        "UPDATE policy_chunks SET created_at = created_at + interval '1 day' WHERE evidence_id = 'policy:t7:1'"
    )
    same = evaluate._lock_evaluation_inputs(app_conn, timeout_ms=100)
    assert same == first
    app_conn.execute(
        "UPDATE policy_chunks SET content = '바뀐 본문' WHERE evidence_id = 'policy:t7:1'"
    )
    changed = evaluate._lock_evaluation_inputs(app_conn, timeout_ms=100)
    assert changed["policy_data_version"] != first["policy_data_version"]
    vector = "[" + ",".join(["0", "1"] + ["0"] * 1534) + "]"
    app_conn.execute(
        "UPDATE policy_chunks SET embedding = %s::vector WHERE evidence_id = 'policy:t7:1'",
        (vector,),
    )
    changed_vector = evaluate._lock_evaluation_inputs(app_conn, timeout_ms=100)
    assert changed_vector["policy_data_version"] != changed["policy_data_version"]
    assert changed_vector["order_data_version"] == first["order_data_version"]


@pytest.mark.db
def test_입력_잠금_실패는_모델호출_전_유한하게_끝내고_연결을_정리한다(
    seeded_order_count: int, db_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    del seeded_order_count
    args = evaluate.build_parser().parse_args(["--live"])
    monkeypatch.setattr(evaluate, "_clients", lambda **kw: (None, None, "synthetic", "synthetic"))
    monkeypatch.setattr(evaluate, "build_pipeline", lambda **kw: None)

    def no_measure(**kwargs: Any) -> Any:
        pytest.fail("입력 잠금 실패 후 측정에 진입했다")

    monkeypatch.setattr(evaluate, "measure_pipeline_agreement", no_measure)
    with connect(settings=db_settings) as other:
        try:
            other.execute("LOCK TABLE orders IN ROW EXCLUSIVE MODE")
            with pytest.raises(psycopg.errors.LockNotAvailable):
                evaluate._run_measurement_two(
                    cases=(),
                    args=args,
                    settings=db_settings.model_copy(update={"sql_statement_timeout_ms": 100}),
                )
        finally:
            other.rollback()
    with connect(settings=db_settings) as other:
        other.execute("LOCK TABLE orders, policy_chunks IN ACCESS EXCLUSIVE MODE NOWAIT")
        other.rollback()


@pytest.mark.db
def test_실행_확인입력과_전체_픽스처_지문은_한번_읽은_내용에서_함께_온다(
    seeded_order_count: int, db_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from reply_gate.evaluation import load_golden_set

    del seeded_order_count
    args = evaluate.build_parser().parse_args(["--live"])
    reads: list[Path] = []
    original = Path.read_text

    def read(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "orders.jsonl":
            reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(evaluate, "_clients", lambda **kw: (None, None, "synthetic", "synthetic"))
    monkeypatch.setattr(evaluate, "build_pipeline", lambda **kw: None)
    seen: list[dict[str, str | None]] = []

    def measure(**kwargs: Any) -> Any:
        seen.append(kwargs["phone_last4_by_order"])
        return _empty_agreement()

    monkeypatch.setattr(evaluate, "measure_pipeline_agreement", measure)
    result, _, _ = evaluate._run_measurement_two(
        cases=load_golden_set(), args=args, settings=db_settings
    )
    assert len(reads) == 1
    assert seen and len(seen[0]) == 500
    assert result.input_fingerprint["order_data_version"].startswith("orders-db-")
