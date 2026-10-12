"""재감사에서 재현한 평가/검색 경계를 실제 공개 호출로 검사한다."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from reply_gate import llm, retrieval_eval, retrieval_strategies
from reply_gate.contracts import EscalationReason, InquiryStatus, Verdict
from reply_gate.evaluation import measure_pipeline_agreement
from reply_gate.regression_guard import (
    BaselineNotRegistered,
    RunSet,
    compare_run_sets,
    load_promoted_baseline,
    run_summary_from_payload,
)
from reply_gate.retrieval_eval import RetrievalEvalConfig, RetrievalQuery, StrategyCutoffs
from reply_gate.retrieval_strategies import RetrievalStage, StrategyDefinition
from tests.test_evaluation import _NO_CONN, ScriptedPipeline, _bait_case, _processed
from tests.test_regression_guard import _healthy_cases, _payload, _promotion
from tests.test_retrieval_eval import _FixedEmbedder


@pytest.mark.parametrize(
    "vector", [[float("nan"), 1.0], [float("inf"), 1.0], [0.0, 0.0], [True, 1.0]]
)
def test_손상된_수치_캐시는_미스로_돌린다(tmp_path: Path, vector: list[float]) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"model": "m", "dimensions": 2, "text_hash": "h", "vector": vector}))
    assert (
        retrieval_eval._read_cached_vector(path=path, model="m", dimensions=2, text_hash="h")
        is None
    )


@pytest.mark.parametrize("kind", ["embedding", "rerank"])
def test_잘못된_UTF8_캐시는_미스로_돌린다(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "cache.json"
    path.write_bytes(b"\xff")
    if kind == "embedding":
        result: object = retrieval_eval._read_cached_vector(
            path=path, model="m", dimensions=2, text_hash="h"
        )
    else:
        result = retrieval_strategies._read_rerank_cache(
            path=path, model="m", input_hash="h", expected=("a",)
        )
    assert result is None


@pytest.mark.parametrize("kind", ["embedding", "rerank"])
def test_캐시_교체_실패에도_자기_임시파일을_정리한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    path = tmp_path / "cache.json"
    path.write_text("old")

    def fail(self: Path, target: Path) -> Path:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        if kind == "embedding":
            retrieval_eval._write_cached_vector(
                path=path, model="m", dimensions=2, text_hash="h", vector=[1.0, 0.0]
            )
        else:
            retrieval_strategies._write_rerank_cache(
                path=path, model="m", input_hash="h", evidence_ids=["a"]
            )
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("rrf_k,ngram", [(0, 2), (-1, 2), (60, 0), (60, -1)])
def test_잘못된_전략_설정은_임베딩_전_거부한다(tmp_path: Path, rrf_k: int, ngram: int) -> None:
    embedder = _FixedEmbedder({"정책": [1.0, 0.0], "문의": [0.0, 1.0]})
    with pytest.raises(ValueError):
        retrieval_eval.retrieve_strategy_ladder(
            queries=[RetrievalQuery("A", "문의")],
            policy_texts=[("p", "정책")],
            rewritten_queries={"A": "문의"},
            embedder=embedder,
            embedding_config=RetrievalEvalConfig(model="fixed", dimensions=2),
            cutoffs=StrategyCutoffs(),
            reranker=None,
            rerank_model="",
            cache_dir=tmp_path / "v",
            rerank_cache_dir=tmp_path / "r",
            rrf_k=rrf_k,
            ngram_size=ngram,
            strategies=[
                StrategyDefinition(
                    name="hybrid",
                    stages=(RetrievalStage.VECTOR, RetrievalStage.REWRITE, RetrievalStage.HYBRID),
                )
            ],
        )
    assert embedder.calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_RRF_컷은_유한해야_한다(value: float) -> None:
    with pytest.raises(ValueError):
        StrategyCutoffs(rrf_score=value)


def test_승격_참조의_중복_회차는_정족수를_채우지_않는다(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    _promotion(path, stems=["evaluation-live-l2-1"] * 3)
    result = load_promoted_baseline(path)
    assert isinstance(result, BaselineNotRegistered)
    assert "중복" in result.reason


def _set(matches: list[bool], *, omit: bool = False, drift: bool = False) -> RunSet:
    runs = []
    for i, matched in enumerate(matches):
        cases = _healthy_cases()
        cases["G04"] = (matched, None, None)
        if omit and i == 0:
            cases.pop("G01")
        runs.append(
            run_summary_from_payload(
                _payload(
                    started_at=str(i), cases=cases, detection_rate=0.5 if drift and i == 0 else 1.0
                ),
                stem=f"run-{i}",
                source="synthetic",
            )
        )
    return RunSet(label="synthetic", runs=tuple(runs))


@pytest.mark.parametrize(
    "before,after,warning",
    [
        ([True] * 3, [True], False),
        ([True, True, False], [True], False),
        ([True, False], [True, False, False], True),
    ],
)
def test_감소_경보는_횟수_대신_일치_비율을_비교한다(
    before: list[bool], after: list[bool], warning: bool
) -> None:
    line = compare_run_sets(
        baseline=_set(before),
        candidate=_set(after),
        label="test",
        role="경보",
        expected_run_count=1,
    )
    assert bool(line.match_decreases) is warning


def test_한_회차의_케이스가_빠져도_조용히_교집합만_통과시키지_않는다() -> None:
    line = compare_run_sets(
        baseline=_set([True] * 3), candidate=_set([True] * 3, omit=True), label="test", role="구속"
    )
    assert line.verdict == "보류"
    assert "G01" in line.verdict_reason


def test_같은_세트의_L1_흔들림은_미상으로_숨기지_않는다() -> None:
    line = compare_run_sets(
        baseline=_set([True] * 3), candidate=_set([True] * 3, drift=True), label="test", role="구속"
    )
    assert line.verdict != "통과"
    assert any("세트" in item for item in line.measurement1_changes)


@pytest.mark.parametrize("stage", ["intent", "draft"])
def test_판정_전_기술적_실패는_미끼_재현율에서_미측정이다(stage: str) -> None:
    processed = _processed(
        status=InquiryStatus.ESCALATED,
        escalation=EscalationReason.LLM_CALL_FAILED,
        failed_stage=stage,
        attempts=(),
    )
    result = measure_pipeline_agreement(
        cases=[_bait_case()],
        pipeline=ScriptedPipeline([processed]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    assert (result.bait_total, result.bait_unmeasured, result.bait_reject_recall) == (0, 1, None)


def test_blind_출력_부모가_없으면_모델을_부르기_전_거부한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import generate_blind_rewrites as script

    from tests.conftest import declared_settings

    calls: list[str] = []
    monkeypatch.setattr(
        script, "get_settings", lambda: declared_settings(openai_api_key=SecretStr("synthetic-key"))
    )
    monkeypatch.setattr(script, "OpenAIGenerationClient", lambda **kw: object())

    def rewrite(**kwargs: Any) -> str:
        calls.append("called")
        return "질의"

    monkeypatch.setattr(script, "rewrite_one", rewrite)
    assert script.main(["--live", "--out", str(tmp_path / "missing" / "output.jsonl")]) == 2
    assert calls == []


def test_blind_출력은_과금_전에_독점_예약하고_실패시_정리한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import generate_blind_rewrites as script

    from tests.conftest import declared_settings

    output = tmp_path / "output.jsonl"
    observed: list[bool] = []
    monkeypatch.setattr(
        script, "get_settings", lambda: declared_settings(openai_api_key=SecretStr("synthetic-key"))
    )
    monkeypatch.setattr(script, "OpenAIGenerationClient", lambda **kw: object())

    def fail(**kwargs: Any) -> str:
        observed.append(output.exists())
        raise llm.LLMFormatError(stage="rewrite", detail="synthetic")

    monkeypatch.setattr(script, "rewrite_one", fail)
    assert script.main(["--live", "--out", str(output)]) == 1
    assert observed == [True]
    assert not output.exists()


@pytest.mark.parametrize(
    "module,constant,field",
    [
        ("evidence", "INTENT_SYSTEM_PROMPT", "intent_prompt_version"),
        ("evidence", "SQL_SYSTEM_PROMPT", "sql_prompt_version"),
        ("draft", "DRAFT_SYSTEM_PROMPT", "draft_prompt_version"),
        ("query_rewrite", "REWRITE_SYSTEM_PROMPT", "rewrite_prompt_version"),
    ],
)
def test_실제로_쓰는_네_생성_프롬프트_변경은_지문을_바꾼다(
    monkeypatch: pytest.MonkeyPatch, module: str, constant: str, field: str
) -> None:
    from importlib import import_module

    from reply_gate.evaluation import code_condition_fingerprint

    target = import_module("reply_gate." + module)
    before = code_condition_fingerprint()
    monkeypatch.setattr(target, constant, getattr(target, constant) + "\n새 규칙")
    after = code_condition_fingerprint()
    assert field in before
    assert before[field] != after[field]


def test_별도_골든셋을_귀인_라벨_검증까지_전달한다(tmp_path: Path) -> None:
    from reply_gate.evaluation import (
        DEFAULT_GOLDEN_SET_PATH,
        SkippedMeasurement,
        build_report,
        load_golden_set,
        measure_gate_accuracy,
    )
    from tests.test_evaluation import FIXTURES, _conditions

    row = json.loads(DEFAULT_GOLDEN_SET_PATH.read_text().splitlines()[0])
    row["id"] = "CUSTOM"
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": "CUSTOM", "relevant_evidence_ids": [], "note": ""}) + "\n")
    agreement = measure_pipeline_agreement(
        cases=load_golden_set(golden),
        pipeline=ScriptedPipeline([_processed()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=agreement,
        judge_accuracy=SkippedMeasurement(reason="not requested"),
        retrieval_labels_path=labels,
        golden_set_path=golden,
    )
    assert getattr(report.failure_attribution, "reason", None) is None


def test_근거없음_업무판정은_기술적_미측정으로_바뀌지_않는다() -> None:
    processed = _processed(
        status=InquiryStatus.ESCALATED, escalation=EscalationReason.NO_EVIDENCE, attempts=()
    )
    result = measure_pipeline_agreement(
        cases=[_bait_case()],
        pipeline=ScriptedPipeline([processed]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    assert (result.bait_total, result.bait_unmeasured, result.bait_reject_recall) == (1, 0, 0.0)


def test_초안_재생성_실패도_앞서_관측한_기각을_지우지_않는다() -> None:
    from reply_gate.pipeline import AttemptRecord

    processed = _processed(
        status=InquiryStatus.ESCALATED,
        escalation=EscalationReason.LLM_CALL_FAILED,
        failed_stage="draft",
        attempts=(
            AttemptRecord(attempt_no=1, verdict=Verdict.REJECT, reject_reasons=(), draft={}),
        ),
    )
    result = measure_pipeline_agreement(
        cases=[_bait_case()],
        pipeline=ScriptedPipeline([processed]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    assert (result.bait_total, result.bait_unmeasured, result.bait_reject_recall) == (1, 0, 1.0)


@pytest.mark.parametrize("kind", ["embedding", "rerank"])
def test_동시_캐시_작성자는_임시파일을_공유하지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "cache.json"
    rendezvous = Barrier(2, timeout=5)
    original = Path.replace
    names: list[str] = []

    def replace_file(self: Path, target: Path) -> Path:
        names.append(self.name)
        rendezvous.wait()
        return original(self, target)

    monkeypatch.setattr(Path, "replace", replace_file)

    def write(index: int) -> None:
        if kind == "embedding":
            retrieval_eval._write_cached_vector(
                path=path, model="m", dimensions=2, text_hash="h", vector=[float(index), 1.0]
            )
        else:
            retrieval_strategies._write_rerank_cache(
                path=path, model="m", input_hash="h", evidence_ids=[str(index)]
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, (1, 2)))
    assert len(set(names)) == 2
    assert json.loads(path.read_text())["model"] == "m"
    assert list(tmp_path.iterdir()) == [path]


def test_임베딩_배치_후반_오류도_앞부분_캐시를_남기지_않는다(tmp_path: Path) -> None:
    embedder = _FixedEmbedder({"정상": [1.0, 0.0], "잘못됨": [0.0, 0.0]})
    with pytest.raises(ValueError):
        retrieval_eval._embed_with_cache(
            texts=["정상", "잘못됨"],
            stage="test",
            embedder=embedder,
            config=RetrievalEvalConfig(model="fixed", dimensions=2),
            cache_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("values,expected", [([], None), ([5, 2, 3], 3), ([5.5, 2.5, 3.5], 3.5)])
def test_공통_백분위는_빈값과_숫자_형식을_보존한다(values: list[Any], expected: Any) -> None:
    from reply_gate.evaluation import _percentile, _percentile_ms

    actual = _percentile(values, 50)
    assert actual == expected
    assert type(actual) is type(expected)
    assert _percentile_ms(values, 50) == expected


def test_float_범위를_넘는_캐시_정수는_재계산_미스다(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"model": "m", "dimensions": 2, "text_hash": "h", "vector": [10**400, 1.0]})
    )
    assert (
        retrieval_eval._read_cached_vector(path=path, model="m", dimensions=2, text_hash="h")
        is None
    )


def test_평가에_고정한_전화뒷자리_입력은_파일을_다시_읽지_않는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reply_gate import evaluation
    from tests.test_evaluation import _case

    calls: list[object] = []

    class Pipeline:
        def run(self, **kwargs: Any) -> Any:
            calls.append(kwargs["phone_last4"])
            return _processed()

    def forbidden(order_no: str | None) -> str | None:
        pytest.fail("고정한 실행 입력 대신 파일을 다시 읽었다")

    monkeypatch.setattr(evaluation, "fixture_phone_last4", forbidden)
    case = replace(_case("CUSTOM"), order_no="ORD-20260101-0001")
    measure_pipeline_agreement(
        cases=[case],
        pipeline=Pipeline(),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
        phone_last4_by_order={"ORD-20260101-0001": "1234"},
    )
    assert calls == ["1234"]


def test_blind_출력_선검사_뒤_생긴_파일도_덮어쓰지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import generate_blind_rewrites as script

    from tests.conftest import declared_settings

    output = tmp_path / "output.jsonl"

    def settings() -> Any:
        output.write_text("다른 작성자의 산출")
        return declared_settings(openai_api_key=SecretStr("synthetic-key"))

    monkeypatch.setattr(script, "get_settings", settings)
    monkeypatch.setattr(
        script, "OpenAIGenerationClient", lambda **kw: pytest.fail("출력 충돌 뒤 모델 조립")
    )
    assert script.main(["--live", "--out", str(output)]) == 2
    assert output.read_text() == "다른 작성자의 산출"


def test_확인입력_판은_전화뒷자리_맵만이_아닌_전체_픽스처를_담는다(tmp_path: Path) -> None:
    from reply_gate.evaluation import load_fixture_phone_inputs

    path = tmp_path / "orders.jsonl"
    row = {
        "order_no": "ORD-20260101-0001",
        "customer_phone": "010-1234-5678",
        "product_name": "원본",
    }
    path.write_text(json.dumps(row))
    original, version = load_fixture_phone_inputs(path)
    row["product_name"] = "변경"
    path.write_text(json.dumps(row))
    changed, new_version = load_fixture_phone_inputs(path)
    assert original == changed == {"ORD-20260101-0001": "5678"}
    assert version != new_version
