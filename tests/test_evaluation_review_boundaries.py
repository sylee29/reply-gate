"""독립 검토 T7-R1~R8의 재현. 실제 제공자 호출 없이 경계만 검증한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from scripts import evaluate, generate_blind_rewrites

from reply_gate import draft, evaluation, llm, regression_guard, retrieval_eval
from reply_gate.retrieval_strategies import default_strategy_ladder
from tests.conftest import declared_settings
from tests.test_regression_guard import _promotion
from tests.test_retrieval_eval import _FixedEmbedder


@pytest.mark.parametrize(
    "stems",
    [("one", "./one", "././one"), ("one", "nested/../one", "two"), ("one", "../outside", "two")],
)
def test_승격_스템은_경로_별칭을_실측_회차로_인정하지_않는다(
    tmp_path: Path, stems: tuple[str, ...]
) -> None:
    reference = tmp_path / "reference.json"
    _promotion(reference, stems=stems)
    assert isinstance(
        regression_guard.load_promoted_baseline(reference), regression_guard.BaselineNotRegistered
    )


def test_정상_서로다른_승격_스템은_계속_인정한다(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    _promotion(reference, stems=["one", "two", "three"])
    assert isinstance(
        regression_guard.load_promoted_baseline(reference), regression_guard.PromotedBaseline
    )


@pytest.mark.parametrize(
    "field", ["fusion_pool_size", "rerank_top_n", "top_k", "dimensions", "ngram_size", "rrf_k"]
)
@pytest.mark.parametrize("value", [True, float("nan"), 1.5, 0, -1])
def test_정수_전략_설정은_비용_전에_bool과_비정수를_거부한다(
    tmp_path: Path, field: str, value: Any
) -> None:
    # 공개 함수 직접 호출의 타입 힌트 밖 값이다. CLI int parser 재현으로 주장하지 않는다.
    embedder = _FixedEmbedder({"정책": [1.0, 0.0], "문의": [0.0, 1.0]})
    config_args: dict[str, Any] = {"model": "fixed", "dimensions": 2, "top_k": 5}
    cutoff_args: dict[str, Any] = {}
    options: dict[str, Any] = {}
    if field in config_args:
        config_args[field] = value
    elif field in ("fusion_pool_size", "rerank_top_n"):
        cutoff_args[field] = value
    else:
        options[field] = value
    with pytest.raises(ValueError):
        retrieval_eval.retrieve_strategy_ladder(
            queries=[retrieval_eval.RetrievalQuery("A", "문의")],
            policy_texts=[("p", "정책")],
            rewritten_queries={"A": "문의"},
            embedder=embedder,
            embedding_config=retrieval_eval.RetrievalEvalConfig(**config_args),
            cutoffs=retrieval_eval.StrategyCutoffs(**cutoff_args),
            reranker=None,
            rerank_model="",
            cache_dir=tmp_path / "v",
            rerank_cache_dir=tmp_path / "r",
            strategies=default_strategy_ladder()[:3],
            **options,
        )
    assert embedder.calls == []


def test_실제_SQL_행수상한의_변경은_프롬프트_지문도_바꾼다() -> None:
    args = evaluate.build_parser().parse_args([])
    original = declared_settings(sql_max_rows=50)
    changed = declared_settings(sql_max_rows=1)
    before = evaluate._condition_fingerprint(args=args, settings=original, run_settings=original)
    after = evaluate._condition_fingerprint(args=args, settings=original, run_settings=changed)
    assert before["sql_prompt_version"] != after["sql_prompt_version"]
    assert before == evaluate._condition_fingerprint(
        args=args, settings=original, run_settings=original
    )
    assert {k for k in before if before[k] != after[k]} == {"sql_prompt_version"}


@pytest.mark.parametrize("failure", ["write_and_close", "flush_and_close", "close"])
def test_blind_출력_close_실패도_불완전_파일을_정리한다(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    output = tmp_path / "output.jsonl"
    real_open = Path.open

    class BrokenOutput:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __getattr__(self, name: str) -> Any:
            return getattr(self.handle, name)

        def write(self, value: str) -> int:
            if failure == "write_and_close":
                raise OSError("synthetic write failure")
            return self.handle.write(value)  # type: ignore[no-any-return]

        def flush(self) -> None:
            if failure == "flush_and_close":
                raise OSError("synthetic flush failure")
            self.handle.flush()

        def close(self) -> None:
            self.handle.close()
            raise OSError("synthetic close failure")

    def open_file(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        return BrokenOutput(handle) if self == output else handle

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(
        generate_blind_rewrites,
        "get_settings",
        lambda: declared_settings(openai_api_key=SecretStr("synthetic")),
    )
    monkeypatch.setattr(generate_blind_rewrites, "OpenAIGenerationClient", lambda **kw: object())
    monkeypatch.setattr(generate_blind_rewrites, "rewrite_one", lambda **kw: "질의")
    with pytest.raises(OSError):
        generate_blind_rewrites.main(["--live", "--out", str(output)])
    assert not output.exists()
    assert "생성 완료" not in capsys.readouterr().out


def test_blind_실패_정리는_예약후_생긴_다른_파일을_지우지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.jsonl"
    monkeypatch.setattr(
        generate_blind_rewrites,
        "get_settings",
        lambda: declared_settings(openai_api_key=SecretStr("synthetic")),
    )
    monkeypatch.setattr(generate_blind_rewrites, "OpenAIGenerationClient", lambda **kw: object())

    def fail(**kwargs: Any) -> str:
        replacement = tmp_path / "replacement"
        replacement.write_text("다른 작성자")
        replacement.replace(output)
        raise llm.LLMFormatError(stage="rewrite", detail="synthetic")

    monkeypatch.setattr(generate_blind_rewrites, "rewrite_one", fail)
    assert generate_blind_rewrites.main(["--live", "--out", str(output)]) == 1
    assert output.read_text() == "다른 작성자"


def test_초안_지문은_실제_소비자에게_재바인딩된_스키마를_읽는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = evaluation.code_condition_fingerprint()["draft_prompt_version"]
    monkeypatch.setattr(
        draft,
        "DRAFT_JSON_SCHEMA",
        {"type": "object", "properties": {"different": {"type": "string"}}},
    )
    assert evaluation.code_condition_fingerprint()["draft_prompt_version"] != before


@pytest.mark.parametrize(
    "argv", [["--top-k", "0"], ["--similarity-threshold", "nan"], ["--similarity-threshold", "2"]]
)
def test_평가_CLI_덮어쓰기는_설정_검증을_다시_통과해야_한다(argv: list[str]) -> None:
    args = evaluate.build_parser().parse_args(argv)
    with pytest.raises(ValueError):
        evaluate._run_settings(args=args, settings=declared_settings())


@pytest.mark.parametrize(
    "argv", [["--top-k", "0"], ["--similarity-threshold", "nan"], ["--similarity-threshold", "2"]]
)
def test_평가_CLI_설정_오류는_연결과_모델_호출에_도달하지_않는다(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(evaluate, "get_settings", lambda: declared_settings())

    def no_preflight(**kwargs: Any) -> str:
        calls.append("db_preflight")
        return "not requested"

    def no_clients(**kwargs: Any) -> Any:
        calls.append("generation_rewrite_embedding")
        pytest.fail("잘못된 CLI 설정이 모델 조립까지 왔다")

    monkeypatch.setattr(evaluate, "_skip_reason", no_preflight)
    monkeypatch.setattr(evaluate, "_clients", no_clients)
    with pytest.raises(SystemExit):
        evaluate.main([*argv, "--out-dir", str(tmp_path)])
    assert calls == []


def test_SQL_프롬프트의_최초와_재시도_모두_실제_행수상한을_쓴다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reply_gate import evidence

    original = evidence.build_sql_user_prompt
    seen: list[int] = []

    def build(**kwargs: Any) -> str:
        seen.append(kwargs["max_rows"])
        return original(**kwargs)

    monkeypatch.setattr(evidence, "build_sql_user_prompt", build)
    evaluation.code_condition_fingerprint(sql_max_rows=1)
    assert seen == [1, 1]


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_서로다른_승격_스템도_같은_실제파일을_중복_집계하지_않는다(
    tmp_path: Path, alias: str
) -> None:
    from tests.test_regression_guard import _healthy_cases, _payload

    payload = _payload(started_at="2026-01-01", cases=_healthy_cases())
    first = tmp_path / "one.json"
    first.write_text(json.dumps(payload))
    for stem in ("two", "three"):
        path = tmp_path / f"{stem}.json"
        if alias == "symlink":
            path.symlink_to(first)
        else:
            path.hardlink_to(first)
    reference = tmp_path / "reference.json"
    _promotion(reference, stems=["one", "two", "three"])
    candidate = regression_guard.RunSet(
        label="candidate",
        runs=tuple(
            regression_guard.run_summary_from_payload(payload, stem=f"c{i}", source="test")
            for i in range(3)
        ),
    )
    line = regression_guard._binding_line(
        promotion=regression_guard.load_promoted_baseline(reference),
        candidate=candidate,
        reports_dir=tmp_path,
        run_set_size=3,
    )
    assert line.verdict == "대조 불가"
    assert "같은 실제 파일" in line.verdict_reason


def test_실제파일_세개의_승격_정족수는_그대로_인정한다(tmp_path: Path) -> None:
    from tests.test_regression_guard import _healthy_cases, _payload

    payload = _payload(started_at="2026-01-01", cases=_healthy_cases())
    for stem in ("one", "two", "three"):
        (tmp_path / f"{stem}.json").write_text(json.dumps(payload))
    reference = tmp_path / "reference.json"
    _promotion(reference, stems=["one", "two", "three"])
    candidate = regression_guard.RunSet(
        label="candidate",
        runs=tuple(
            regression_guard.run_summary_from_payload(payload, stem=f"c{i}", source="test")
            for i in range(3)
        ),
    )
    line = regression_guard._binding_line(
        promotion=regression_guard.load_promoted_baseline(reference),
        candidate=candidate,
        reports_dir=tmp_path,
        run_set_size=3,
    )
    assert line.verdict == "통과"
