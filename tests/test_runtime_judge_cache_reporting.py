"""런타임 L2 캐시 관측이 평가 산출까지 보존되는지 검사한다. 모델 호출은 없다."""

from dataclasses import replace
from typing import cast

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.evaluation import measure_pipeline_agreement, render_markdown, report_to_json
from tests.test_evaluation import GOLDEN, ScriptedPipeline, _processed, _report


@pytest.mark.parametrize("creation,read", [(None, None), (0, 0), (11, None), (None, 13), (11, 13)])
def test_런타임_판정_캐시를_JSON과_문면에_같은_값으로_보존한다(
    creation: int | None,
    read: int | None,
) -> None:
    """SDK까지 읽은 캐시 값이 파이프라인 결과에서 평가 직렬화 전에 유실됐었다."""
    plain = _processed(judge_input_tokens=40, judge_output_tokens=20)
    measured = replace(plain, judge_cache_creation_tokens=creation, judge_cache_read_tokens=read)
    agreement = measure_pipeline_agreement(
        cases=GOLDEN[:2],
        pipeline=ScriptedPipeline([measured, plain]),
        app_conn=cast(psycopg.Connection[DictRow], None),
        readonly_conn=cast(psycopg.Connection[DictRow], None),
    )
    report = _report(pipeline=agreement)
    payload = report_to_json(report)["measurement_2_pipeline_agreement"]
    tokens = payload["tokens"]
    assert tokens["judge_cache_creation_total"] == creation
    assert tokens["judge_cache_read_total"] == read
    assert payload["outcomes"][0]["judge_cache_creation_tokens"] == creation
    assert payload["outcomes"][0]["judge_cache_read_tokens"] == read
    assert payload["outcomes"][1]["judge_cache_creation_tokens"] is None
    assert payload["outcomes"][1]["judge_cache_read_tokens"] is None
    line = next(
        line
        for line in render_markdown(report).splitlines()
        if line.startswith("- 판정 계열 프롬프트 캐시")
    )
    if creation is None and read is None:
        assert "미측정" in line
    else:
        assert f"write {'미측정' if creation is None else format(creation, ',')}" in line
        assert f"read {'미측정' if read is None else format(read, ',')}" in line
        assert "별도" in line
    # 기존 네 계열 소계 정의를 유지한다. 판정 캐시는 별도 관측값이고 소계에 몰래 더하지 않는다.
    assert tokens["judge_input_total"] == 80
    assert tokens["judge_output_total"] == 40
    assert tokens["total_per_inquiry"] == 60
