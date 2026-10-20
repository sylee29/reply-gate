"""제공자 응답 수용 규칙도 비교 조건이다. 고정 탐침은 평가 정답이나 과금 호출이 아니다."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest

from reply_gate import evaluation, llm
from reply_gate.regression_guard import FINGERPRINT_FIELDS, ConditionFingerprint


def test_생성_응답_규칙은_별도_지문이고_과거의_값은_미상이다() -> None:
    """완료 여부가 달라도 같은 JSON이던 실행을 프롬프트 동일성으로 묶지 않는다."""
    assert "generation_rule_version" in FINGERPRINT_FIELDS
    value = evaluation.code_condition_fingerprint()["generation_rule_version"]
    assert value.startswith("generationrules-")
    assert ConditionFingerprint.from_values({}).values["generation_rule_version"] is None


@pytest.mark.parametrize(
    "helper,field",
    [
        ("_openai_response_text", "generation_rule_version"),
        ("_anthropic_response_text", "judge_rule_version"),
    ],
)
def test_미완료와_복수_텍스트를_허용하면_해당_응답_규칙_지문이_바뀐다(
    monkeypatch: pytest.MonkeyPatch,
    helper: str,
    field: str,
) -> None:
    """규칙이 겨냥하는 실패 모드를 실제 helper 교체로 만들면 비교 조건이 달라져야 한다."""
    before = evaluation.code_condition_fingerprint()[field]
    monkeypatch.setattr(llm, helper, lambda response: '{"ok":true}')
    assert evaluation.code_condition_fingerprint()[field] != before


def test_OpenAI의_독립_reasoning_항목을_거부하면_지문이_바뀐다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """독립 리뷰 음성 대조: Anthropic식 thinking을 OpenAI 정상 대조로 대신하지 않는다."""
    before = evaluation.code_condition_fingerprint()["generation_rule_version"]
    original = llm._openai_response_text

    def reject_reasoning(response: object) -> str:
        if any(
            getattr(item, "type", None) == "reasoning" for item in getattr(response, "output", [])
        ):
            raise ValueError("정상 reasoning을 잘못 거부")
        return original(response)

    monkeypatch.setattr(llm, "_openai_response_text", reject_reasoning)
    assert evaluation.code_condition_fingerprint()["generation_rule_version"] != before


def test_알려진_종료상태_하나만_추가_허용해도_해당_지문이_바뀐다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """독립 리뷰 음성 대조: unknown 거부만으로 실제 종료 상태 전체를 대표하지 않는다."""
    providers = (
        (
            "_openai_response_text",
            "generation_rule_version",
            "status",
            "completed",
            ("failed", "in_progress", "cancelled", "queued", "incomplete"),
        ),
        (
            "_anthropic_response_text",
            "judge_rule_version",
            "stop_reason",
            "end_turn",
            (
                "max_tokens",
                "stop_sequence",
                "tool_use",
                "pause_turn",
                "refusal",
                "model_context_window_exceeded",
            ),
        ),
    )
    for helper, field, state_field, normal, states in providers:
        original = cast(Callable[[object], str], getattr(llm, helper))
        before = evaluation.code_condition_fingerprint()[field]
        for state in states:

            def allow_one(
                response: object,
                parser: Callable[[object], str] = original,
                state_field: str = state_field,
                state: str = state,
                normal: str = normal,
            ) -> str:
                if getattr(response, state_field, None) == state:
                    response = SimpleNamespace(**(vars(response) | {state_field: normal}))
                return parser(response)

            with monkeypatch.context() as patch:
                patch.setattr(llm, helper, allow_one)
                assert evaluation.code_condition_fingerprint()[field] != before, (helper, state)


@pytest.mark.parametrize(
    "helper,field",
    [
        ("_openai_response_text", "generation_rule_version"),
        ("_anthropic_response_text", "judge_rule_version"),
    ],
)
def test_정상_완료도_모두_거부하면_응답_규칙_지문이_바뀐다(
    monkeypatch: pytest.MonkeyPatch,
    helper: str,
    field: str,
) -> None:
    """거부 사례만 모으면 정상 JSON을 전부 버리는 구현과 구분하지 못한다."""
    before = evaluation.code_condition_fingerprint()[field]

    def reject(response: Any) -> str:
        raise ValueError("진단 문면은 지문이 아니다")

    monkeypatch.setattr(llm, helper, reject)
    assert evaluation.code_condition_fingerprint()[field] != before
