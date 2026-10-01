"""생성·임베딩 래퍼의 실패 정책 단위 테스트 (외부 호출 없음, 목 사용)."""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import openai
import pytest
from pydantic import SecretStr

from reply_gate.draft import DraftGenerator
from reply_gate.llm import (
    MAX_ATTEMPTS,
    AnthropicGenerationClient,
    BgeM3EmbeddingClient,
    LLMCallError,
    LLMFormatError,
    OpenAIEmbeddingClient,
    OpenAIGenerationClient,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"intent": {"type": "string"}},
    "required": ["intent"],
    "additionalProperties": False,
}


def _response(text: str, *, refusal: str | None = None) -> SimpleNamespace:
    parts = [SimpleNamespace(type="output_text", text=text)]
    if refusal is not None:
        parts = [SimpleNamespace(type="refusal", refusal=refusal)]
    return SimpleNamespace(
        output=[SimpleNamespace(content=parts)],
        output_text=text,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


class _RecordingCalls:
    """SDK 엔드포인트 대역 — 호출 인자를 모으고 정해진 결과를 순서대로 돌려준다."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _generation_client(outcomes: list[Any]) -> tuple[OpenAIGenerationClient, _RecordingCalls]:
    responses = _RecordingCalls(outcomes)
    fake_sdk = cast(openai.OpenAI, SimpleNamespace(responses=responses))
    client = OpenAIGenerationClient(
        api_key=SecretStr("test"), model="gpt-5.6-terra", client=fake_sdk
    )
    return client, responses


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))


def _status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status_code, request=request, json={"error": {"message": "x"}})
    return openai.APIStatusError("boom", response=response, body=None)


def test_sdk_자동재시도를_끈다() -> None:
    """래퍼 재시도와 SDK 재시도가 중첩되면 docs/standards.md "재시도 상한"이 깨진다."""
    client = OpenAIGenerationClient(api_key=SecretStr("test"), model="gpt-5.6-terra")
    assert client._client.max_retries == 0

    embedder = OpenAIEmbeddingClient(
        api_key=SecretStr("test"), model="text-embedding-3-small", dimensions=1536
    )
    assert embedder._client.max_retries == 0


# ── 주입 경로도 같은 정책을 받는다 (`or` 우변은 주입 시 평가되지 않는다) ──────


def test_주입한_SDK_클라이언트도_자동재시도가_꺼지고_타임아웃이_고정된다() -> None:
    """구 코드는 `client or SDK(..., max_retries=0, timeout=...)` 라 주입 시 우변을 평가조차
    하지 않았다. SDK 기본 `max_retries=2` 와 래퍼 재시도가 곱해져 **최대 6회 전송**인데
    `LLMCallError.attempts` 는 2 로 신고됐고, 타임아웃도 120초 의도가 SDK 기본 600초였다.
    """
    injected_openai = openai.OpenAI(api_key="test")
    injected_anthropic = anthropic.Anthropic(api_key="test")
    assert injected_openai.max_retries == 2  # 주입 전에는 SDK 기본값이다
    assert injected_anthropic.max_retries == 2

    generation = OpenAIGenerationClient(
        api_key=SecretStr("test"), model="gpt-5.6-terra", client=injected_openai
    )
    judging = AnthropicGenerationClient(
        api_key=SecretStr("test"), model="claude-sonnet-5", client=injected_anthropic
    )
    embedding = OpenAIEmbeddingClient(
        api_key=SecretStr("test"),
        model="text-embedding-3-small",
        dimensions=1536,
        client=openai.OpenAI(api_key="test"),
    )

    for wrapper in (generation, judging, embedding):
        assert wrapper._client.max_retries == 0

    # 타임아웃도 같은 표현식의 같은 구멍이었다 — 래퍼 값으로 고정한다.
    assert generation._client.timeout == 120.0
    assert judging._client.timeout == 120.0
    assert embedding._client.timeout == 60.0


def test_테스트_대역_주입은_그대로_통과한다() -> None:
    """대역 주입은 정상 용법이다 — `max_retries`·`with_options` 가 없어도 막지 않는다."""
    double = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: None))

    client = OpenAIGenerationClient(
        api_key=SecretStr("test"), model="gpt-5.6-terra", client=cast(openai.OpenAI, double)
    )

    assert cast(object, client._client) is double  # 사본을 만들지도 않는다


def test_재시도를_끌_수_없는_클라이언트는_조립에서_거부된다() -> None:
    """fail-closed — 재시도가 켜져 있는데 끌 수단이 없으면 조용히 받아들이지 않는다."""
    unfixable = SimpleNamespace(max_retries=2)

    with pytest.raises(ValueError, match="자동 재시도를 끌 수 없다"):
        OpenAIGenerationClient(
            api_key=SecretStr("test"), model="gpt-5.6-terra", client=cast(openai.OpenAI, unfixable)
        )


def test_llm_모듈의_모든_클라이언트_대입은_관문을_지난다() -> None:
    """구조 테스트 — 네 번째 래퍼가 생기거나 누가 `client or SDK(...)` 로 되돌리면 잡는다.

    인스턴스 속성 assert 만으로는 **새로 추가된 래퍼**를 잡지 못한다. 대입 지점 자체를
    검사해야 관문이 우회 불가능한 통로가 된다(`tests/test_gate.py` 의 격리 구조 테스트와
    같은 방식).
    """
    source = pathlib.Path("src/reply_gate/llm.py").read_text(encoding="utf-8")
    assignments = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr == "_client"
    ]

    assert len(assignments) >= 3, "클라이언트 대입 지점을 찾지 못했다 — 검사가 헛돌고 있다"
    for node in assignments:
        call = node.value
        assert isinstance(call, ast.Call), f"line {node.lineno}: 관문 호출이 아니다"
        assert isinstance(call.func, ast.Name) and call.func.id == "_pin_transport_policy", (
            f"line {node.lineno}: `self._client` 는 `_pin_transport_policy(...)` 결과여야 한다"
        )


def test_정상_응답은_데이터와_토큰을_돌려준다() -> None:
    client, responses = _generation_client([_response(json.dumps({"intent": "policy"}))])

    result = client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)

    assert result.data == {"intent": "policy"}
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert len(responses.calls) == 1

    call = responses.calls[0]
    # 결정론을 샘플링 파라미터로 보장하지 않는다 (docs/standards.md "샘플링 파라미터를 보내지 않는다") — 아예 보내지 않는다.
    assert not {"temperature", "top_p"} & call.keys()
    # reasoning 은 지정했을 때만 — 지원하지 않는 모델 등급에서 요청이 거부되기 때문.
    assert "reasoning" not in call
    assert call["model"] == "gpt-5.6-terra"
    assert call["text"]["format"]["schema"] is SCHEMA
    assert call["text"]["format"]["type"] == "json_schema"


def test_effort_는_지정했을_때만_전달된다() -> None:
    client, responses = _generation_client([_response(json.dumps({"intent": "policy"}))])

    client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA, effort="low")

    assert responses.calls[0]["reasoning"] == {"effort": "low"}


def test_전송오류는_1회만_재시도한다() -> None:
    client, responses = _generation_client(
        [_connection_error(), _response(json.dumps({"intent": "order"}))]
    )

    result = client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)

    assert result.data == {"intent": "order"}
    assert len(responses.calls) == MAX_ATTEMPTS


def test_전송오류가_지속되면_인계용_예외를_던진다() -> None:
    client, responses = _generation_client([_connection_error()])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.stage == "draft"
    assert excinfo.value.reason == "transport_error"
    assert excinfo.value.attempts == MAX_ATTEMPTS
    assert len(responses.calls) == MAX_ATTEMPTS


def test_5xx_는_전송오류로_재시도한다() -> None:
    client, responses = _generation_client(
        [_status_error(503), _response(json.dumps({"intent": "both"}))]
    )

    assert client.complete_json(stage="sql", system="s", user="u", schema=SCHEMA).data == {
        "intent": "both"
    }
    assert len(responses.calls) == MAX_ATTEMPTS


def test_4xx_는_재시도하지_않고_즉시_실패한다() -> None:
    client, responses = _generation_client([_status_error(400)])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.reason == "api_error"
    assert excinfo.value.attempts == 1
    assert len(responses.calls) == 1


def test_거절_응답은_사용가능한_산출이_없으므로_실패다() -> None:
    client, _ = _generation_client([_response("", refusal="정책상 답할 수 없습니다")])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.reason == "refusal"
    # 거절은 200 으로 온 응답이라 토큰이 이미 과금됐다 — 실패에 실어야 실비용이 남는다.
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (11, 7)


# ── attempts 는 상수가 아니라 실측이다 ─────────────────────────────────────


def test_전송_수는_상수가_아니라_실제로_나간_횟수다() -> None:
    """성공한 산출에도 실린다 — 형식 루프를 도는 호출자가 이 값을 누적하기 때문이다."""
    once, _ = _generation_client([_response(json.dumps({"intent": "policy"}))])
    twice, calls = _generation_client(
        [_connection_error(), _response(json.dumps({"intent": "policy"}))]
    )

    assert once.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)
    assert (
        once.complete_json(stage="intent", system="s", user="u", schema=SCHEMA).transport_attempts
        == 1
    )
    result = twice.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)
    assert result.transport_attempts == len(calls.calls) == 2


def test_형식오류에도_그때까지_나간_전송_수가_실린다() -> None:
    """1차 전송 실패 뒤 2차가 비 JSON 이면 전송은 2회다 — 호출자가 이어 세야 한다."""
    client, calls = _generation_client([_connection_error(), _response("이건 JSON 이 아니다")])

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.transport_attempts == len(calls.calls) == 2


def test_거절_경로의_attempts_도_실측이다() -> None:
    """구 코드는 `attempts=1` 하드코딩이라, 재시도 뒤에 온 거절이 1회로 신고됐다."""
    client, calls = _generation_client([_connection_error(), _response("", refusal="정책상 불가")])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.reason == "refusal"
    assert excinfo.value.attempts == len(calls.calls) == 2


def test_형식오류는_재시도하지_않고_호출자에게_위임한다() -> None:
    client, responses = _generation_client([_response("이건 JSON 이 아니다")])

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.stage == "intent"
    assert len(responses.calls) == 1
    # 초안 생성은 이 원문을 그대로 L1 에 넘겨 schema_violation 으로 판정시킨다.
    assert excinfo.value.raw_text == "이건 JSON 이 아니다"
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (11, 7)


def test_빈_응답도_형식오류다() -> None:
    client, _ = _generation_client([_response("   ")])

    with pytest.raises(LLMFormatError):
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)


# ── Anthropic 판정 클라이언트 ────────────────────────────────────────────────


def _anthropic_response(text: str, *, stop_reason: str = "end_turn") -> SimpleNamespace:
    """Messages API 응답 대역.

    adaptive thinking 이 모델 기본이라 text 블록 앞에 thinking 블록이 올 수 있다 —
    래퍼가 블록 타입으로 골라내는지 검증하기 위해 정상 응답에도 항상 끼워 넣는다.
    거절(stop_reason="refusal")은 content 가 비어 있다.
    """
    content: list[SimpleNamespace] = []
    if stop_reason != "refusal":
        content = [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ]
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def _anthropic_client(outcomes: list[Any]) -> tuple[AnthropicGenerationClient, _RecordingCalls]:
    messages = _RecordingCalls(outcomes)
    fake_sdk = cast(anthropic.Anthropic, SimpleNamespace(messages=messages))
    client = AnthropicGenerationClient(
        api_key=SecretStr("test"), model="claude-sonnet-5", client=fake_sdk
    )
    return client, messages


def _anthropic_connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))


def _anthropic_status_error(status_code: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status_code, request=request, json={"error": {"message": "x"}})
    return anthropic.APIStatusError("boom", response=response, body=None)


def test_anthropic_sdk_자동재시도를_끈다() -> None:
    """Anthropic SDK 도 기본 2회 자동 재시도한다 — 래퍼가 단독 통제하도록 차단한다."""
    client = AnthropicGenerationClient(api_key=SecretStr("test"), model="claude-sonnet-5")
    assert client._client.max_retries == 0


def test_anthropic_정상_응답은_데이터와_토큰을_돌려준다() -> None:
    client, messages = _anthropic_client([_anthropic_response(json.dumps({"verdict": "pass"}))])

    result = client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert result.data == {"verdict": "pass"}
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert len(messages.calls) == 1

    call = messages.calls[0]
    # 결정론을 샘플링 파라미터로 보장하지 않는다 — Sonnet 5 는 기본값 아닌 값을 보내면 400.
    assert not {"temperature", "top_p", "top_k"} & call.keys()
    # thinking 설정 미전송 — 미전송은 '끔'이 아니라 adaptive thinking 켜짐이 모델 기본이다.
    assert "thinking" not in call
    # effort 는 지정했을 때만 (output_config 안에 실린다).
    assert "effort" not in call["output_config"]
    assert call["model"] == "claude-sonnet-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"] is SCHEMA
    # max_tokens 는 thinking+응답 합산 상한이므로 반드시 실린다.
    assert call["max_tokens"] > 0
    assert call["system"] == "s"
    assert call["messages"] == [{"role": "user", "content": "u"}]


def test_anthropic_effort_는_지정했을_때만_전달된다() -> None:
    client, messages = _anthropic_client([_anthropic_response(json.dumps({"verdict": "pass"}))])

    client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA, effort="low")

    assert messages.calls[0]["output_config"]["effort"] == "low"


def test_anthropic_전송오류는_1회만_재시도한다() -> None:
    client, messages = _anthropic_client(
        [_anthropic_connection_error(), _anthropic_response(json.dumps({"verdict": "pass"}))]
    )

    result = client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert result.data == {"verdict": "pass"}
    assert len(messages.calls) == MAX_ATTEMPTS


def test_anthropic_전송오류가_지속되면_인계용_예외를_던진다() -> None:
    client, messages = _anthropic_client([_anthropic_connection_error()])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.stage == "judge"
    assert excinfo.value.reason == "transport_error"
    assert excinfo.value.attempts == MAX_ATTEMPTS
    assert len(messages.calls) == MAX_ATTEMPTS


def test_anthropic_5xx_는_전송오류로_재시도한다() -> None:
    client, messages = _anthropic_client(
        [_anthropic_status_error(503), _anthropic_response(json.dumps({"verdict": "fail"}))]
    )

    assert client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA).data == {
        "verdict": "fail"
    }
    assert len(messages.calls) == MAX_ATTEMPTS


def test_anthropic_4xx_는_재시도하지_않고_즉시_실패한다() -> None:
    client, messages = _anthropic_client([_anthropic_status_error(400)])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.reason == "api_error"
    assert excinfo.value.attempts == 1
    assert len(messages.calls) == 1


def test_anthropic_거절_응답은_사용가능한_산출이_없으므로_실패다() -> None:
    """안전 분류기 거절은 HTTP 200 + stop_reason="refusal" 로 온다 — 오류가 아니라 응답이다."""
    client, _ = _anthropic_client([_anthropic_response("", stop_reason="refusal")])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.reason == "refusal"
    # 200 으로 온 응답이므로 입력 토큰은 이미 과금됐다 — 판정 토큰이 0 으로 기록되면
    # 실제로 돈이 나간 판정 호출이 처리 기록에서 사라진다.
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (11, 7)


def test_anthropic_형식오류는_재시도하지_않고_호출자에게_위임한다() -> None:
    client, messages = _anthropic_client([_anthropic_response("이건 JSON 이 아니다")])

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.stage == "judge"
    assert len(messages.calls) == 1
    assert excinfo.value.raw_text == "이건 JSON 이 아니다"
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (11, 7)


def test_anthropic_빈_응답도_형식오류다() -> None:
    client, _ = _anthropic_client([_anthropic_response("   ")])

    with pytest.raises(LLMFormatError):
        client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)


# ── 임베딩 ──────────────────────────────────────────────────────────────────


def _embedding_response(vectors: list[tuple[int, list[float]]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in vectors],
        usage=SimpleNamespace(total_tokens=42),
    )


def _openai_client(outcomes: list[Any]) -> tuple[OpenAIEmbeddingClient, _RecordingCalls]:
    embeddings = _RecordingCalls(outcomes)
    fake_sdk = cast(openai.OpenAI, SimpleNamespace(embeddings=embeddings))
    client = OpenAIEmbeddingClient(
        api_key=SecretStr("test"), model="text-embedding-3-small", dimensions=3, client=fake_sdk
    )
    return client, embeddings


def test_임베딩은_입력_순서대로_정렬해_돌려준다() -> None:
    client, _ = _openai_client([_embedding_response([(1, [0.0, 1.0, 0.0]), (0, [1.0, 0.0, 0.0])])])

    result = client.embed(stage="policy_index", texts=["가", "나"])

    assert result.vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert result.total_tokens == 42


def test_임베딩_전송오류도_1회만_재시도한다() -> None:
    client, embeddings = _openai_client(
        [_connection_error(), _embedding_response([(0, [1.0, 0.0, 0.0])])]
    )

    result = client.embed(stage="inquiry", texts=["문의"])

    assert result.vectors == [[1.0, 0.0, 0.0]]
    assert len(embeddings.calls) == MAX_ATTEMPTS


def test_빈_입력은_호출하지_않는다() -> None:
    client, embeddings = _openai_client([])

    result = client.embed(stage="inquiry", texts=[])

    assert result.vectors == []
    assert embeddings.calls == []


class _LocalEmbeddingModel:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def encode(self, sentences: Sequence[str], *, normalize_embeddings: bool) -> object:
        self.calls.append((tuple(sentences), normalize_embeddings))
        return self.vectors


def test_BGE_M3_는_선택_모델_경계에서_1024차원_dense_벡터를_만든다() -> None:
    model = _LocalEmbeddingModel([[0.0] * 1024, [1.0] * 1024])
    client = BgeM3EmbeddingClient(model=model)

    result = client.embed(stage="retrieval", texts=["문의", "정책"])

    assert client.dimensions == 1024
    assert result.vectors == model.vectors
    assert result.total_tokens == 0
    assert model.calls == [(("문의", "정책"), True)]


def test_BGE_M3_는_잘못된_차원을_거부한다() -> None:
    client = BgeM3EmbeddingClient(model=_LocalEmbeddingModel([[0.0] * 1536]))

    with pytest.raises(ValueError, match="1024차원"):
        client.embed(stage="retrieval", texts=["문의"])


def _json_client(
    provider: str, text: str
) -> tuple[OpenAIGenerationClient | AnthropicGenerationClient, _RecordingCalls]:
    if provider == "openai":
        return _generation_client([_response(text)])
    return _anthropic_client([_anthropic_response(text)])


def _assert_json_format_failure(provider: str, text: str) -> None:
    client, calls = _json_client(provider, text)
    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    error = excinfo.value
    assert error.stage == "draft"
    assert error.raw_text == text
    assert text not in str(error)
    assert (error.input_tokens, error.output_tokens) == (11, 7)
    assert error.transport_attempts == len(calls.calls) == 1
    assert error.elapsed_ms >= 0


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_유한하지_않은_JSON_숫자는_원문과_토큰을_보존한_형식오류다(
    provider: str, number: str
) -> None:
    """Python 확장 상수와 float 오버플로를 받아들이면 JSONB 저장 단계에서야 실패한다."""
    _assert_json_format_failure(provider, '{"intent": "policy", "extra": ' + number + "}")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_JSON_정수_자릿수_제한도_형식오류로_변환한다(provider: str) -> None:
    """json.loads의 일반 ValueError가 초안/L1 경로와 실패 호출 토큰을 우회했던 입력이다."""
    original_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        _assert_json_format_failure(provider, '{"claims": [' + "1" * 4301 + "]}")
    finally:
        sys.set_int_max_str_digits(original_limit)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_JSON_중첩_깊이_제한도_형식오류로_변환한다(provider: str) -> None:
    """파서의 깊이 제한 역시 모델 산출 실패다. Python 3.13의 C 파서 한계도 넘긴다."""
    _assert_json_format_failure(provider, "[" * 10000 + "0" + "]" * 10000)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_유효한_JSON_숫자와_상수_모양_문자열은_바꾸지_않는다(provider: str) -> None:
    """음성 입력을 거부하는 동안 정상 숫자와 고객 문장 안의 문자열까지 훼손하지 않는다."""
    payload = {"values": [0, -3, 1.25, 1e100], "text": "NaN Infinity -Infinity 1e999"}
    client, calls = _json_client(provider, json.dumps(payload))

    result = client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert result.data == payload
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert len(calls.calls) == 1


def test_JSON_숫자_형식실패도_초안은_원문으로_L1에_넘긴다() -> None:
    """실제 래퍼→초안 생성 경로: 형식 오류는 초안 생성기의 재시도나 500이 되지 않는다."""
    text = '{"claims": [{"text": "환불 안내", "citation_ids": [], "extra": NaN}]}'
    client, calls = _json_client("openai", text)

    result = DraftGenerator(client=client).generate(inquiry="환불 정책", evidence=())

    assert result.raw == text
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert len(calls.calls) == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "text",
    [
        r'{"nested":[{"text":"\u0000"}]}',
        r'{"nested":[{"\u0000":"value"}]}',
        r'{"nested":[{"text":"\ud800"}]}',
        r'{"nested":[{"\ud800":"value"}]}',
        r'{"nested":[{"text":"\udfff"}]}',
        r'{"nested":[{"\udfff":"value"}]}',
    ],
    ids=["nul-value", "nul-key", "high-value", "high-key", "low-value", "low-key"],
)
def test_JSON_저장불가_문자는_키와_값_모두_형식오류다(provider: str, text: str) -> None:
    """escape된 원문은 저장 가능해도 파싱 결과의 문자열이 저장 불가일 수 있다."""
    _assert_json_format_failure(provider, text)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("character", ["\x00", "\ud800", "\udfff"], ids=["nul", "high", "low"])
@pytest.mark.parametrize("malformed", [False, True], ids=["json", "malformed"])
def test_JSON_저장불가_원문은_역복원_가능한_진단문자열로_보존한다(
    provider: str, character: str, malformed: bool
) -> None:
    """파싱 실패 원문 자체에도 NUL/surrogate가 있으면 기록 저장 단계가 다시 실패한다."""
    text = "잘못된 JSON " + character if malformed else '{"text":"' + character + '"}'
    client, calls = _json_client(provider, text)

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    error = excinfo.value
    prefix = "[Unicode escaped raw text] "
    assert error.raw_text.startswith(prefix)
    assert error.raw_text.removeprefix(prefix).encode("ascii").decode("unicode_escape") == text
    assert "\x00" not in error.raw_text
    error.raw_text.encode("utf-8")  # lone surrogate가 남으면 이 줄에서 실패한다.
    assert (error.input_tokens, error.output_tokens) == (11, 7)
    assert error.transport_attempts == len(calls.calls) == 1
    assert error.elapsed_ms >= 0


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_JSON_정상_유니코드와_백슬래시_문자열은_보존한다(provider: str, ensure_ascii: bool) -> None:
    """실제 emoji와 escaped surrogate pair는 정상이며 '\\u0000' 글자열도 NUL이 아니다."""
    payload = {
        "한글\n": [{"text": "안내 😀\n\t", "literal": r"\u0000 \ud800 \udfff"}],
        "other": [None, True, 42],
    }
    client, calls = _json_client(provider, json.dumps(payload, ensure_ascii=ensure_ascii))

    result = client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert result.data == payload
    assert len(calls.calls) == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_JSON_문자검사는_깊은_정상배열을_추가로_거부하지_않는다(provider: str) -> None:
    """JSON 파서는 허용하는 1,100단을 후속 재귀 문자열 검사가 새로 실패시키면 안 된다."""
    text = "[" * 1100 + '"한글 😀"' + "]" * 1100
    client, calls = _json_client(provider, text)

    result = client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    value = result.data
    for _ in range(1100):
        assert isinstance(value, list)
        value = value[0]
    assert value == "한글 😀"
    assert len(calls.calls) == 1


@pytest.mark.parametrize("text", ["잘못된 JSON", '[Unicode escaped raw text] "원문"'])
def test_JSON_안전한_실패원문은_진단접두어처럼_보여도_바꾸지_않는다(text: str) -> None:
    """접두어만 보고 원문을 재인코딩하거나 자동 decode하지 않는다."""
    _assert_json_format_failure("openai", text)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_JSON_문자실패도_캐시와_시간을_보존한다(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """저장 적합성 검사가 기존 실패 호출의 계열별 실비용이나 시간을 지우면 안 된다."""
    client, calls = _json_client(provider, r'{"text":"\u0000"}')
    usage = calls._outcomes[0].usage
    if provider == "openai":
        usage.input_tokens_details = SimpleNamespace(cache_write_tokens=3, cached_tokens=5)
    else:
        usage.cache_creation_input_tokens = 3
        usage.cache_read_input_tokens = 5
    clock = iter([1.0, 1.25])
    monkeypatch.setattr("reply_gate.llm.perf_counter", lambda: next(clock))

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    error = excinfo.value
    assert (error.input_tokens, error.output_tokens) == (11, 7)
    assert (error.cache_creation_input_tokens, error.cache_read_input_tokens) == (3, 5)
    assert error.elapsed_ms == 250.0
    assert error.transport_attempts == len(calls.calls) == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "characters",
    ["\ud83d\ude00\x00", "😀\x00", r"\ud83d\ude00\u0000" + "\x00"],
    ids=["literal-pair-nul", "emoji-nul", "backslash-nul"],
)
def test_JSON_진단표현은_surrogate쌍과_emoji와_백슬래시를_구별해_복원한다(
    provider: str, characters: str
) -> None:
    """JSON 재파싱은 literal surrogate 두 개를 emoji 하나로 합쳐 진단 원문을 잃었다."""
    text = "잘못된 원문 " + characters
    client, calls = _json_client(provider, text)

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    error = excinfo.value
    prefix = "[Unicode escaped raw text] "
    assert error.raw_text.startswith(prefix)
    restored = error.raw_text.removeprefix(prefix).encode("ascii").decode("unicode_escape")
    assert restored == text
    assert [ord(character) for character in restored] == [ord(character) for character in text]
    error.raw_text.encode("ascii")
    assert (error.input_tokens, error.output_tokens) == (11, 7)
    assert error.transport_attempts == len(calls.calls) == 1
