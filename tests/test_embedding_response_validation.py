"""모델 산출 원문과 임베딩 배치 경계. 실제 API 호출은 없다."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import openai
import pytest
from pydantic import SecretStr

from reply_gate import llm
from reply_gate.testing import LexicalEmbeddingClient


@pytest.mark.parametrize(
    "text",
    ['{"verdict":"reject","verdict":"pass"}', '{"x":{"ok":false,"ok":true}}', '{"x":1,"x":1}'],
)
def test_중복_JSON_키는_계층과_값에_관계없이_형식_오류다(text: str) -> None:
    with pytest.raises(llm.LLMFormatError):
        llm._parse_json_completion(stage="judge", text=text, input_tokens=7, output_tokens=9)


def test_추가_JSON_키는_중복이_아니면_그대로_남는다() -> None:
    result = llm._parse_json_completion(
        stage="judge", text='{"verdict":"pass","extra":1}', input_tokens=7, output_tokens=9
    )
    assert result.data == {"verdict": "pass", "extra": 1}
    assert (result.input_tokens, result.output_tokens) == (7, 9)


@pytest.mark.parametrize(
    "indices,vectors",
    [
        ([0], [[1.0, 0.0]]),
        ([0, 0], [[1.0, 0.0], [0.0, 1.0]]),
        ([0, 2], [[1.0, 0.0], [0.0, 1.0]]),
        ([0, 1], [[1.0], [0.0, 1.0]]),
        ([0, 1], [[float("nan"), 1.0], [0.0, 1.0]]),
        ([0, 1], [[float("inf"), 1.0], [0.0, 1.0]]),
        ([0, 1], [[0.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_임베딩_배치의_잘못된_개수와_색인과_벡터를_거부한다(
    indices: list[int], vectors: list[list[float]]
) -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=v) for i, v in zip(indices, vectors, strict=True)],
        usage=SimpleNamespace(total_tokens=7),
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: response))
    client.with_options = lambda **kw: client
    embedder = llm.OpenAIEmbeddingClient(
        api_key=SecretStr(""), model="fixed", dimensions=2, client=cast(openai.OpenAI, client)
    )
    with pytest.raises(ValueError):
        embedder.embed(stage="query", texts=["a", "b"])


@pytest.mark.parametrize("text", ["", " ", "!@#", "문의"])
def test_어휘_대역의_빈_입력도_비영_벡터_계약을_지킨다(text: str) -> None:
    result = LexicalEmbeddingClient(dimensions=8).embed(stage="query", texts=[text])
    llm.validate_embedding_vectors(result.vectors, count=1, dimensions=8)


def test_임베딩_색인_순서는_복원하되_정상_희소벡터를_허용한다() -> None:
    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.0, 2.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ],
        usage=SimpleNamespace(total_tokens=7),
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: response))
    client.with_options = lambda **kw: client
    result = llm.OpenAIEmbeddingClient(
        api_key=SecretStr(""), model="fixed", dimensions=2, client=cast(openai.OpenAI, client)
    ).embed(stage="query", texts=["a", "b"])
    assert result.vectors == [[1.0, 0.0], [0.0, 2.0]]
    assert result.total_tokens == 7


@pytest.mark.parametrize("vector", [[0.0] * 1024, [float("nan")] * 1024, [True] * 1024])
def test_로컬_임베딩도_영벡터와_잘못된_수치를_거부한다(vector: list[float]) -> None:
    from tests.test_llm_client import _LocalEmbeddingModel

    client = llm.BgeM3EmbeddingClient(model=_LocalEmbeddingModel([vector]))
    with pytest.raises(ValueError):
        client.embed(stage="query", texts=["문의"])
