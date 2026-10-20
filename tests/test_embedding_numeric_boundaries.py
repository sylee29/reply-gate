"""T7-R4/R5 — 비정상 임베딩 수치의 합성 경계, 제공자 호출 0회."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from reply_gate import llm, retrieval_eval
from tests.test_llm_client import _LocalEmbeddingModel
from tests.test_retrieval_eval import _FixedEmbedder


@pytest.mark.parametrize("value", [1e308, 1e-300])
def test_연산할_수_없는_노름은_검색과_캐시_전_거부한다(tmp_path: Path, value: float) -> None:
    embedder = _FixedEmbedder({"정책": [value, 0.0], "문의": [value, 0.0]})
    with pytest.raises(ValueError):
        retrieval_eval.retrieve_cases(
            queries=[retrieval_eval.RetrievalQuery("A", "문의")],
            policy_texts=[("p", "정책")],
            embedder=embedder,
            config=retrieval_eval.RetrievalEvalConfig(model="fixed", dimensions=2),
            cache_dir=tmp_path,
        )
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("value", [1e308, 1e-300])
def test_연산할_수_없는_노름의_기존_캐시도_미스로_돌린다(tmp_path: Path, value: float) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"model": "m", "dimensions": 2, "text_hash": "h", "vector": [value, 0.0]})
    )
    assert (
        retrieval_eval._read_cached_vector(path=path, model="m", dimensions=2, text_hash="h")
        is None
    )


def test_BGE의_거대한_정수도_공통_ValueError로_거부한다() -> None:
    client = llm.BgeM3EmbeddingClient(
        model=_LocalEmbeddingModel([cast(list[float], [10**400] + [0] * 1023)])
    )
    with pytest.raises(ValueError):
        client.embed(stage="query", texts=["문의"])
