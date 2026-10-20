"""테스트 대역 자체의 성질 검증 — 대역이 틀리면 이 대역을 쓰는 모든 검증이 무의미해진다."""

from __future__ import annotations

from typing import Any, cast

import pytest

from reply_gate.api import build_embedding_client
from reply_gate.config import AbstentionGateWiringError
from reply_gate.contracts import Claim, Draft, Evidence, EvidenceSource, RejectReason, Verdict
from reply_gate.evidence import EvidenceCollector
from reply_gate.llm import GenerationClient
from reply_gate.pipeline import build_pipeline
from reply_gate.testing import (
    LexicalEmbeddingClient,
    StubEvidenceCollector,
    StubJudge,
    build_stub_pipeline,
)
from tests.conftest import declared_settings

_EVIDENCE = Evidence(
    id="policy:refund:2-4",
    source=EvidenceSource.POLICY,
    content="[환불 정책 2-4] 검수가 완료되면 3영업일 이내에 환불이 진행됩니다.",
    evidence_text="[환불 정책 2-4] 검수가 완료되면 3영업일 이내에 환불이 진행됩니다.",
)


def _draft(text: str) -> Draft:
    return Draft(claims=(Claim(text=text, citation_ids=(_EVIDENCE.id,)),))


class _UnusableGenerationClient:
    """조립만 확인하는 테스트용 — 호출되면 실패한다."""

    def complete_json(self, **kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("조립 확인 테스트는 생성 클라이언트를 부르지 않는다")


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_같은_입력은_같은_벡터를_준다() -> None:
    client = LexicalEmbeddingClient(dimensions=64)

    first = client.embed(stage="t", texts=["환불은 수령 후 7일 이내"]).vectors[0]
    second = client.embed(stage="t", texts=["환불은 수령 후 7일 이내"]).vectors[0]

    assert first == second


def test_어휘가_겹칠수록_유사도가_높다() -> None:
    client = LexicalEmbeddingClient(dimensions=512)
    query = "환불 신청 기간이 어떻게 되나요"
    related = "환불 신청은 상품 수령 후 7일 이내에 가능합니다"
    unrelated = "임직원 주차장 이용 규정 안내"

    vectors = client.embed(stage="t", texts=[query, related, unrelated]).vectors

    assert _cosine(vectors[0], vectors[1]) > _cosine(vectors[0], vectors[2])


def test_단위벡터를_돌려준다() -> None:
    client = LexicalEmbeddingClient(dimensions=128)

    for text in ["배송", "", "교환 규정"]:
        vector = client.embed(stage="t", texts=[text]).vectors[0]
        assert abs(_cosine(vector, vector) - 1.0) < 1e-9
        assert len(vector) == 128


@pytest.mark.parametrize(
    "model",
    [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "future-provider/model-v99",
        "real:stub:pretending",
        "lexical-2gram-v2",
        "stub:",
        "   ",
    ],
)
def test_어휘_대역은_실제_모델이나_불명확한_출처를_사칭하지_못한다(model: str) -> None:
    """known fake의 자유 model alias가 실제 τ 배선 검사를 통과시키던 경로를 닫는다."""
    with pytest.raises(ValueError, match=r"대역.*출처"):
        LexicalEmbeddingClient(model=model)


@pytest.mark.parametrize("model", ["stub:lexical-2gram", "stub:custom-model", "lexical-2gram-v1"])
def test_어휘_대역의_명시_출처와_기존_평가_alias는_보존한다(model: str) -> None:
    """양성 대조: 대역 실험의 모델 구분과 기존 캐시 계열은 유지한다."""
    embedder = LexicalEmbeddingClient(dimensions=8, model=model)
    assert embedder.model == model
    assert len(embedder.embed(stage="test", texts=["정책"]).vectors[0]) == 8


# ── 결정론 판정 대역 ────────────────────────────────────────────────────────


def test_판정_대역은_근거에_없는_값을_기각하고_있는_값은_통과시킨다() -> None:
    """양성·음성 짝 — 전부 기각하는 대역은 배관 검증에 쓸 수 없다."""
    judge = StubJudge()

    supported = judge.judge(
        draft=_draft("검수 후 3영업일 이내에 환불됩니다."), evidence=[_EVIDENCE]
    )
    fabricated = judge.judge(
        draft=_draft("검수 후 7영업일 이내에 환불됩니다."), evidence=[_EVIDENCE]
    )

    assert supported.result.verdict is Verdict.PASS
    assert supported.result.reject_reasons == ()
    assert fabricated.result.verdict is Verdict.REJECT
    assert fabricated.result.reject_reasons == (RejectReason.UNSUPPORTED_CLAIM,)
    # claim 판정은 초안의 claim 전부와 1:1 로 대응한다 (판정 계약과 같은 모양).
    assert [judgment.claim_text for judgment in fabricated.result.claim_judgments] == [
        "검수 후 7영업일 이내에 환불됩니다."
    ]
    assert fabricated.input_tokens > 0 and fabricated.output_tokens > 0


@pytest.mark.parametrize(
    ("amount", "expected"),
    [("30,000", Verdict.PASS), ("30000", Verdict.PASS), ("30", Verdict.REJECT)],
)
def test_판정_대역은_천_단위_구분자를_금액의_일부로_본다(
    amount: str,
    expected: Verdict,
) -> None:
    """30,000을 30과 0으로 쪼개면 동치 금액은 기각하고 다른 금액은 통과한다."""
    evidence = Evidence(
        id=_EVIDENCE.id,
        source=EvidenceSource.POLICY,
        content="환불 금액은 30,000원입니다.",
        evidence_text="환불 금액은 30,000원입니다.",
    )
    result = StubJudge().judge(draft=_draft(f"환불 금액은 {amount}원입니다."), evidence=[evidence])
    assert result.result.verdict is expected


def test_판정_대역은_같은_입력에_같은_판정을_준다() -> None:
    first = StubJudge().judge(draft=_draft("검수 후 3영업일."), evidence=[_EVIDENCE])
    second = StubJudge().judge(draft=_draft("검수 후 3영업일."), evidence=[_EVIDENCE])

    assert first == second


def test_판정_대역은_호출을_기록한다() -> None:
    judge = StubJudge()
    judge.judge(draft=_draft("검수 후 3영업일."), evidence=[_EVIDENCE])

    assert len(judge.calls) == 1
    assert judge.calls[0].claims[0].text == "검수 후 3영업일."


def test_대역_조립기는_임베딩_검증_경계만_가르고_같은_수집_동작을_쓴다() -> None:
    """실제 τ 조건 검사에 대역 우회를 넣지 않고, 수집·채택 동작은 그대로 공유한다."""
    settings = declared_settings(l2_enabled=True)
    generation = cast(GenerationClient, _UnusableGenerationClient())
    embedding = LexicalEmbeddingClient(dimensions=8)

    real = build_pipeline(
        generation_client=generation,
        embedding_client=build_embedding_client(settings),
        settings=settings,
    )
    stub = build_stub_pipeline(
        generation_client=generation,
        embedding_client=embedding,
        judge=StubJudge(),
        settings=settings,
    )

    assert type(stub._collector) is StubEvidenceCollector
    assert type(real._collector) is EvidenceCollector
    assert StubEvidenceCollector.collect is EvidenceCollector.collect
    assert StubEvidenceCollector._collect_policy is EvidenceCollector._collect_policy
    assert type(stub._drafter) is type(real._drafter)
    assert stub._l2_enabled == real._l2_enabled is True
    same_identity_settings = stub._settings is real._settings is settings
    assert same_identity_settings, "확인 TTL·실패 제한·HMAC 설정도 수집기 밖까지 전달해야 한다"
    # 검색 전략 배선까지 같아야 한다 — 여기서 `None` 이면 대역 실행이 재작성 없는
    # 파이프라인을 재고, 그 수치가 "대역 완주"로 보고된다. 타입 대조만으로는 못 잡는다.
    stub_collector = stub._collector
    real_collector = real._collector
    assert stub_collector._rewrite_client is real_collector._rewrite_client is generation
    # 채택 축도 같아야 한다 — 기권 게이트가 한쪽에만 걸리면 대역이 재는 채택 집합이
    # 제품과 달라지고, 그 차이가 "대역 완주" 수치에 조용히 섞인다.
    assert stub_collector._abstention_gate == real_collector._abstention_gate
    assert stub_collector._abstention_gate == settings.abstention_gate()
    assert stub_collector._settings.vector_similarity_threshold == pytest.approx(
        real_collector._settings.vector_similarity_threshold
    )
    assert stub_collector._settings.vector_top_k == real_collector._settings.vector_top_k
    # 판정 대역과 어휘 임베딩 조건만 다르고 수집·채택 구현은 공유한다.
    assert isinstance(stub._judge, StubJudge)
    assert not isinstance(real._judge, StubJudge)

    # 실제 조립에는 어휘 대역 예외가 없다. 출처를 실제 모델로 바꿔 속이지도 않는다.
    assert stub_collector._embedder.model == LexicalEmbeddingClient.MODEL
    with pytest.raises(AbstentionGateWiringError, match="임베더"):
        build_pipeline(generation_client=generation, embedding_client=embedding, settings=settings)


@pytest.mark.parametrize("gate_enabled", [False, True])
@pytest.mark.parametrize("model", ["text-embedding-3-small", "stub:pretending"])
def test_대역_수집기는_어휘_임베딩이_아닌_클라이언트를_거부한다(
    gate_enabled: bool, model: str
) -> None:
    """모델 이름이나 off 플래그가 실제 임베더를 대역 예외로 승격시키지 않는다."""
    settings = declared_settings(query_rewrite_enabled=False, abstention_gate_enabled=gate_enabled)
    with pytest.raises(AbstentionGateWiringError, match="LexicalEmbeddingClient"):
        StubEvidenceCollector(
            generation_client=cast(GenerationClient, _UnusableGenerationClient()),
            embedding_client=build_embedding_client(declared_settings(embedding_model=model)),
            settings=settings,
        )


def test_대역_수집기도_설정의_τ_검증을_건너뛰지_않는다() -> None:
    """대역 예외는 실험용 점수 공급만 허용하며 미검증 설정 조합을 허용하지 않는다."""
    with pytest.raises(AbstentionGateWiringError, match="검증된 적이 없다"):
        StubEvidenceCollector(
            generation_client=cast(GenerationClient, _UnusableGenerationClient()),
            embedding_client=LexicalEmbeddingClient(),
            settings=declared_settings(
                query_rewrite_enabled=False, embedding_model="text-embedding-3-large"
            ),
        )
