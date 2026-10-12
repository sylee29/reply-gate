"""FastAPI 표면 — 엔드포인트 5개와 응답 스키마 (docs/contracts.md).

| `POST /inquiries`       | 문의 접수 + 동기 처리 |
| `POST /inquiries/{id}/confirm` | 상품 확인 + 같은 문의 처리 |
| `GET  /inquiries/{id}`  | 처리 기록 조회 (저장된 값에서 재구성) |
| `GET  /`                | 최소 웹 폼 1장 |
| `GET  /health`          | 헬스 체크 |

응답 스키마의 계약은 **모든 키가 항상 존재한다** 이다: `answer`·`escalation_reason` 은
해당 없을 때 null, `claims`·`citations`·`attempts` 는 해당 없을 때 빈 배열. 초안 전
인계라도 그때까지 수집된 근거는 `citations` 에 남는다(감사 목적).

`attempts[]` 는 **종합 판정과 층별 판정을 함께** 싣는다: `verdict`/`reject_reasons` 는
기존 키 그대로 종합이고(값 집합에만 L2 사유 2종이 더해졌다), `l1`/`l2` 가 층별 내역이다.
`l2` 는 **미실행이면 `null`** 이고(L1 reject · 스위치 꺼짐 · L2 호출 실패) 키는 사라지지
않는다 — `null` 은 "통과"가 아니라 "판정이 없었다"는 뜻이며, 그 시도의 진실은 인계 사유가
들고 있다.

`metrics.tokens` 의 `input`/`output` 은 **생성 LLM 합산**이고 임베딩·판정·검색 토큰을 여기
섞지 않는다. 판정 토큰은 `judge_input`/`judge_output` 으로(L2 미실행이면 0), 검색 단계
재작성 토큰은 `retrieval_input`/`retrieval_output` 으로 **분리 신설**한다 — provider·단가가
다르고, 검색 계열은 특히 "초안 없이 끝난 문의"의 비용을 초안 생성 칸에 찍지 않기 위한
분리다. 임베딩 토큰은 지금도 응답에 싣지 않고 처리 기록에만 남는다.

`metrics.retrieval_fallback_reason` 은 검색 단계가 폴백한 사유다. `null` 이 기본이고
**인계 사유가 아니다** — 폴백한 문의도 그대로 답변될 수 있다.

접수 검증(내용 필수·주문번호 형식)은 **파이프라인에 들어가기 전에** 422 로 끝난다.
형식이 틀린 주문번호는 인계가 아니라 요청 오류다.

**DB 커넥션도 첫 사용 직전에 연다** — 의존성 해석 시점에 열면 DB 가 없을 때 요청 오류와
설정 오류가 모두 500 에 가린다(422 도 503 도 커넥션이 먼저 터진다). 라우트가 자기 선검사를
끝낸 뒤 `with open_service() as service` 로 여는 형태라, 이 순서는 프레임워크가 의존성을
어떤 차례로 푸는지와 무관하게 성립한다. DB 가 **진짜** 필요한 경로(조회·처리)는 그대로
500 이다 — DB 장애를 업무 판정으로 바꾸지 않는다(docs/standards.md).

생성·임베딩 클라이언트는 **첫 호출 직전에** 만든다(`_LazyGenerationClient`). 의존성 해석
시점에 만들면 API 키가 없는 환경에서 LLM 을 전혀 쓰지 않는 경로 — 조회 전용
`GET /inquiries/{id}`, 접수 거부 422 — 까지 500 으로 무너진다. 자격 증명 부재는 업무
판정이 아니므로 인계 사유(`llm_call_failed`)로 기록하지 않고 **설정 오류(503)** 로 끝낸다
(`MissingCredentialsError` 의 정의는 `pipeline.py` 가 소유한다 — 파이프라인 내부의 지연
클라이언트도 같은 예외를 던져야 이 핸들러가 잡는다).

판정 키(`ANTHROPIC_API_KEY`)만 예외적으로 **eager** 다: L2 가 켜져 있으면
`POST /inquiries` 는 처리에 진입하자마자 키를 선검사해 503 을 낸다(`_require_judge_credentials`).
검증하지 못할 답변을 만들기 시작하지 않기 위해서다. 선검사는 **POST 경로에만** 걸고
조회·422 경로는 건드리지 않는다.

핸들러는 동기(`def`)다 — psycopg 는 동기 드라이버이고, FastAPI 가 스레드풀에서 돌린다.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Annotated, Any, Self

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from psycopg.rows import DictRow
from pydantic import BaseModel, Field, SecretStr, StrictBool, model_validator

from reply_gate.config import Settings, get_settings
from reply_gate.contracts import EvidenceSource, GateResult, InquiryStatus, JudgeResult
from reply_gate.db import connect, readonly_connect
from reply_gate.evidence import render_sql_rows
from reply_gate.gate import redact_pii
from reply_gate.llm import (
    EmbeddingClient,
    EmbeddingResult,
    GenerationClient,
    JsonCompletion,
    OpenAIEmbeddingClient,
    OpenAIGenerationClient,
)
from reply_gate.pipeline import (
    AttemptRecord,
    ConfirmationConflict,
    ConfirmationNotFound,
    InquiryPipeline,
    MissingCredentialsError,
    ProcessedInquiry,
    ReceiptError,
    accept_inquiry,
    build_pipeline,
    new_inquiry_id,
    require_identity_credentials,
)
from reply_gate.policy_index import PolicyIndexEmptyError, PolicyIndexProvenanceError
from reply_gate.records import load_inquiry, save_inquiry
from reply_gate.sql_guard import SqlGuardRejection, private_output_columns

__all__ = [
    "InquiryRequest",
    "InquiryResponse",
    "InquiryService",
    "MissingCredentialsError",
    "ServiceOpener",
    "app",
    "build_embedding_client",
    "build_generation_client",
    "get_service",
]

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

app = FastAPI(
    title="Reply-Gate",
    description="근거 없는 답변을 스스로 기각하는 이커머스 CS 답변 에이전트",
    version="0.1.0",
)


# ── 요청 ────────────────────────────────────────────────────────────────────


class InquiryRequest(BaseModel):
    """접수 입력. 검증은 `pipeline.accept_inquiry` 한 곳에서만 한다.

    주문번호 형식 정의는 `reply_gate.order_ref` 가 단독 소유한다 — 여기서 정규식을
    다시 쓰지 않는다. 검증을 통과하면 정규화된 값으로 갈아 끼워 아래로 흘린다.
    """

    content: str = Field(description="문의 내용 (필수)")
    order_no: str | None = Field(default=None, description="주문번호 ORD-YYYYMMDD-NNNN (선택)")
    phone_last4: str | None = Field(default=None, repr=False, description="전화번호 뒷자리 (선택)")

    @model_validator(mode="after")
    def _accept(self) -> Self:
        try:
            accepted = accept_inquiry(
                content=self.content, order_no=self.order_no, phone_last4=self.phone_last4
            )
        except ReceiptError as exc:
            raise ValueError(str(exc)) from exc
        self.content = accepted.content
        self.order_no = accepted.order_no
        self.phone_last4 = accepted.phone_last4
        return self


class ConfirmationRequest(BaseModel):
    confirmed: StrictBool


# ── 응답 ────────────────────────────────────────────────────────────────────


class ClaimOut(BaseModel):
    text: str
    citation_ids: list[str]


class CitationOut(BaseModel):
    id: str
    source: str
    content: str


class ClaimJudgmentOut(BaseModel):
    """L2 가 claim 1개에 내린 판정. 데모의 "왜 기각됐는지" 장면이 여기서 나온다.

    `claim_text` 는 판정 대상 claim 의 text 참조다 — 답변 계약의 claim 에는 별도 ID 가
    없으므로 text 로 가리킨다(`contracts.ClaimJudgment` 와 같은 규칙).
    """

    claim_text: str
    verdict: str
    explanation: str


class ContradictionOut(BaseModel):
    """L2 가 찾은 근거쌍 모순 1건.

    모순은 **근거쌍 단위**라 특정 claim 에 귀속되지 않을 수 있어 claim 판정과 별도 배열이다.
    기록됐다고 곧 기각은 아니다 — 초안이 모순을 명시하고 두 기준을 모두 안내했으면
    `reject_reasons` 에 오르지 않고 기록만 남는다.
    """

    evidence_id_a: str
    evidence_id_b: str
    explanation: str


class GateOut(BaseModel):
    """시도 1건의 **L1 층** 판정 — 기계 검사(LLM 호출 0회)의 결과."""

    verdict: str
    reject_reasons: list[str]


class JudgeOut(BaseModel):
    """시도 1건의 **L2 층** 판정 — 층 판정 + claim 단위 판정 + 근거쌍 모순."""

    verdict: str
    reject_reasons: list[str]
    claim_judgments: list[ClaimJudgmentOut]
    contradictions: list[ContradictionOut]


class AttemptOut(BaseModel):
    """초안 1건에 대한 판정. **종합과 층별을 함께** 싣는다.

    `verdict`/`reject_reasons` 는 기존 키이고 의미도 그대로 **종합**이다(값 집합에만
    L2 사유 2종이 더해졌다): 종합 pass ⟺ L1 pass 이고 L2 가 실행됐다면 L2 도 pass.

    `l1`/`l2` 는 층별 내역이며 **미실행은 `null`, 키는 항상 존재**한다. `l2` 가 `null` 인
    경우는 셋이다 — L1 reject(L2 는 L1 통과분에만 돈다) · 스위치 꺼짐 · L2 호출 실패.
    셋 다 "통과"가 아니라 "판정이 없었다"이므로, 종합 verdict 만 보고 통과로 읽으면 안
    된다(L2 호출 실패 시도는 층 결합 정의상 종합 pass 인데 문의는 인계된다).
    `l1` 이 `null` 인 것은 층별 컬럼이 없던 시절의 처리 기록을 복원한 경우뿐이다.
    """

    verdict: str
    reject_reasons: list[str]
    l1: GateOut | None
    l2: JudgeOut | None


class TokensOut(BaseModel):
    """토큰은 **계열별로 분리**한다 — provider 와 단가가 다르기 때문이다.

    `input`/`output` 은 기존 키·기존 의미 그대로 **생성 LLM 합산**이다(의도 해석 + SQL
    생성 + 초안 생성). `judge_input`/`judge_output` 은 판정 모델 토큰으로 **분리 신설**한
    키이며 **L2 미실행이면 0** 이다. 판정 토큰을 생성 합산에 섞으면 건당 비용 지표가
    무너진다 — 임베딩 토큰을 섞지 않는 것과 같은 이유다(임베딩은 지금도 응답에 싣지 않고
    처리 기록에만 남는다).

    `retrieval_input`/`retrieval_output` 은 **검색 단계 생성 호출**(질의 재작성)의 토큰이다.
    같은 규칙의 세 번째 계열이고 분리하는 이유가 특히 뾰족하다: 무근거 문의가 검색에서
    걸러져 **초안 없이** 끝나는 것이 이 사이클의 성공 판정인데, 재작성 토큰이 생성 칸에
    들어가면 초안을 만들지도 않은 문의가 초안 생성 토큰을 쓴 것으로 찍힌다.
    """

    input: int
    output: int
    judge_input: int
    judge_output: int
    retrieval_input: int
    retrieval_output: int


class MetricsOut(BaseModel):
    """`latency_ms` 는 파이프라인 벽시계 시간이다 — **L2 호출과 검색 단계 재작성 호출을
    포함**하고 처리 기록 저장은 포함하지 않는다."""

    latency_ms: int
    tokens: TokensOut
    #: 검색 단계가 폴백한 사유. `null` 은 "폴백하지 않았다"이고 **인계 사유가 아니다** —
    #: 폴백한 문의도 그대로 답변될 수 있다.
    retrieval_fallback_reason: str | None


def _gate_out(result: GateResult | None) -> GateOut | None:
    """L1 층 내역 → 응답. 층별 내역이 없는 기록(복원 경로)은 `None` 으로 남긴다."""
    if result is None:
        return None
    return GateOut(
        verdict=result.verdict.value,
        reject_reasons=[reason.value for reason in result.reject_reasons],
    )


def _judge_out(result: JudgeResult | None, mask: Callable[[str], str]) -> JudgeOut | None:
    """L2 층 내역 → 응답. **미실행이면 `None`** 이고 키는 그대로 남는다.

    claim 판정·모순쌍이 0건인 것과 L2 가 아예 돌지 않은 것은 다른 상태다 — 전자는 빈
    배열, 후자는 `null` 이다. 저장 층도 같은 구분을 지킨다(`records._l2_columns`).
    """
    if result is None:
        return None
    return JudgeOut(
        verdict=result.verdict.value,
        reject_reasons=[reason.value for reason in result.reject_reasons],
        claim_judgments=[
            ClaimJudgmentOut(
                claim_text=mask(judgment.claim_text),
                verdict=judgment.verdict.value,
                explanation=mask(judgment.explanation),
            )
            for judgment in result.claim_judgments
        ],
        contradictions=[
            ContradictionOut(
                evidence_id_a=contradiction.evidence_id_a,
                evidence_id_b=contradiction.evidence_id_b,
                explanation=mask(contradiction.explanation),
            )
            for contradiction in result.contradictions
        ],
    )


def _attempt_out(attempt: AttemptRecord, mask: Callable[[str], str]) -> AttemptOut:
    return AttemptOut(
        verdict=attempt.verdict.value,
        reject_reasons=[reason.value for reason in attempt.reject_reasons],
        l1=_gate_out(attempt.l1_result),
        l2=_judge_out(attempt.l2_result, mask),
    )


def _public_evidence(
    processed: ProcessedInquiry,
) -> tuple[list[CitationOut], Callable[[str], str]]:
    """원본 스냅샷을 읽기만 하고 응답용 사본에서만 공개 범위를 줄인다."""
    allowed = (
        processed.answer or ""
        if processed.status is InquiryStatus.ANSWERED
        and processed.verification_method != "product_confirm"
        else ""
    )
    snapshots = {snapshot.evidence_id: snapshot for snapshot in processed.sql_snapshots}
    private_columns: dict[str, frozenset[str]] = {}
    private_values: set[str] = set()
    unknown_sql = False
    for item in processed.evidence:
        if item.source is not EvidenceSource.SQL:
            continue
        snapshot = snapshots.get(item.id)
        if snapshot is None:
            unknown_sql = True
            continue
        try:
            columns = frozenset(private_output_columns(snapshot.query_sql))
        except SqlGuardRejection:
            unknown_sql = True
            continue
        private_columns[item.id] = columns
        private_values.update(
            str(value)
            for row in snapshot.result_rows
            for key, value in row.items()
            if key in columns and value is not None and str(value)
        )

    def mask(text: str) -> str:
        return redact_pii(text, allowed_text=allowed, private_values=tuple(private_values))

    citations: list[CitationOut] = []
    for item in processed.evidence:
        if item.source is EvidenceSource.POLICY:
            content = mask(item.content)
        elif item.id not in private_columns:
            # 예전 기록이나 분석 불가 SQL을 현재 주문 재조회/추정으로 복구하지 않는다.
            content = "[가림: 저장된 SQL 근거의 출력 유래를 확인할 수 없음]"
        else:
            snapshot = snapshots[item.id]
            columns = private_columns[item.id]
            rows = tuple(
                {
                    key: (
                        value
                        if key not in columns
                        and (value is None or isinstance(value, bool | int | float))
                        else mask("= " + str(value)).removeprefix("= ")
                        if value is not None
                        else None
                    )
                    for key, value in sorted(row.items())
                }
                for row in snapshot.result_rows
            )
            content = render_sql_rows(rows, mask_key=mask)
        citations.append(CitationOut(id=item.id, source=item.source.value, content=content))
    # 상품 확인의 오래된 분석 불가 기록은 L2 설명의 비패턴 값을 증명할 수 없다.
    if unknown_sql and processed.verification_method == "product_confirm":
        return citations, lambda text: "[가림]" if text else text
    return citations, mask


class InquiryResponse(BaseModel):
    """docs/contracts.md "공통 규약" 의 공통 골격. 값이 없을 때도 키는 사라지지 않는다.

    `POST /inquiries` 와 `GET /inquiries/{id}` 가 **같은 조립기**를 쓴다 — 조회는 저장된
    기록에서 복원한 `ProcessedInquiry` 를 넣을 뿐이라 층별 판정·판정 토큰까지 같은 모양으로
    다시 나온다.
    """

    inquiry_id: str
    status: str
    answer: str | None
    claims: list[ClaimOut]
    citations: list[CitationOut]
    attempts: list[AttemptOut]
    escalation_reason: str | None
    metrics: MetricsOut

    @classmethod
    def of(cls, processed: ProcessedInquiry) -> InquiryResponse:
        citations, mask = _public_evidence(processed)
        restrict_answer = processed.verification_method == "product_confirm"
        return cls(
            inquiry_id=processed.inquiry_id,
            status=processed.status.value,
            answer=(
                mask(processed.answer) if restrict_answer and processed.answer else processed.answer
            ),
            claims=[
                ClaimOut(
                    text=mask(claim.text) if restrict_answer else claim.text,
                    citation_ids=list(claim.citation_ids),
                )
                for claim in processed.claims
            ],
            citations=citations,
            attempts=[_attempt_out(attempt, mask) for attempt in processed.attempts],
            escalation_reason=(
                None if processed.escalation_reason is None else processed.escalation_reason.value
            ),
            metrics=MetricsOut(
                latency_ms=processed.latency_ms,
                tokens=TokensOut(
                    input=processed.input_tokens,
                    output=processed.output_tokens,
                    judge_input=processed.judge_input_tokens,
                    judge_output=processed.judge_output_tokens,
                    retrieval_input=processed.retrieval_input_tokens,
                    retrieval_output=processed.retrieval_output_tokens,
                ),
                retrieval_fallback_reason=processed.retrieval_fallback_reason,
            ),
        )


class HealthResponse(BaseModel):
    status: str


# ── 서비스 (접수 → 파이프라인 → 처리 기록) ─────────────────────────────────


class InquiryService:
    """요청 수명의 앱 커넥션 + 처리 시점에 여는 read-only 커넥션과 파이프라인.

    문의 ID 는 **근거 수집 전에** 확정한다: SQL 근거 ID 가 그 값을 품고, DB CHECK 가
    둘의 일치를 강제하기 때문이다.

    **파이프라인은 `process()` 가 처음 부를 때 조립한다.** 조회(`fetch`)는 파이프라인을
    한 번도 쓰지 않는데, 조립을 앞당기면 **접수와 무관한 설정 조합 오류가 조회 라우트를
    죽인다** — 이 모듈이 커넥션과 LLM 클라이언트에 이미 적용한 것과 같은 교훈이고
    (`get_service` · `_LazyGenerationClient`), τ 미검증 임베딩 조건에서 실제로
    `GET /inquiries/{id}` 가 500 이 됐다. 조립 시점 거부라는 성질은 그대로다 —
    접수 경로는 여전히 **데이터가 흐르기 전에** 죽는다.
    """

    def __init__(
        self,
        *,
        open_pipeline: Callable[[], InquiryPipeline],
        app_conn: psycopg.Connection[DictRow],
        open_readonly: Callable[[], AbstractContextManager[psycopg.Connection[DictRow]]],
    ) -> None:
        self._open_pipeline = open_pipeline
        self._app_conn = app_conn
        self._open_readonly = open_readonly

    def process(
        self, *, content: str, order_no: str | None, phone_last4: str | None = None
    ) -> ProcessedInquiry:
        with self._open_readonly() as readonly_conn:
            processed = self._open_pipeline().run(
                inquiry_id=new_inquiry_id(),
                content=content,
                order_no=order_no,
                app_conn=self._app_conn,
                readonly_conn=readonly_conn,
                phone_last4=phone_last4,
            )
        save_inquiry(conn=self._app_conn, processed=processed)
        return processed

    def confirm(self, *, inquiry_id: str, confirmed: bool) -> ProcessedInquiry:
        with self._open_readonly() as readonly_conn:
            processed = self._open_pipeline().confirm(
                inquiry_id=inquiry_id,
                confirmed=confirmed,
                app_conn=self._app_conn,
                readonly_conn=readonly_conn,
            )
        save_inquiry(conn=self._app_conn, processed=processed, update_pending=True)
        return processed

    def fetch(self, inquiry_id: str) -> ProcessedInquiry | None:
        return load_inquiry(conn=self._app_conn, inquiry_id=inquiry_id)


#: 요청 수명의 조립품을 **필요해진 순간에** 여는 것. 의존성 해석 시점이 아니다 —
#: 이유는 `get_service` 의 docstring 에 있다.
type ServiceOpener = Callable[[], AbstractContextManager[InquiryService]]


def _require_api_key(settings: Settings) -> SecretStr:
    """자격 증명은 **비밀 전용 타입 그대로** 넘긴다 — 평문으로 꺼내는 자리는
    호출 래퍼의 SDK 인자 한 줄뿐이다(`docs/security.md` "비밀 관리")."""
    if not settings.openai_api_key:
        raise MissingCredentialsError(
            "OPENAI_API_KEY 가 설정되지 않았다. `.env` 또는 환경 변수에 키를 넣고 다시 실행한다."
        )
    return settings.openai_api_key


class _LazyGenerationClient:
    """실제 클라이언트를 **첫 호출 때** 만드는 `GenerationClient` 어댑터.

    조립 시점에 만들면 자격 증명이 없는 환경에서 LLM 을 쓰지 않는 경로까지 무너진다
    (모듈 docstring 참조). 만들어진 클라이언트는 요청 수명 동안 재사용한다.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: GenerationClient | None = None

    def _resolve(self) -> GenerationClient:
        if self._client is None:
            self._client = OpenAIGenerationClient(
                api_key=_require_api_key(self._settings),
                model=self._settings.generation_model,
            )
        return self._client

    def complete_json(self, **kwargs: Any) -> JsonCompletion:
        return self._resolve().complete_json(**kwargs)


class _LazyEmbeddingClient:
    """`EmbeddingClient` 의 지연 생성판. `model`·`dimensions` 는 설정만으로 답한다."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: EmbeddingClient | None = None

    @property
    def model(self) -> str:
        return self._settings.embedding_model

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions

    def _resolve(self) -> EmbeddingClient:
        if self._client is None:
            self._client = OpenAIEmbeddingClient(
                api_key=_require_api_key(self._settings),
                model=self._settings.embedding_model,
                dimensions=self._settings.embedding_dimensions,
            )
        return self._client

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        return self._resolve().embed(stage=stage, texts=texts)


def build_generation_client(settings: Settings) -> GenerationClient:
    return _LazyGenerationClient(settings)


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    return _LazyEmbeddingClient(settings)


def get_service() -> ServiceOpener:
    """조립만 돌려준다 — **커넥션은 아직 열지 않는다.**

    의존성 해석 시점에 연결하면 DB 가 없을 때 **요청 오류와 설정 오류가 500 에 가린다**:
    형식이 틀린 주문번호(422)도, 판정 키 부재(503)도 커넥션이 먼저 터져 전부 500 이 됐다.
    라우트가 자기 선검사를 끝낸 뒤 `with open_service() as service` 로 여는 형태여야
    그 순서가 **프레임워크의 의존성 해석 순서와 무관하게** 성립한다.

    이 모듈은 같은 교훈을 LLM 클라이언트에는 이미 적용해 두었다(`_LazyGenerationClient`)
    — 커넥션도 같은 종류의 자원이다.

    테스트는 `app.dependency_overrides[get_service]` 로 이 함수를 통째로 갈아 끼운다 —
    LLM 목과 트랜잭션 커넥션을 주입하는 지점이다.
    """
    settings = get_settings()

    def _pipeline() -> InquiryPipeline:
        """**접수 경로가 실제로 부를 때** 조립한다 — 조회 라우트까지 끌고 가지 않는다."""
        return build_pipeline(
            generation_client=build_generation_client(settings),
            embedding_client=build_embedding_client(settings),
            settings=settings,
        )

    @contextmanager
    def _open() -> Iterator[InquiryService]:
        """앱 연결은 요청마다, RO 연결은 process 안에서 열고 닫는다."""
        with connect(settings=settings) as app_conn:
            yield InquiryService(
                open_pipeline=_pipeline,
                app_conn=app_conn,
                open_readonly=lambda: readonly_connect(settings=settings),
            )

    return _open


ServiceDep = Annotated[ServiceOpener, Depends(get_service)]


@app.exception_handler(RequestValidationError)
def request_validation_handler(request: Request, exc: RequestValidationError) -> Response:
    """오류 위치와 종류만 반환한다. input/ctx/원문 메시지는 민감 입력을 반사할 수 있다."""
    del request
    detail = [
        {"loc": error["loc"], "type": error["type"], "msg": "입력 형식을 확인하세요"}
        for error in exc.errors()
    ]
    return Response(
        status_code=422,
        media_type="application/json",
        content=json.dumps({"detail": detail}, ensure_ascii=True, allow_nan=False),
    )


@app.exception_handler(MissingCredentialsError)
def missing_credentials_handler(request: Request, exc: Exception) -> JSONResponse:
    """설정 오류는 503 — 처리 기록을 남기지 않는다 (인계로 집계되면 지표가 오염된다)."""
    del request
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(PolicyIndexEmptyError)
@app.exception_handler(PolicyIndexProvenanceError)
def stale_policy_index_handler(request: Request, exc: Exception) -> JSONResponse:
    """미적재·낡은 정책 인덱스는 **설정 오류(503)** 다 — 자격 증명 부재와 같은 계열이다.

    `no_evidence` 인계로 바꾸면 "검색이 못 찾았다"로 집계되어, 재색인을 빠뜨린 운영 실수가
    검색 품질 지표로 위장한다. 처리 기록은 커넥션 롤백으로 남지 않는다.
    """
    del request
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ── 라우트 ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def web_form(request: Request) -> HTMLResponse:
    """최소 웹 폼 1장. 대시보드가 아니라 **기각 순간을 보여주는 화면**이다."""
    return templates.TemplateResponse(request, "form.html")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _require_judge_credentials() -> None:
    """L2 가 켜져 있으면 판정 키를 **POST 처리 진입 시점에** 선검사한다.

    지연 검사(L2 에 도달해서야 실패)로 두면 두 가지가 무너진다: 생성 토큰을 태운 뒤에야
    503 이 나고, L1 이 2회 기각한 문의는 판정 키 없이도 정상 종결되어 "키 없으면 503"이
    조건부 규칙이 된다. 검증하지 못할 답변을 만들기 시작하지 않는 것이 fail-closed 다.

    **이 검사는 POST 경로에만 건다.** `get_service`(Depends)는 GET 라우트와 공유하므로
    거기에 넣으면 판정 키가 없는 환경에서 조회 전용 경로까지 죽는다
    (docs/engineering-notes.md "목만으로는 잡히지 않는 결함"). 접수 거부 422 도 판정 키와
    무관하다 — 그 경로는 이 함수에 닿기 전에 끝난다.
    """
    settings = get_settings()
    require_identity_credentials(settings)
    if settings.l2_enabled and not settings.anthropic_api_key:
        raise MissingCredentialsError(
            "ANTHROPIC_API_KEY 가 설정되지 않았다. L2 판정이 켜져 있으면 판정 없이 답변을 "
            "내보내지 않는다 — `.env` 또는 환경 변수에 키를 넣거나 L2_ENABLED=false 로 "
            "판정을 끄고 다시 실행한다."
        )


@app.post("/inquiries")
def create_inquiry(payload: InquiryRequest, open_service: ServiceDep) -> InquiryResponse:
    """문의 접수 + 동기 처리. 접수 검증 실패는 이 함수에 닿기 전에 422 로 끝난다.

    **커넥션은 선검사 뒤에 연다.** 먼저 열면 DB 가 없을 때 422·503 이 500 에 가린다.
    """
    # 처리 기록을 남기기 전에 끝낸다 — 설정 오류는 인계(`llm_call_failed`)가 아니다.
    _require_judge_credentials()
    with open_service() as service:
        processed = service.process(
            content=payload.content, order_no=payload.order_no, phone_last4=payload.phone_last4
        )
    return InquiryResponse.of(processed)


@app.post("/inquiries/{inquiry_id}/confirm")
def confirm_inquiry(
    inquiry_id: str, payload: ConfirmationRequest, open_service: ServiceDep
) -> InquiryResponse:
    _require_judge_credentials()
    try:
        with open_service() as service:
            processed = service.confirm(inquiry_id=inquiry_id, confirmed=payload.confirmed)
    except ConfirmationNotFound as exc:
        raise HTTPException(status_code=404, detail="그런 문의가 없다") from exc
    except ConfirmationConflict as exc:
        raise HTTPException(
            status_code=409, detail="확인 대기가 아니거나 확인 시간이 만료되었다"
        ) from exc
    return InquiryResponse.of(processed)


@app.get("/inquiries/{inquiry_id}")
def read_inquiry(inquiry_id: str, open_service: ServiceDep) -> InquiryResponse:
    """처리 기록 조회 — **DB 에 저장된 값**에서 같은 골격을 재구성한다."""
    with open_service() as service:
        processed = service.fetch(inquiry_id)
    if processed is None:
        raise HTTPException(status_code=404, detail="그런 문의가 없다")
    return InquiryResponse.of(processed)
