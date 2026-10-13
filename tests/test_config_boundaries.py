"""잘못된 실행 설정은 무제한 SQL·빈 검색 결과가 되기 전에 거부한다."""

from __future__ import annotations

import secrets
import traceback
from typing import Any

import psycopg
import pytest
from pydantic import ValidationError

from reply_gate.db import connect, database_unavailable_reason, readonly_connect
from tests.conftest import declared_settings


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sql_statement_timeout_ms", 0),
        ("sql_statement_timeout_ms", -1),
        ("vector_top_k", 0),
        ("vector_top_k", -1),
        ("vector_similarity_threshold", -1.01),
        ("vector_similarity_threshold", 1.01),
        ("vector_similarity_threshold", float("nan")),
        ("vector_similarity_threshold", float("inf")),
        ("vector_similarity_threshold", float("-inf")),
        ("abstention_tau", -0.01),
        ("abstention_tau", float("nan")),
        ("abstention_tau", float("inf")),
        ("sql_max_rows", 0),
    ],
)
def test_실행_범위를_벗어난_설정을_입구에서_거부한다(name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        declared_settings(**{name: value})


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0])
def test_코사인_경계값과_양수_상한은_정상_설정이다(threshold: float) -> None:
    settings = declared_settings(
        vector_top_k=1,
        vector_similarity_threshold=threshold,
        abstention_tau=0,
        sql_statement_timeout_ms=1,
        sql_max_rows=1,
    )
    accepted = settings.vector_similarity_threshold == threshold
    assert accepted


def test_앱과_읽기전용_계정이_같으면_설정을_거부한다() -> None:
    with pytest.raises(ValidationError):
        declared_settings(postgres_app_user="same_role", postgres_ro_user="same_role")


def test_설정_검증_실패_문면에_다른_비밀_입력이_실리지_않는다() -> None:
    secret = secrets.token_urlsafe(24)
    with pytest.raises(ValidationError) as caught:
        declared_settings(
            postgres_app_user="same_role",
            postgres_ro_user="same_role",
            postgres_app_password=secret,
            postgres_ro_password=secret,
        )
    leaked = secret in "".join(traceback.format_exception(caught.value))
    assert not leaked


@pytest.mark.parametrize("name", ["", "x" * 64, "가" * 22, "reply_gate_readers"])
@pytest.mark.parametrize("field", ["postgres_app_user", "postgres_ro_user"])
def test_로그인_계정은_빈_이름이나_잘림이나_권한_그룹을_허용하지_않는다(
    field: str, name: str
) -> None:
    with pytest.raises(ValidationError):
        declared_settings(**{field: name})


@pytest.mark.parametrize("readonly", [False, True])
def test_잘못된_호스트_오류와_트레이스백과_skip에_비밀번호가_없다(readonly: bool) -> None:
    secret = secrets.token_urlsafe(24)
    settings = declared_settings(
        postgres_host="[",
        postgres_port=1,
        postgres_app_password=secret,
        postgres_ro_password=secret,
    )
    with pytest.raises(psycopg.Error) as caught:
        connect(settings=settings, readonly=readonly, connect_timeout=1)
    leaked = secret in str(caught.value) or secret in "".join(
        traceback.format_exception(caught.value)
    )
    reason = database_unavailable_reason(settings=settings, readonly=readonly)
    skipped_with_secret = secret in (reason or "")
    assert not leaked
    assert not skipped_with_secret
    assert reason is not None and "[" in reason and "docker compose" in reason


@pytest.mark.parametrize("readonly", [False, True])
def test_드라이버의_비밀_예외는_원인과_문맥_연결까지_제거한다(
    monkeypatch: pytest.MonkeyPatch,
    readonly: bool,
) -> None:
    """from None 은 표시만 숨긴다. 원래 오류를 순회하는 보고 도구에도 비밀이 없어야 한다."""
    secret = secrets.token_urlsafe(24)
    settings = declared_settings(postgres_app_password=secret, postgres_ro_password=secret)

    def fail(*args: Any, **kwargs: Any) -> None:
        try:
            raise ValueError(secret)
        except ValueError as original:
            raise psycopg.OperationalError(secret) from original

    monkeypatch.setattr(psycopg, "connect", fail)
    with pytest.raises(psycopg.OperationalError) as caught:
        connect(settings=settings, readonly=readonly)
    pending: list[BaseException] = [caught.value]
    observed: set[int] = set()
    leaked = False
    while pending:
        error = pending.pop()
        if id(error) in observed:
            continue
        observed.add(id(error))
        leaked |= secret in str(error) or secret in "".join(traceback.format_exception(error))
        pending.extend(x for x in (error.__cause__, error.__context__) if x is not None)
    reason = database_unavailable_reason(settings=settings, readonly=readonly)
    skip_leaked = secret in (reason or "")
    assert not leaked
    assert not skip_leaked
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("timeout", [0, -1])
def test_직접_연결도_SQL_시간_상한_해제를_거부한다(timeout: int) -> None:
    with pytest.raises(ValueError, match="statement_timeout"):
        connect(settings=declared_settings(postgres_port=1), statement_timeout_ms=timeout)


def test_읽기전용_연결이_문자열_해석과_시간_상한을_접속_옵션으로_고정한다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        observed.append(kwargs["options"])
        return None

    monkeypatch.setattr(psycopg, "connect", capture)
    readonly_connect(settings=declared_settings(sql_statement_timeout_ms=250))
    assert observed == ["-c statement_timeout=250 -c standard_conforming_strings=on"]
