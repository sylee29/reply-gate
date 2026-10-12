"""표시용 개인정보 유래는 출력 별칭을 넘어 네 주문 열에서 이어진다."""

import pytest

from reply_gate import sql_guard


@pytest.mark.parametrize(
    "projection, expected",
    [
        (
            "customer_name AS recipient, shipping_address AS destination, status",
            {"recipient", "destination"},
        ),
        ("upper(customer_name) AS recipient, quantity + 1 AS amount", {"recipient"}),
        ("*", {"customer_name", "customer_phone", "customer_email", "shipping_address"}),
        ("coalesce(customer_name, '미정') AS recipient, status", {"recipient"}),
    ],
)
def test_별칭과_계산_개인정보_출력만_식별한다(projection: str, expected: set[str]) -> None:
    assert set(sql_guard.private_output_columns(f"SELECT {projection} FROM orders")) == expected


@pytest.mark.parametrize(
    "query",
    [
        "WITH x AS (SELECT customer_name AS n, status FROM orders) SELECT n AS recipient, status FROM x",
        "SELECT n AS recipient, status FROM (SELECT customer_name AS n, status FROM orders) x",
    ],
)
def test_CTE와_파생테이블의_개인정보_별칭도_이어진다(query: str) -> None:
    assert sql_guard.private_output_columns(query) == ("recipient",)


@pytest.mark.parametrize("query", ["", "DELETE FROM orders", "SELECT secret FROM unknown"])
def test_해석할수없는_저장_SQL은_무가림으로_추정하지_않는다(query: str) -> None:
    with pytest.raises(sql_guard.SqlGuardRejection):
        sql_guard.private_output_columns(query)


@pytest.mark.parametrize(
    "projection",
    [
        "CASE WHEN customer_name = '홍길동' THEN customer_name ELSE status END AS x",
        "concat(customer_name, shipping_address) AS x",
        "CAST(customer_name AS text) AS x",
    ],
)
def test_조건과_계산식_안의_개인정보도_보수적으로_가린다(projection: str) -> None:
    assert sql_guard.private_output_columns(f"SELECT {projection} FROM orders") == ("x",)


def test_긴_출력명은_Postgres의_바이트_경계로_복원한다() -> None:
    name = "긴이름" * 12
    expected = name.encode()[:63].decode()
    assert sql_guard.private_output_columns(f'SELECT customer_name AS "{name}" FROM orders') == (
        expected,
    )


def test_동명_출력과_기저열_이름교체는_추정하지_않는다() -> None:
    for query in (
        "SELECT customer_name AS x, status AS x FROM orders",
        "SELECT x FROM orders o(x)",
    ):
        with pytest.raises(sql_guard.SqlGuardRejection):
            sql_guard.private_output_columns(query)


def test_CTE_위치별_이름교체는_개인정보가_있으면_전부_가린다() -> None:
    assert sql_guard.private_output_columns(
        "WITH x(a,b) AS (SELECT customer_name,status FROM orders) SELECT a,b FROM x"
    ) == ("a", "b")
