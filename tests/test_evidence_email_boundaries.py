"""SQL 별칭과 이메일 값의 경계가 서로 다른 근거를 같은 문면으로 만들지 않는다."""

from reply_gate.evidence import _sql_evidence_texts


def test_별칭의_등호와_이메일_값의_등호는_서로_다른_근거로_남는다() -> None:
    """a + b=user@…와 a=b + user@…가 같으면 L1도 주소를 구분할 수 없다."""
    rendered = []
    for key, address in (("a", "b=user@example.com"), ("a=b", "user@example.com")):
        _, evidence_text = _sql_evidence_texts(
            sql="표시용 조회문",
            rows=({key: address},),
            pii_safe_output_columns=(key,),
        )
        rendered.append(evidence_text)

    assert rendered[0] != rendered[1]
