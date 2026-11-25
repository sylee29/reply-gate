"""기존 리포트를 쓰지 않는 무과금 파일럿 검사: 결과는 stdout JSON뿐이다."""

from __future__ import annotations

import json

from reply_gate.evaluation_pilot import check_development, load_pilot


def main() -> int:
    report = check_development(load_pilot())
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    sql_matches = all(row.get("sql_guard", "accepted") == "accepted" for row in report["outcomes"])
    return 0 if report["l1_checked"] == report["l1_label_matches"] and sql_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
