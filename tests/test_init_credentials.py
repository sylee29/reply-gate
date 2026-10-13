"""초기화 프로세스의 argv·오류 출력에는 DB 비밀번호를 싣지 않는다."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def init_environment(tmp_path: Path) -> dict[str, str]:
    """psql 대역은 실행 인자만 기록하고 비밀은 환경에만 둔다."""
    probe = tmp_path / "psql"
    probe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['PROBE_OUT']).write_text(json.dumps(sys.argv[1:]))\n"
        "sql = sys.stdin.read()\n"
        "Path(os.environ['PROBE_SQL']).write_text(sql)\n"
        "if os.environ.get('PROBE_FAIL'):\n"
        "    print(os.environ['POSTGRES_APP_PASSWORD'], file=sys.stderr)\n"
        "    sys.exit(1)\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)
    return {
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "POSTGRES_USER": "admin_role",
        "POSTGRES_DB": "probe_db",
        "POSTGRES_APP_USER": "app_role",
        "POSTGRES_RO_USER": "ro_role",
        "POSTGRES_APP_PASSWORD": secrets.token_urlsafe(24),
        "POSTGRES_RO_PASSWORD": secrets.token_urlsafe(24),
        "PROBE_OUT": str(tmp_path / "argv.json"),
        "PROBE_SQL": str(tmp_path / "input.sql"),
    }


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROOT / "db/init/01_roles.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_초기화_비밀번호가_argv와_입력_SQL에_들어가지_않는다(
    init_environment: dict[str, str],
) -> None:
    result = _run(init_environment)
    argv = json.loads(Path(init_environment["PROBE_OUT"]).read_text())
    sql = Path(init_environment["PROBE_SQL"]).read_text()
    secrets_in_output = any(
        init_environment[key] in repr(argv) + sql + result.stdout + result.stderr
        for key in ("POSTGRES_APP_PASSWORD", "POSTGRES_RO_PASSWORD")
    )
    assert result.returncode == 0
    assert not secrets_in_output
    assert "\\getenv app_password POSTGRES_APP_PASSWORD" in sql
    assert "\\getenv ro_password POSTGRES_RO_PASSWORD" in sql


def test_초기화_SQL_실패_출력에도_비밀번호가_없다(init_environment: dict[str, str]) -> None:
    init_environment["PROBE_FAIL"] = "1"
    result = _run(init_environment)
    leaked = init_environment["POSTGRES_APP_PASSWORD"] in result.stdout + result.stderr
    assert result.returncode != 0
    assert not leaked
    assert "초기화" in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("POSTGRES_RO_USER", "app_role"),
        ("POSTGRES_APP_USER", "admin_role"),
        ("POSTGRES_RO_USER", "admin_role"),
        ("POSTGRES_APP_USER", "reply_gate_readers"),
        ("POSTGRES_RO_USER", "reply_gate_readers"),
        ("POSTGRES_RO_USER", "x" * 64),
        ("POSTGRES_APP_USER", "가" * 22),
    ],
)
def test_같은_계정이나_잘리는_이름이면_psql_전에_초기화를_거부한다(
    init_environment: dict[str, str],
    field: str,
    value: str,
) -> None:
    init_environment[field] = value
    result = _run(init_environment)
    assert result.returncode != 0
    assert not Path(init_environment["PROBE_OUT"]).exists()
