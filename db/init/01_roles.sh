#!/bin/bash
# 컨테이너 최초 기동 시 1회 실행된다 (docker-entrypoint-initdb.d, superuser 권한).
#
# 계정 2개를 만든다 (docs/security.md "text-to-SQL 안전장치" 1층 — 계정 분리):
#   * 앱 계정      — public 스키마에 객체를 만들고 읽고 쓴다.
#   * read-only 계정 — NOLOGIN 그룹 `reply_gate_readers` 의 멤버로서 SELECT 만 상속받는다.
#     테이블 단위 SELECT 부여는 `db/schema.sql` 이 그 그룹에 한다(계정 이름이 환경 변수라
#     정적 DDL 에 적을 수 없기 때문 — 그룹 이름은 비밀이 아니므로 고정한다).
#
# 비밀은 환경 변수에서만 온다. 이 파일에는 값이 없다. psql 의 `:'var'` 는 값을 SQL 문자열
# 리터럴로 안전하게 인용하므로 비밀번호에 특수문자가 있어도 깨지지 않는다.
set -euo pipefail

: "${POSTGRES_DB:?POSTGRES_DB 가 필요하다}"
: "${POSTGRES_APP_USER:?POSTGRES_APP_USER 가 필요하다}"
: "${POSTGRES_APP_PASSWORD:?POSTGRES_APP_PASSWORD 가 필요하다 (.env 에 채운다)}"
: "${POSTGRES_RO_USER:?POSTGRES_RO_USER 가 필요하다}"
: "${POSTGRES_RO_PASSWORD:?POSTGRES_RO_PASSWORD 가 필요하다 (.env 에 채운다)}"

if [[ "$POSTGRES_APP_USER" == "$POSTGRES_RO_USER" ||
      "$POSTGRES_APP_USER" == "$POSTGRES_USER" ||
      "$POSTGRES_RO_USER" == "$POSTGRES_USER" ||
      "$POSTGRES_APP_USER" == "reply_gate_readers" ||
      "$POSTGRES_RO_USER" == "reply_gate_readers" ]]; then
    printf '%s\n' 'DB 초기화 거부: 앱·읽기전용·관리자·권한 그룹의 이름이 충돌한다' >&2
    exit 1
fi
# PostgreSQL 이 이름을 63바이트에서 자르면 서로 다른 설정도 같은 계정이 된다.
for role in "$POSTGRES_APP_USER" "$POSTGRES_RO_USER"; do
    if (( $(LC_ALL=C printf '%s' "$role" | wc -c) > 63 )); then
        printf '%s\n' 'DB 초기화 거부: 계정 이름은 63바이트 이하여야 한다' >&2
        exit 1
    fi
done

# 비밀번호는 argv 나 SQL 원문에 확장하지 않는다. psql 내부에서 환경을 읽는다.
# 오류 SQL 을 psql 또는 서버 로그가 되찍어 자격 증명을 남기지 않게 두 출력을 모두 닫는다.
if ! psql -X -q -v ON_ERROR_STOP=1 -v VERBOSITY=terse \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" >/dev/null 2>/dev/null <<'SQL'
SET log_statement = 'none';
SET log_min_error_statement = 'panic';
\set ECHO none
\set ECHO_HIDDEN off
\getenv db_name POSTGRES_DB
\getenv app_user POSTGRES_APP_USER
\getenv app_password POSTGRES_APP_PASSWORD
\getenv ro_user POSTGRES_RO_USER
\getenv ro_password POSTGRES_RO_PASSWORD

-- read-only 그룹 (로그인 불가, 비밀 아님 → 이름 고정).
SELECT 'CREATE ROLE reply_gate_readers NOLOGIN'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'reply_gate_readers')
\gexec

-- 애플리케이션 계정 (읽기/쓰기 + 스키마 생성).
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'app_user', :'app_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user')
\gexec

-- text-to-SQL 실행 전용 계정 (SELECT 만 — 안전장치 1).
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'ro_user', :'ro_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'ro_user')
\gexec

SELECT format('GRANT reply_gate_readers TO %I', :'ro_user')
\gexec

-- DB 접속: PUBLIC 에서 회수한다. TEMP 테이블 생성 권한(PUBLIC 기본값)도 함께 사라지므로
-- read-only 계정은 임시 테이블조차 만들 수 없다.
REVOKE ALL ON DATABASE :"db_name" FROM PUBLIC;
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', :'db_name', :'app_user')
\gexec
GRANT CONNECT ON DATABASE :"db_name" TO reply_gate_readers;

-- public 스키마: 앱 계정만 객체를 만든다. read-only 는 참조(USAGE)만 한다.
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO reply_gate_readers;
SELECT format('GRANT USAGE, CREATE ON SCHEMA public TO %I', :'app_user')
\gexec

SQL
then
    printf '%s\n' 'DB 계정 초기화 실패: 환경 설정과 계정 권한을 확인한다 (비밀 포함 진단 생략)' >&2
    exit 1
fi
