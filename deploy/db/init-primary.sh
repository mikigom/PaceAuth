#!/bin/sh
# First boot of the initial primary only (run by the stock postgres entrypoint from /docker-entrypoint-initdb.d,
# as OS user postgres over the unix socket). Roles per docs/IMPLEMENTATION.md §11; passwords from /run/secrets.
set -eu
psql -v ON_ERROR_STOP=1 -X -q -d postgres <<'SQL'
\set api `cat /run/secrets/db_api_password`
\set ro `cat /run/secrets/db_ro_password`
\set kms `cat /run/secrets/db_kms_password`
\set repl `cat /run/secrets/db_repl_password`
CREATE ROLE faceauth_api LOGIN PASSWORD :'api';
CREATE ROLE faceauth_ro LOGIN PASSWORD :'ro';
CREATE ROLE kms LOGIN PASSWORD :'kms';
CREATE ROLE replicator LOGIN REPLICATION PASSWORD :'repl';
CREATE DATABASE faceauth OWNER faceauth_api;
CREATE DATABASE kms OWNER kms;
REVOKE ALL ON DATABASE faceauth, kms FROM PUBLIC;
GRANT CONNECT ON DATABASE faceauth TO faceauth_ro;
-- Effective once the server restarts after init; scripts/failover.sh rewrites it on the new primary.
ALTER SYSTEM SET synchronous_standby_names = 'ANY 1 (db_standby1, db_standby2)';
SQL
# Objects are created AS their owners so the api / kms roles own their tables.
psql -v ON_ERROR_STOP=1 -X -q -d faceauth <<'SQL'
SET ROLE faceauth_api;
\i /etc/faceauth-db/faceauth.sql
GRANT SELECT ON templates, template_events, model_versions TO faceauth_ro;
SQL
psql -v ON_ERROR_STOP=1 -X -q -d kms <<'SQL'
SET ROLE kms;
\i /etc/faceauth-db/kms.sql
SQL
