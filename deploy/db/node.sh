#!/bin/sh
# Entrypoint of every Postgres node (db, db-standby1, db-standby2); runs as OS user postgres.
# PGDATA empty:
#   FA_PG_INIT=1 (service db only): initdb as the first primary via the stock entrypoint + init-primary.sh,
#                refused if any peer already answers (never create a second, empty primary).
#   otherwise:   clone the current primary $FA_PG_PRIMARY with pg_basebackup and start as its standby.
# PGDATA present: start as whatever it is (standby.signal => standby). See docs/OPERATIONS.md.
# application_name of a standby = its cluster_name (walreceiver fallback), so the clone's own WAL stream
# ("pg_basebackup") never counts as a synchronous standby.
set -eu
umask 077
# Replication password for pg_basebackup / primary_conninfo; tmpfs only, never in the data volume.
printf '*:*:*:replicator:%s\n' "$(cat /run/secrets/db_repl_password)" > /tmp/pgpass

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  if [ "${FA_PG_INIT:-0}" = 1 ]; then
    for peer in ${FA_PG_PEERS:-}; do
      rc=0; pg_isready -q -h "$peer" -t 2 || rc=$?
      if [ "$rc" -le 1 ]; then
        echo "node.sh: refusing initdb: peer $peer is up; rebuild this node with scripts/failover.sh rebuild" >&2
        exit 1
      fi
    done
  else
    until pg_basebackup -D "$PGDATA" -X stream -c fast -R \
        -d "host=${FA_PG_PRIMARY:?} port=5432 user=replicator passfile=/tmp/pgpass connect_timeout=5"; do
      echo "node.sh: clone from $FA_PG_PRIMARY failed; retrying in 2s" >&2
      rm -rf "$PGDATA"
      sleep 2
    done
  fi
fi
exec docker-entrypoint.sh "$@"
