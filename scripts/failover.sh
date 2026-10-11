#!/usr/bin/env bash
# Failover tooling for the 3-node Postgres cluster of deploy/compose.platform.yml (docs/OPERATIONS.md §5).
#
#   scripts/failover.sh status                 role / LSN / sync state of every node
#   scripts/failover.sh primary                print the current primary (exit 1 if none or several)
#   scripts/failover.sh promote [old_primary]  fence the dead primary, promote the most advanced standby,
#                                              re-point the other standby, require 1 sync standby again
#   scripts/failover.sh watch [interval] [n]   probe the primary; after n consecutive failures run promote
#   scripts/failover.sh rebuild <node>         re-clone a fenced / failed node as standby of the current primary
#
# Env: FA_PROJECT (compose project, default faceauth); COMPOSE_FILE (default: .env in the repo root);
#      FORCE=1 lets promote fence and replace a primary that still answers.
# promote prints one JSON line (timings, unix seconds) on stdout; everything else goes to stderr.
#
# Sync rule: the new primary gets synchronous_standby_names = 'ANY 1 (<the other two nodes>)' before it
# accepts its first write. If no standby is left, commits block (fail closed) - never relaxed (RFP §4, OPS-01).
set -euo pipefail
cd "$(dirname "$0")/.."
P=${FA_PROJECT:-faceauth}
NODES=(db db-standby1 db-standby2)

dc() { docker compose -p "$P" "$@"; }
c() { echo "$P-$1-1"; }                         # compose container name of a node
app() { echo "${1//-/_}"; }                     # application_name (= cluster_name) of a node
ts() { date +%s.%N; }
log() { echo "[failover $(date +%T.%3N)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }
sql() { timeout 10 docker exec "$(c "$1")" psql -X -qAt -v ON_ERROR_STOP=1 -c "$2"; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$(c "$1")" 2>/dev/null)" = true ]; }
role() {  # primary | standby | down
  if running "$1" && timeout 5 docker exec "$(c "$1")" pg_isready -q -h /var/run/postgresql -t 2 &&
     r=$(sql "$1" "select pg_is_in_recovery()" 2>/dev/null); then
    [ "$r" = f ] && echo primary || echo standby
  else
    echo down
  fi
}
primary() {
  local n p=()
  for n in "${NODES[@]}"; do [ "$(role "$n")" = primary ] && p+=("$n"); done
  [ ${#p[@]} -eq 1 ] || { log "primaries found: ${p[*]:-none}"; return 1; }
  echo "${p[0]}"
}
upstream() {  # host in a standby's primary_conninfo
  sql "$1" "show primary_conninfo" | sed -n "s/.*host='\\{0,1\\}\\([^ ']*\\).*/\\1/p"
}
lsn() {  # bytes of WAL this standby has durably received (or replayed)
  sql "$1" "select (coalesce(greatest(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()), '0/0') - '0/0')::bigint"
}

status() {
  local n
  for n in "${NODES[@]}"; do
    case $(role "$n") in
      primary) echo "$n primary lsn=$(sql "$n" "select pg_current_wal_lsn()") $(sql "$n" "show synchronous_standby_names")" \
                    "replicas=[$(sql "$n" "select string_agg(application_name || ':' || state || ':' || sync_state, ' ') from pg_stat_replication")]" ;;
      standby) echo "$n standby lsn=$(lsn "$n") upstream=$(upstream "$n")" ;;
      *) echo "$n down" ;;
    esac
  done
}

promote() {
  local t0 t_fence t_promote t_sync old best="" best_lsn=-1 n l others pick=""
  t0=$(ts)
  old=${1:-}
  if [ -z "$old" ]; then  # the node the surviving standbys stream from
    for n in "${NODES[@]}"; do [ "$(role "$n")" = standby ] && old=$(upstream "$n") && [ -n "$old" ] && break; done
  fi
  [ -n "$old" ] || die "cannot determine the old primary; pass it explicitly"
  if [ "$(role "$old")" = primary ] && [ "${FORCE:-0}" != 1 ]; then
    die "$old still answers as primary; refusing (FORCE=1 fences it anyway)"
  fi
  # 1. Fence: the old primary must never accept a write again. No restart, killed now; its storage gets
  #    standby.signal after the promotion (step 6), so even a manual `compose up` brings it back read-only.
  docker update --restart=no "$(c "$old")" >/dev/null 2>&1 || log "warn: no container for $old (fence the host!)"
  docker kill "$(c "$old")" >/dev/null 2>&1 || true
  t_fence=$(ts)
  # 2. Every acknowledged commit is on >= 1 standby (ANY 1) and WAL is one stream, so the standby with the
  #    highest durable LSN has all of them. Promotion replays everything it has received first.
  #    Only standbys of the dead primary count: a fenced ex-primary started read-only may hold diverged WAL.
  for n in "${NODES[@]}"; do
    [ "$n" = "$old" ] && continue
    [ "$(role "$n")" = standby ] || { log "$n is not a live standby"; continue; }
    [ "$(upstream "$n")" = "$old" ] || { log "$n does not stream from $old; skipped (rebuild it)"; continue; }
    l=$(lsn "$n"); log "$n lsn=$l"; pick="$pick\"$n\": $l, "
    if [ "$l" -gt "$best_lsn" ]; then best=$n; best_lsn=$l; fi
  done
  [ -n "$best" ] || die "no live standby to promote (writes stay blocked; rebuild a node from backup)"
  # 3. Sync requirement before the first write: ANY 1 of the other two nodes (incl. the old primary, so it
  #    counts again once rebuilt). The inherited value already names >= 1 other node, so there is no window.
  others=""
  for n in "${NODES[@]}"; do [ "$n" = "$best" ] || others+="$(app "$n"), "; done
  others=${others%, }
  sql "$best" "ALTER SYSTEM SET synchronous_standby_names = 'ANY 1 ($others)'" >/dev/null
  sql "$best" "ALTER SYSTEM RESET primary_conninfo" >/dev/null
  sql "$best" "select pg_reload_conf()" >/dev/null
  [ "$(sql "$best" "select pg_promote(true, 60)")" = t ] || die "pg_promote failed on $best"
  t_promote=$(ts)
  log "promoted $best (sync: ANY 1 ($others))"
  # 4. Re-point the remaining standby(s); recovery_target_timeline=latest makes them follow the new timeline.
  for n in "${NODES[@]}"; do
    if [ "$n" = "$best" ] || [ "$n" = "$old" ] || [ "$(role "$n")" != standby ] || [ "$(upstream "$n")" != "$old" ]; then
      continue
    fi
    sql "$n" "ALTER SYSTEM SET primary_conninfo = 'host=$best port=5432 user=replicator passfile=/tmp/pgpass connect_timeout=5'" >/dev/null
    sql "$n" "select pg_reload_conf()" >/dev/null
    log "re-pointed $n -> $best"
  done
  # 5. Commits are acknowledged only once a standby is attached synchronously.
  for _ in $(seq 300); do
    [ "$(sql "$best" "select count(*) from pg_stat_replication where sync_state in ('sync', 'quorum')")" -ge 1 ] && break
    sleep 0.2
  done
  t_sync=$(ts)
  [ "$(sql "$best" "select count(*) from pg_stat_replication where sync_state in ('sync', 'quorum')")" -ge 1 ] ||
    log "WARNING: no sync standby attached; writes on $best block until one is (fail closed). Rebuild a node."
  # 6. Storage fence (after promotion: not on the RTO path).
  dc run --rm --no-deps -T --entrypoint sh "$old" -c 'test -d "$PGDATA" && touch "$PGDATA/standby.signal"' \
    >/dev/null 2>&1 || log "warn: could not write standby.signal for $old; it must not be started before rebuild"
  log "done: $old -> $best. Next: reload every search replica (docs/OPERATIONS.md §5.3), then rebuild $old"
  printf '{"old": "%s", "new": "%s", "lsn": {%s}, "t_start": %s, "t_fenced": %s, "t_promoted": %s, "t_sync": %s}\n' \
    "$old" "$best" "${pick%, }" "$t0" "$t_fence" "$t_promote" "$t_sync"
}

watch() {
  local interval=${1:-1} need=${2:-3} fails=0 p
  p=$(primary) || die "no single healthy primary to watch"
  log "watching $p every ${interval}s; failover after $need consecutive failures"
  while :; do
    if [ "$(role "$p")" = primary ]; then fails=0; else fails=$((fails + 1)); log "$p probe failed ($fails/$need)"; fi
    [ "$fails" -ge "$need" ] && break
    sleep "$interval"
  done
  printf '{"detected": %s}\n' "$(ts)"
  promote "$p"
}

rebuild() {
  local n=${1:?node} p
  p=$(primary) || die "no primary to clone from"
  [ "$n" != "$p" ] || die "$n is the current primary"
  log "rebuilding $n from $p (its old data is discarded)"
  dc rm -sf "$n" >/dev/null 2>&1 || true
  dc run --rm --no-deps -T -e FA_PG_INIT=0 -e FA_PG_PRIMARY="$p" --entrypoint sh "$n" \
    -c 'rm -rf "$PGDATA" && exec /etc/faceauth-db/node.sh true' >&2
  dc up -d --no-deps "$n" >&2
  for _ in $(seq 150); do
    [ "$(sql "$p" "select count(*) from pg_stat_replication where application_name = '$(app "$n")' and state = 'streaming'" 2>/dev/null)" = 1 ] &&
      { log "$n streaming from $p"; return 0; }
    sleep 1
  done
  die "$n did not start streaming from $p"
}

case ${1:-status} in
  status) status ;;
  primary) primary ;;
  promote) promote "${2:-}" ;;
  watch) watch "${2:-1}" "${3:-3}" ;;
  rebuild) rebuild "${2:-}" ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
