#!/usr/bin/env bash
# E10 availability test (RFP §4; docs/OPERATIONS.md §5). Brings the platform stack up in its own compose project,
# runs the enroll/withdraw write loop of tests/test_failover.py and:
#   - ROUNDS x: kill the current primary; `failover.sh watch 1 3` detects it and promotes; RPO / RTO checked;
#     the dead node is rebuilt as a standby before the next round
#   - kill one standby: writes must continue (ANY 1); restart it
#   - kill both standbys: commits must block, none acknowledged (fail closed); restart one
# Output: reports/E10_availability/{results.json,summary.md}; raw logs in reports/raw/E10/ (gitignored).
#   scripts/test_failover.sh        env: ROUNDS=3, FA_PROJECT=fa-e10, KEEP=1 leaves the stack up
set -euo pipefail
cd "$(dirname "$0")/.."
export FA_PROJECT=${FA_PROJECT:-fa-e10} COMPOSE_FILE=docker-compose.yml:deploy/compose.platform.yml
ROUNDS=${ROUNDS:-3}
RAW=reports/raw/E10
OUT=reports/E10_availability
LOAD="$FA_PROJECT-e10-load"

dc() { docker compose -p "$FA_PROJECT" "$@"; }
ts() { date +%s.%N; }
log() { echo "[e10 $(date +%T)] $*" >&2; }
node() { echo "$FA_PROJECT-$1-1"; }
innet() {  # python inside the compose network, as the invoking user (files in the repo stay user-owned)
  docker run --rm --network "${FA_PROJECT}_internal" --dns-opt timeout:1 --dns-opt attempts:1 \
    -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/app" -w /app "$@"
}
cleanup() {
  docker rm -f "$LOAD" >/dev/null 2>&1 || true
  [ "${KEEP:-0}" = 1 ] || dc down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_ready() {  # 3 nodes, 2 quorum standbys on the primary, kms healthy
  for _ in $(seq 180); do
    if grep -q 'replicas=\[[^]]*quorum[^]]*quorum' <<< "$(scripts/failover.sh status 2>/dev/null)" &&
       [ "$(docker inspect -f '{{.State.Health.Status}}' "$(node kms)" 2>/dev/null)" = healthy ]; then return 0; fi
    sleep 1
  done
  log "cluster not ready"; scripts/failover.sh status >&2; return 1
}
start_load() { innet -d --name "$LOAD" faceauth:dev python tests/test_failover.py load "$1" --workers 4 >/dev/null; }
stop_load() {
  touch "$1/stop"
  timeout 90 docker wait "$LOAD" >/dev/null 2>&1 || docker rm -f "$LOAD" >/dev/null 2>&1 || true
}
acked_after() {  # number of acknowledged ops in $1 with ack time > $2
  python3 - "$1" "$2" <<'EOF'
import json, pathlib, sys
d, t = pathlib.Path(sys.argv[1]), float(sys.argv[2])
print(sum(1 for f in d.glob("ops-*.jsonl") for l in f.read_text().splitlines()
          if l.endswith("}") and (o := json.loads(l))["ok"] and o["t"] > t))
EOF
}
wait_acks() { for _ in $(seq 180); do [ "$(acked_after "$1" "$2")" -ge 20 ] && return 0; sleep 1; done; log "no writes after $2"; }
verify() { innet faceauth:dev python tests/test_failover.py verify "$1" > "$1/verify.out" || log "RPO CHECK FAILED in $1"; cat "$1/verify.out" >&2; }
meta() {  # $1 dir, $2 json fragment; merges failover.sh JSON lines when present
  python3 - "$1" "$2" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); m = json.loads(sys.argv[2]); fo = {}
f = d / "failover.jsonl"
if f.exists():
    for l in f.read_text().splitlines():
        if l.startswith("{"): fo.update(json.loads(l))
    m["failover"] = fo
(d / "meta.json").write_text(json.dumps(m, indent=1))
EOF
}

rm -rf "$RAW"; mkdir -p "$RAW" "$OUT"
printf '{"date": "%s", "docker": "%s", "postgres": "%s", "load_workers": 4, "kms_workers": 4, "detector": "failover.sh watch 1 3"}\n' \
  "$(date -Is)" "$(docker version --format '{{.Server.Version}}')" "$(docker run --rm postgres:16-alpine postgres --version)" > "$RAW/env.json"
dc down -v >/dev/null 2>&1 || true
log "starting platform stack ($FA_PROJECT)"
dc up -d db db-standby1 db-standby2 kms >/dev/null 2>&1
wait_ready

for r in $(seq "$ROUNDS"); do
  d="$RAW/r$r-primary"; mkdir -p "$d"
  old=$(scripts/failover.sh primary)
  log "round $r: load on, then kill primary $old"
  start_load "$d"; sleep 8
  scripts/failover.sh watch 1 3 > "$d/failover.jsonl" 2> "$d/failover.log" & wpid=$!
  sleep 2
  t_kill=$(ts); docker kill "$(node "$old")" >/dev/null; t_dead=$(ts)
  wait "$wpid" || log "failover.sh failed (see $d/failover.log)"
  wait_acks "$d" "$t_dead"; sleep 3
  stop_load "$d"
  meta "$d" "{\"kind\": \"primary\", \"scenario\": \"kill primary $old\", \"killed\": \"$old\", \"t_kill\": $t_kill, \"t_dead\": $t_dead}"
  verify "$d"
  scripts/failover.sh rebuild "$old" 2>> "$d/failover.log"
  wait_ready
done

p=$(scripts/failover.sh primary)
sb=(); for n in db db-standby1 db-standby2; do [ "$n" != "$p" ] && sb+=("$n"); done

d="$RAW/r$((ROUNDS + 1))-standby"; mkdir -p "$d"
log "kill one standby (${sb[0]}) under load: writes must continue"
start_load "$d"; sleep 5
t_kill=$(ts); docker kill "$(node "${sb[0]}")" >/dev/null; t_dead=$(ts)
sleep 10
t_restart=$(ts); docker start "$(node "${sb[0]}")" >/dev/null
sleep 5
stop_load "$d"
meta "$d" "{\"kind\": \"standby\", \"scenario\": \"kill standby ${sb[0]}\", \"t_kill\": $t_kill, \"t_dead\": $t_dead, \"t_restart\": $t_restart}"
verify "$d"
wait_ready

d="$RAW/r$((ROUNDS + 2))-both-standbys"; mkdir -p "$d"
log "kill both standbys under load: commits must block (fail closed), none acknowledged"
start_load "$d"; sleep 5
docker kill "$(node "${sb[0]}")" >/dev/null
t_kill=$(ts); docker kill "$(node "${sb[1]}")" >/dev/null; t_dead=$(ts)
sleep 8
t_restart=$(ts); docker start "$(node "${sb[0]}")" >/dev/null
wait_acks "$d" "$t_restart"; sleep 3
stop_load "$d"
meta "$d" "{\"kind\": \"standby\", \"scenario\": \"kill both standbys\", \"t_kill\": $t_kill, \"t_dead\": $t_dead, \"t_restart\": $t_restart}"
verify "$d"
docker start "$(node "${sb[1]}")" >/dev/null
wait_ready

innet faceauth:dev python tests/test_failover.py report "$RAW" "$OUT"
log "done: $OUT/summary.md"
cat "$OUT/summary.md"
