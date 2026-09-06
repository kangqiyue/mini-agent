#!/bin/bash
# Restart the existing local gateway without creating or reconfiguring services.
set -euo pipefail

if [[ "${1:-}" == "--help" && "$#" -eq 1 ]]; then
    cat <<'HELP'
Usage: bash start.sh

Start the existing Colima VM and the db/litellm Compose containers, then wait
for http://127.0.0.1:4000/health/liveliness. Repeated runs are safe.

Environment overrides:
  LITELLM_COMPOSE_DIR             Default: $HOME/work/litellm_config
  LITELLM_STARTUP_TIMEOUT_SECONDS Default: 120 (integer from 1 to 600)

Uses Docker context "colima". Does not create containers, change gateway
configuration, install auto-start, or require API credentials.
HELP
    exit 0
fi

fail() {
    printf 'Gateway startup failed: %s\n' "$1" >&2
    exit 1
}

if [[ "$#" -ne 0 ]]; then
    printf 'Unknown arguments. Use --help for usage.\n' >&2
    exit 2
fi

export PATH="${PATH:-}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
compose_dir="${LITELLM_COMPOSE_DIR:-${HOME}/work/litellm_config}"
timeout_seconds="${LITELLM_STARTUP_TIMEOUT_SECONDS:-120}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]{0,2}$ ]] || ((timeout_seconds > 600)); then
    fail 'LITELLM_STARTUP_TIMEOUT_SECONDS must be an integer from 1 to 600.'
fi

for executable in colima docker curl; do
    command -v "$executable" >/dev/null 2>&1 || fail "Missing executable: $executable"
done
[[ -d "$compose_dir" ]] || fail 'The configured Compose directory does not exist.'
cd -- "$compose_dir"

if colima status >/dev/null 2>&1; then
    printf 'Colima is already running.\n'
else
    printf 'Starting Colima...\n'
    colima start >/dev/null 2>&1 || fail 'Colima could not start; inspect colima status.'
fi

docker --context colima info >/dev/null 2>&1 || fail 'The local Docker daemon is unavailable.'
services="$(docker --context colima compose ps -a --services 2>/dev/null)" \
    || fail 'Could not inspect the existing Compose project.'
has_db=false
has_litellm=false
while IFS= read -r service; do
    case "$service" in
        db) has_db=true ;;
        litellm) has_litellm=true ;;
    esac
done <<< "$services"
if [[ "$has_db" != true || "$has_litellm" != true ]]; then
    fail 'Existing db/litellm containers are missing; check the gateway installation.'
fi

printf 'Starting the existing database and LiteLLM containers...\n'
docker --context colima compose start db litellm >/dev/null 2>&1 \
    || fail 'Compose startup failed; inspect docker --context colima compose ps -a.'

printf 'Waiting for the local gateway...\n'
deadline=$((SECONDS + timeout_seconds))
while ! curl --noproxy '*' --fail --silent --connect-timeout 2 --max-time 3 \
    --output /dev/null 'http://127.0.0.1:4000/health/liveliness'; do
    if ((SECONDS >= deadline)); then
        fail 'The gateway did not become healthy before the timeout; inspect Compose status.'
    fi
    sleep 1
done
printf 'Gateway ready: http://127.0.0.1:4000/v1\n'
