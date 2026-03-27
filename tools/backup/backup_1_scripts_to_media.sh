#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_PARENT="$(dirname "${REPO_ROOT}")"

SRC_DEFAULT="${REPO_ROOT}"
DEST_DEFAULT="/media/synnaseet_messdaten/MG_Farm/1_Scripts"
LOG_DIR_DEFAULT="${REPO_PARENT}/backup_logs"
LOCK_FILE_DEFAULT="/tmp/backup_1_scripts_to_media.lock"

SRC="${SRC:-$SRC_DEFAULT}"
DEST="${DEST:-$DEST_DEFAULT}"
LOG_DIR="${LOG_DIR:-$LOG_DIR_DEFAULT}"
LOCK_FILE="${LOCK_FILE:-$LOCK_FILE_DEFAULT}"
MODE="${MODE:-incremental}"
VERBOSE="${VERBOSE:-0}"

usage() {
  cat <<'EOF'
Usage:
  backup_1_scripts_to_media.sh [--mirror] [--verbose] [--source PATH] [--dest PATH]

Description:
  Creates a CIFS-safe rsync backup of /home/.../MG_Farm/1_Scripts to the media folder.
  The source is never modified. By default the backup is incremental and does not delete
  files on the destination. Use --mirror only when you explicitly want the destination
  to match the source.

Options:
  --mirror        Delete destination files that no longer exist in the source.
  --verbose       Print rsync file-level progress in addition to the log file.
  --source PATH   Override the source directory.
  --dest PATH     Override the destination directory.
  -h, --help      Show this help.

Environment overrides:
  SRC, DEST, LOG_DIR, LOCK_FILE, MODE, VERBOSE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mirror)
      MODE="mirror"
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    --source)
      SRC="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_FILE="${LOG_DIR}/backup_1_scripts_${TIMESTAMP}.log"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another backup is already running. Lock: ${LOCK_FILE}" | tee -a "${LOG_FILE}"
  exit 3
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}"
}

fail() {
  log "ERROR: $*"
  exit 1
}

[[ -d "${SRC}" ]] || fail "Source directory does not exist: ${SRC}"

DEST_PARENT="$(dirname "${DEST}")"
[[ -d "${DEST_PARENT}" ]] || fail "Destination parent does not exist: ${DEST_PARENT}"
[[ -w "${DEST_PARENT}" ]] || fail "Destination parent is not writable: ${DEST_PARENT}"

mkdir -p "${DEST}"

RSYNC_OPTS=(
  -rltD
  --modify-window=2
  --partial
  --partial-dir=.rsync-partial
  --human-readable
  --stats
  --info=stats2,progress2
  --omit-dir-times
  --no-perms
  --no-owner
  --no-group
  --safe-links
  --copy-links
  --exclude=.rsync-partial/
)

if [[ "${MODE}" == "mirror" ]]; then
  RSYNC_OPTS+=(--delete-delay)
fi

if [[ "${VERBOSE}" == "1" ]]; then
  RSYNC_OPTS+=(-v)
fi

log "Backup start"
log "Mode: ${MODE}"
log "Source: ${SRC}"
log "Destination: ${DEST}"
log "Log file: ${LOG_FILE}"

START_EPOCH="$(date +%s)"

set +e
rsync "${RSYNC_OPTS[@]}" "${SRC}/" "${DEST}/" 2>&1 | tee -a "${LOG_FILE}"
RSYNC_EXIT="${PIPESTATUS[0]}"
set -e

END_EPOCH="$(date +%s)"
DURATION="$((END_EPOCH - START_EPOCH))"

case "${RSYNC_EXIT}" in
  0)
    log "Backup finished successfully in ${DURATION}s"
    ;;
  23)
    log "Backup finished with rsync exit code 23."
    log "This often means CIFS metadata operations such as setting times were not permitted."
    log "Check the log before trusting the backup as complete."
    ;;
  *)
    log "Backup failed with rsync exit code ${RSYNC_EXIT} after ${DURATION}s"
    exit "${RSYNC_EXIT}"
    ;;
esac

log "Backup end"
