#!/bin/bash

# ─────────────────────────────────────────────
#  run_seq.sh — Sequential Document Generator
# ─────────────────────────────────────────────

TOTAL=${1:-10}
DELAY=${2:-8}
DOC_TYPE=${3:-passport}
STATE=${4:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colors & Styles ───────────────────────────
RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"

CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
BLUE="\033[34m"

BG_GREEN="\033[42m"
BG_RED="\033[41m"
BG_YELLOW="\033[43m"
BLACK="\033[30m"

# ── Helpers ───────────────────────────────────
divider() {
    printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────"
}

badge() {
    local color="$1" label="$2"
    printf "${color}${BLACK}${BOLD} %s ${RESET}" "$label"
}

log_step() {
    printf "\n${BOLD}${BLUE}[%02d/%02d]${RESET} Generating %s...\n" "$1" "$2" "$3"
}

log_ok() {
    printf "  ${GREEN}${BOLD}OK${RESET}   document generated successfully\n"
}

log_fail() {
    printf "  ${RED}${BOLD}FAIL${RESET} generation encountered an error\n"
}

log_delay() {
    printf "  ${DIM}Waiting ${DELAY}s before next run...${RESET}\n"
}

# ── Load .env ─────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    printf "\n${RED}${BOLD}  ERROR${RESET}  .env not found at: ${DIM}$SCRIPT_DIR/.env${RESET}\n\n"
    exit 1
fi

# ── State arg ─────────────────────────────────
STATE_ARG=""
if [ -n "$STATE" ]; then
    STATE_ARG="--state $STATE"
fi

# ── Header ────────────────────────────────────
clear
printf "\n"
printf "  ${BOLD}${CYAN}SEQUENTIAL DOCUMENT GENERATOR${RESET}\n"
divider
printf "  Total   : ${BOLD}%s${RESET} documents\n" "$TOTAL"
printf "  Type    : ${BOLD}%s${RESET}\n" "$DOC_TYPE"
printf "  Delay   : ${BOLD}%ss${RESET} between runs\n" "$DELAY"
[ -n "$STATE" ] && printf "  State   : ${BOLD}%s${RESET}\n" "$STATE"
divider
printf "\n"

# ── Run loop ──────────────────────────────────
SUCCESS=0
FAIL=0

for i in $(seq 1 $TOTAL); do
    log_step "$i" "$TOTAL" "$DOC_TYPE"

    python "$SCRIPT_DIR/main.py" \
        --type "$DOC_TYPE" \
        $STATE_ARG \
        --count 1 \
        --workers 1 \
        --project-id "$PROJECT_ID" \
        --image-model gemini-2.5-flash-image \
        2>&1 | tee /tmp/run_output.txt > /dev/null

    STATUS=${PIPESTATUS[0]}
    FAIL_COUNT=$(grep -E "Failure\s+[1-9]" /tmp/run_output.txt | wc -l)

    if [ $STATUS -eq 0 ] && [ "$FAIL_COUNT" -eq 0 ]; then
        log_ok
        SUCCESS=$((SUCCESS + 1))
        if [ "$i" -lt "$TOTAL" ]; then
            log_delay
            sleep "$DELAY"
        fi
    else
        FAIL=$((FAIL + 1))
        log_fail
        printf "\n  ${YELLOW}${BOLD}Error Details:${RESET}\n"
        divider
        grep -iE "error|exception|traceback|fail|loi" /tmp/run_output.txt \
            | tail -n 10 \
            | while IFS= read -r line; do
                printf "  ${DIM}%s${RESET}\n" "$line"
              done
        divider
    fi
done

# ── Summary ───────────────────────────────────
printf "\n"
divider

if [ "$FAIL" -eq 0 ]; then
    RESULT_BADGE="${BG_GREEN}"
elif [ "$SUCCESS" -eq 0 ]; then
    RESULT_BADGE="${BG_RED}"
else
    RESULT_BADGE="${BG_YELLOW}"
fi

printf "  ${BOLD}SUMMARY${RESET}   "
badge "$RESULT_BADGE" "${SUCCESS}/${TOTAL} succeeded"
printf "   ${RED}${FAIL} failed${RESET}\n"
divider
printf "\n"