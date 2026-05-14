#!/bin/bash
TOTAL=${1:-10}
DELAY=${2:-8}
DOC_TYPE=${3:-passport}
STATE=${4:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "Error: File .env not found at $SCRIPT_DIR/.env"
    exit 1
fi

STATE_ARG=""
if [ -n "$STATE" ]; then
    STATE_ARG="--state $STATE"
fi

echo "======================================"
echo " Running sequentially: $TOTAL documents | delay: ${DELAY}s | type: $DOC_TYPE ${STATE}"
echo "======================================"

SUCCESS=0
FAIL=0

for i in $(seq 1 $TOTAL); do
    echo ""
    echo "[$i/$TOTAL] Begin generating..."

    python "$SCRIPT_DIR/main.py" \
        --type "$DOC_TYPE" \
        $STATE_ARG \
        --count 1 \
        --workers 1 \
        --project-id "$PROJECT_ID" \
        --image-model gemini-2.5-flash-image \
        2>&1 | tee /tmp/run_output.txt | tail -5

    STATUS=${PIPESTATUS[0]}

    if [ $STATUS -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "[$i/$TOTAL] OK"
    else
        FAIL=$((FAIL + 1))
        echo "[$i/$TOTAL] FAIL"
    fi
done

echo ""
echo "======================================"
echo " Success: $SUCCESS/$TOTAL"
echo "======================================"

