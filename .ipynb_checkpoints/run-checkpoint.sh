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
    echo "Loi: Khong tim thay tep .env tai $SCRIPT_DIR/.env"
    exit 1
fi

STATE_ARG=""
if [ -n "$STATE" ]; then
    STATE_ARG="--state $STATE"
fi

echo "======================================"
echo " Chay tuan tu: $TOTAL mau | delay: ${DELAY}s | loai: $DOC_TYPE ${STATE}"
echo "======================================"

SUCCESS=0
FAIL=0

for i in $(seq 1 $TOTAL); do
    echo ""
    echo "[$i/$TOTAL] Bat dau sinh mau..."

    python "$SCRIPT_DIR/main.py" \
        --type "$DOC_TYPE" \
        $STATE_ARG \
        --count 1 \
        --workers 1 \
        --project-id "$PROJECT_ID" \
        --image-model gemini-2.5-flash-image \
        2>&1 | tee /tmp/run_output.txt | tail -5

    if grep -qE "That bai\s+[1-9]" /tmp/run_output.txt; then
        FAIL=$((FAIL + 1))
        echo "[$i/$TOTAL] THAT BAI - Thanh cong: $SUCCESS | That bai: $FAIL"
    else
        SUCCESS=$((SUCCESS + 1))
        echo "[$i/$TOTAL] OK - Thanh cong: $SUCCESS | That bai: $FAIL"
    fi

    if [ $i -lt $TOTAL ]; then
        echo "Cho ${DELAY}s truoc lan tiep theo..."
        sleep "$DELAY"
    fi
done

echo ""
echo "======================================"
echo " Success: $SUCCESS/$TOTAL"
echo "======================================"