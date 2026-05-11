#!/bin/bash
TOTAL=${1:-10}
DELAY=${2:-8}
DOC_TYPE=${3:-passport}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================"
echo " Chay tuan tu: $TOTAL mau | delay: ${DELAY}s | loai: $DOC_TYPE"
echo "======================================"

SUCCESS=0
FAIL=0

for i in $(seq 1 $TOTAL); do
    echo ""
    echo "[$i/$TOTAL] Bat dau sinh mau..."

    python "$SCRIPT_DIR/main.py" \
        --type "$DOC_TYPE" \
        --count 1 \
        --workers 1 \
        --project-id first-orc-chien \
        --image-model gemini-2.5-flash-image \
        2>&1 | tee /tmp/run_output.txt | tail -5

    # Nếu có "That bai   1" trong output = fail
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
echo " XONG: Thanh cong $SUCCESS/$TOTAL"
echo "======================================"