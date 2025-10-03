#!/bin/bash
# Test runner script for route optimization POC

echo "========================================"
echo "Running Route Optimization POC Tests"
echo "========================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run graph tests
echo "Testing Graph Module..."
python3 -m unittest tests/test_graph.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Graph tests passed${NC}"
else
    echo -e "${RED}✗ Graph tests failed${NC}"
    exit 1
fi
echo ""

# Run batch tests
echo "Testing Batch Module..."
python3 -m unittest tests/test_batch.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Batch tests passed${NC}"
else
    echo -e "${RED}✗ Batch tests failed${NC}"
    exit 1
fi
echo ""

echo "========================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "========================================"
