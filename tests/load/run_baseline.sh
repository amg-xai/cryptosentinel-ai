#!/bin/bash
# Run load test baseline and save results
# Usage: bash tests/load/run_baseline.sh

echo "=== CryptoSentinel AI Load Test ==="
echo "Make sure API is running: uvicorn src.api.main:app --port 8000"
echo ""

mkdir -p benchmark-data

locust \
  -f tests/load/locustfile.py \
  --headless \
  --users 30 \
  --spawn-rate 5 \
  --run-time 2m \
  --host http://localhost:8000 \
  --csv benchmark-data/load_test_baseline \
  --html benchmark-data/load_test_report.html

echo ""
echo "=== Results ==="
echo "CSV: benchmark-data/load_test_baseline_stats.csv"
echo "HTML report: benchmark-data/load_test_report.html"

if [ -f benchmark-data/load_test_baseline_stats.csv ]; then
  echo ""
  echo "Top endpoints by request count:"
  cat benchmark-data/load_test_baseline_stats.csv | \
    awk -F',' 'NR>1 {print $2, $3, $6"ms p50", $8"ms p95"}' | \
    sort -k2 -rn | head -10
fi
