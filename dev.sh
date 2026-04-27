#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $(jobs -p) 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# Activate venv if needed (for fastapi/uvicorn)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source .venv/bin/activate
fi

echo "Starting Serve Analyzer dev environment..."
echo ""

# Start backend on port 8001
echo "Starting backend on http://localhost:8001 ..."
python -m web.backend &

# Wait for backend to be ready
echo "Waiting for backend..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8001/api/job > /dev/null 2>&1; then
        echo "Backend ready!"
        break
    fi
    sleep 0.5
done

# Start frontend on port 5173
echo "Starting frontend on http://localhost:5173 ..."
(cd web && npm run dev) &

echo ""
echo "==========================================="
echo "  Serve Analyzer is running!"
echo "  Open: http://localhost:5173"
echo "  API:  http://localhost:8001/api/job"
echo "  Press Ctrl+C to stop"
echo "==========================================="
echo ""

wait
