#!/bin/sh

PORT=${1:-8000}

exec uv run uvicorn src.service.service:app --host 0.0.0.0 --port "$PORT" --reload
