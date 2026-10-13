#!/bin/sh
# Apply database migrations, then start the given command (uvicorn by default).
set -e

if [ "${DOCUMIND_RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "Applying database migrations..."
  attempt=1
  until alembic upgrade head; do
    if [ "$attempt" -ge 10 ]; then
      echo "Migrations failed after $attempt attempts." >&2
      exit 1
    fi
    echo "Database not ready yet (attempt $attempt); retrying in 3s..."
    attempt=$((attempt + 1))
    sleep 3
  done
fi

exec "$@"
