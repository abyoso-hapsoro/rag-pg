#!/bin/bash

# Full reset, wipe volumes
docker compose down -v

# Start fresh Postgres instance
docker compose up -d

# Wait for Postgres to start
sleep 5

# Apply database schema migrations
alembic upgrade head

# Run tests with print output visible
pytest -s