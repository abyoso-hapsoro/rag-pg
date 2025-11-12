# Overview
This test suite validates the minimal baseline that our local development environment is provisioned correctly.

# Usage
```bash
>>> docker compose down -v  # full reset, wipe volumes
>>> docker compose up -d    # start fresh Postgres instance
>>> alembic upgrade head    # apply database schema migrations
>>> pytest -s               # run tests with print output visible
```

# Tests
| Test Name	| Purpose |
| :-: | :-- |
| `test_env_provision` | Confirms database container and dependency environment is functional and responding |
| `test_schema_valid` | Confirms Alembic schema upgraded properly and tables exist as expected |
| `test_vector_search` | Confirms pgvector extension is active and similarity operator works |
| `test_bigram_search` | Confirms pg_bigm extension is active and LIKE search returns multiple matches |
| `test_ingest_document_basic` | Confirms deterministic embedding ingestion |
| `test_ingest_document_minilm` | Confirms MiniLM embedding ingestion pipeline |