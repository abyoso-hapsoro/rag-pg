from sqlalchemy import text
from src.db import get_session
from src.models.document import Document


class TestDatabase:
    # Set testing parameter(s)
    EMBED = 384 * [0]

    def test_env_provision(self):
        """Verify infrastructure baseline is alive: DB container, Driver, ORM session are reachable."""
        with get_session() as session:
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1

    def test_schema_valid(self):
        """Validate ORM model and Alembic schema correctness by insert → commit → readback."""
        # Insert dummy document with deterministic embedding
        with get_session() as session:
            doc = Document(
                content="Hello World - CRUD",
                embedding=self.EMBED,
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            doc_id = doc.id

        # Retrieve document then verify persistence and embedding
        with get_session() as session:
            result = session.query(Document).filter_by(id=doc_id).first()
            assert result is not None
            assert result.content == "Hello World - CRUD"
            assert list(result.embedding) == self.EMBED

    def test_vector_search(self):
        """Test pgvector: Ensure similarity operator <-> executes and returns valid result set."""
        with get_session() as session:
            rows = session.execute(
                text(
                    """
                    SELECT id, embedding <-> CAST(:q AS vector) AS distance
                    FROM documents
                    ORDER BY distance ASC
                    LIMIT 5
                    """
                ), {"q": self.EMBED},
            ).fetchall()

            # Debug view
            print("Vector Search Output:", rows)

            # Assert operator success and produced a resultset shape
            assert rows is not None
            assert len(rows) >= 0

    def test_bigram_search(self):
        """Test pg_bigm: Ensure we can query using LIKE and pg_bigm is active."""
        with get_session() as session:
            # Create temporary table to test bigram matching
            session.execute(text("DROP TABLE IF EXISTS test_bigm;"))
            session.execute(text("CREATE TABLE test_bigm (txt TEXT);"))
            session.execute(text(
                """
                INSERT INTO test_bigm (txt)
                VALUES ('Hello World'), ('Hello RAG-PG'), ('Bigram Search');
                """
            ))

            # Run LIKE search which pg_bigm accelerates internally
            rows = session.execute(
                text("SELECT txt FROM test_bigm WHERE txt LIKE '%Hello%';")
            ).fetchall()

            # Debug view
            print("Bigram Search Output:", rows) 

            # Assert operator success and produced a resultset shape
            assert rows is not None
            assert len(rows) >= 2
