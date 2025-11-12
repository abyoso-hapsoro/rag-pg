from src.db import get_session
from src.models.document import Document
from src.ingestion.store import add_document, ingest_document


class TestEmbeddingIngestion:
    EMBED_SAMPLE = 384 * [0]

    def test_ingest_document_basic(self):
        """Confirm deterministic embedding ingestion: ORM commit and readback."""
        content = "Basic Ingestion Test"
        with get_session() as session:
            doc = add_document(session, content=content, embedding=self.EMBED_SAMPLE)
            result = session.query(Document).filter_by(id=doc.id).first()
            print(f"Empty embedding sample: [{', '.join(f'{el:.6f}' for el in result.embedding[:5])}, ...]")
            assert result is not None
            assert result.content == content
            assert len(result.embedding) == 384

    def test_ingest_document_minilm(self):
        """Confirm MiniLM ingestion pipeline: embedding generation, commit, and readback."""
        sample_title = "MiniLM Ingestion Test"
        sample_text = "This is a test sentence for embedding."
        with get_session() as session:
            doc = ingest_document(session, title=sample_title, content=sample_text)
            print(f"Text embedding sample: [{', '.join(f'{el:.6f}' for el in doc.embedding[:5])}, ...]")
            assert doc.id is not None
            assert doc.embedding is not None
            assert len(doc.embedding) == 384
