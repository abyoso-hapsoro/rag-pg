from sqlalchemy import text
from src.db import get_session
from src.models.document import Document
from src.ingestion.store import ingest_document
from src.retrieval.search import vector_search, fuzzy_search, synonym_vector_search, synonym_fuzzy_search


class TestSearch:
    def setup_method(self):
        """Truncate table and insert dummy documents."""
        with get_session() as session:
            session.execute(text("TRUNCATE TABLE documents RESTART IDENTITY CASCADE;"))
            ingest_document(session, "Apple", "Apples are red fruits")
            ingest_document(session, "Car", "Cars drive on the road")
            ingest_document(session, "Banana", "Bananas are yellow and sweet")
            ingest_document(session, "Vehicle", "Vehicles include cars, trucks and buses")
            ingest_document(session, "Fruit", "Fruit can be apples, bananas or oranges")


    def test_vector_search_empty(self):
        with get_session() as session:
            results = vector_search(session, "")
            assert len(results) == 0
            # print(
            #     "Vector Search (Empty Query):",
            #     [(doc.title, round(score, 4)) for doc, score in results] if len(results) == 0 else []
            # )


    def test_vector_search(self):
        with get_session() as session:
            results = vector_search(session, "fruit", limit=5)
            assert len(results) > 0
            print(
                "Vector Search:",
                [(doc.title, round(score, 4)) for doc, score in results]
            )
            first_doc, first_score = results[0]
            assert isinstance(first_doc, Document)
            assert isinstance(first_score, float)


    def test_fuzzy_search_empty(self):
        with get_session() as session:
            results = fuzzy_search(session, "")
            assert len(results) == 0
            # print(
            #     "Fuzzy Search (Empty Query):",
            #     [(doc.title, round(score, 4)) for doc, score in results] if len(results) == 0 else []
            # )


    def test_fuzzy_search(self):
        with get_session() as session:
            # session.execute(text("SET pg_bigm.similarity_threshold = 0.2;"))
            results = fuzzy_search(session, "fruit", limit=5)
            assert len(results) > 0
            print("Fuzzy Search:", [(doc.title, round(score, 4)) for doc, score in results])
            first_doc, first_score = results[0]
            assert isinstance(first_doc, Document)
            assert isinstance(first_score, float)


    def test_synonym_vector_search(self):
        with get_session() as session:
            results = synonym_vector_search(session, "automobile", limit=5)
            assert len(results) > 0
            print("Synonym Vector Search:", [(doc.title, round(score, 4)) for doc, score in results])
            first_doc, first_score = results[0]
            assert isinstance(first_doc, Document)
            assert isinstance(first_score, float)


    def test_synonym_fuzzy_search(self):
        with get_session() as session:
            results = synonym_fuzzy_search(session, "automobile", limit=5)
            assert len(results) > 0
            print("Synonym Fuzzy Search:", [(doc.title, round(score, 4)) for doc, score in results])
            first_doc, first_score = results[0]
            assert isinstance(first_doc, Document)
            assert isinstance(first_score, float)
