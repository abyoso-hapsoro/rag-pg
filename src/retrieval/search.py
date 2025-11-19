import spacy
from sqlalchemy import text
from src.models.document import Document
from src.ingestion.embedding import embed_text


# Initialize language model for synonym expansion
nlp = None


def get_nlp():
    """Lazy-load SpaCy model exactly once."""

    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_md", disable=["ner", "parser", "tagger"])
        except OSError:
            from spacy.cli import download
            download("en_core_web_md")
            nlp = spacy.load("en_core_web_md", disable=["ner", "parser", "tagger"])
    return nlp


def vector_search(session, query: str, limit: int = 5) -> list[tuple[Document, float]]:
    """
    Perform semantic similarity search using MiniLM embeddings and pgvector.

    Args:
        session (sqlalchemy.orm.Session):
            Active SQLAlchemy session bound to PostgreSQL.
        query (str):
            Text query to search.
        limit (int, optional):
            Maximum number of results to return (default: 5).

    Returns:
        list[tuple[Document, float]]: Closest documents in vector space with their similarity score.
    """

    # Embed query with MiniLM
    query_embedding = embed_text(query.lower()).tolist()

    # Prepare query with HNSW index accelerating the pgvector cosine distance operator
    sql = text("""
        SELECT 
            id,
            title,
            content,
            1 - (embedding <-> CAST(:q AS vector)) AS score
        FROM documents
        ORDER BY score DESC
        LIMIT :limit
    """)

    # Execute query
    rows = session.execute(sql, {"q": query_embedding, "limit": limit}).fetchall()

    # Return top results, filtering out non-positive scores
    return [
        (Document(id=r.id, title=r.title, content=r.content), float(r.score))
        for r in rows if float(r.score) > 0
    ]


def fuzzy_search(session, query: str, limit: int = 5, threshold: float = 0.1) -> list[tuple[Document, float]]:
    """
    Perform lexical search using pg_bigm.

    Args:
        session (sqlalchemy.orm.Session):
            Active SQLAlchemy session bound to PostgreSQL.
        query (str):
            Text query to search.
        limit (int, optional):
            Maximum number of results to return (default: 5).
        threshold (float, optional):
            Minimum similarity score (default: 0.1).

    Returns:
        list[tuple[Document, float]]: Closest documents in vector space with their similarity score.
    """
    
    # Prepare query with GIN index accelerating the pg_bigm bigram distance operator
    sql = text("""
        SELECT 
            id, 
            title, 
            content,
            bigm_similarity(LOWER(content), :q) AS score
        FROM documents
        -- WHERE LOWER(content) % :q
        ORDER BY score DESC
        LIMIT :limit
    """)

    # Execute query
    rows = session.execute(sql, {"q": query.lower(), "limit": limit}).fetchall()

    # Return top results, filtering out less than threshold scores
    return [
        (Document(id=r.id, title=r.title, content=r.content), float(r.score))
        for r in rows if float(r.score) >= threshold
    ]


def _synonym_expansion(session, query: str, threshold: float) -> str:
    """
    Expand query with synonyms from SpaCy vocabulary.

    Args:
        session (sqlalchemy.orm.Session):
            Active SQLAlchemy session bound to PostgreSQL.
        query (str):
            Text query to search.
        threshold (float):
            Similarity threshold for synonym inclusion.

    Returns:
        list[tuple[Document, float]]: Closest documents in vector space with their similarity score.
    """

    # Load language model
    nlp = get_nlp()

    # Vectorize query
    query_vec = nlp.make_doc(query.lower())

    # Fetch documents
    docs = session.execute(text("SELECT content FROM documents")).fetchall()

    # Initialize expanded terms
    expanded_terms = []

    # Iterate each row and add to expanded terms if surpass threshold
    for row in docs:
        doc_vec = nlp.make_doc(row.content)
        for token in doc_vec:
            similarity = token.similarity(query_vec)
            if similarity >= threshold:
                expanded_terms.append(token.text)

    # Deduplicate and prepare expanded query
    expanded_terms = list({t.lower() for t in expanded_terms})
    query_expanded = " ".join([query, *expanded_terms])

    # Return expanded query
    return query_expanded


def synonym_vector_search(session, query: str, limit: int = 5, threshold: float = 0.3) -> list[tuple[Document, float]]:
    """
    Perform synonym search using SpaCy similarity and pgvector.

    Args:
        session (sqlalchemy.orm.Session):
            Active SQLAlchemy session bound to PostgreSQL.
        query (str):
            Text query to search.
        limit (int, optional):
            Maximum number of results to return (default: 5).
        threshold (float, optional):
            Similarity threshold for synonym inclusion (default: 0.3).

    Returns:
        list[tuple[Document, float]]: Closest documents in vector space with their similarity score.
    """

    # Expand query with synonyms
    query_expanded = _synonym_expansion(session, query, threshold)

    # Run vector search with expanded query
    return vector_search(session, query_expanded, limit=limit)


def synonym_fuzzy_search(session, query: str, limit: int = 5, threshold: float = 0.3) -> list[tuple[Document, float]]:
    """
    Perform synonym search using SpaCy similarity and pg_bigm.

    Args:
        session (sqlalchemy.orm.Session):
            Active SQLAlchemy session bound to PostgreSQL.
        query (str):
            Text query to search.
        limit (int, optional):
            Maximum number of results to return (default: 5).
        threshold (float, optional):
            Similarity threshold for synonym inclusion (default: 0.3).

    Returns:
        list[tuple[Document, float]]: Closest documents in vector space with their similarity score.
    """

    # Expand query with synonyms
    query_expanded = _synonym_expansion(session, query, threshold)

    # Run fuzzy search with expanded query
    return fuzzy_search(session, query_expanded, limit=limit, threshold=threshold)
