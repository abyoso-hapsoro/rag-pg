from sqlalchemy.orm import Session
from ..models.document import Document
from .embedding import embed_text


def add_document(session: Session, content: str, embedding: list, title: str = None) -> Document:
    """
    Insert a document with a specified embedding.

    Args:
        session (sqlalchemy.orm.Session):
            Active SQLAlchemy session bound to PostgreSQL.
        content (str):
            Main textual content.
        embedding (list):
            Precomputed embedding vector, e.g., 384 * [0].
        title (str, optional):
            Title of the document.

    Returns:
        Document: ORM object after being committed to database.
    """

    # Create a new Document ORM object with specified embedding
    doc = Document(title=title, content=content, embedding=embedding)

    # Execute transaction: add, commit and refresh    
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Return the persisted Document object    
    return doc


def ingest_document(session: Session, title: str, content: str) -> Document:
    """
    Ingest a document by embedding and storing it.

    Args:
        session (sqlalchemy.orm.Session):
            Active SQLAlchemy session bound to PostgreSQL.
        title (str):
            Title of the document.
        content (str):
            Main textual content.

    Returns:
        Document: ORM object after being committed to database.
    """
    
    # Embed text using MiniLM
    embedding = embed_text(content)

    # Create a new Document ORM object with computed embedding    
    doc = Document(title=title, content=content, embedding=embedding.tolist())
    
    # Execute transaction: add, commit and refresh
    session.add(doc)
    session.commit()
    session.refresh(doc)
    
    # Return the persisted Document object
    return doc
