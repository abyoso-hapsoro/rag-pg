"""add vector and bigram indexes

Revision ID: 6dd45946c9f9
Revises: 1a1a07131e53
Create Date: 2025-11-13 06:14:57.476063

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector


# revision identifiers, used by Alembic.
revision: str = '6dd45946c9f9'
down_revision: Union[str, Sequence[str], None] = '1a1a07131e53'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Vector index (HNSW for cosine similarity)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
        ON documents
        USING hnsw (embedding vector_cosine_ops);
    """)

    # Bigram index (GIN + pg_bigm)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_content_bigm
        ON documents
        USING gin (content gin_bigm_ops);
    """)


def downgrade():
    # Drop vector and bigram indexes
    op.execute("DROP INDEX IF EXISTS idx_documents_content_bigm;")
    op.execute("DROP INDEX IF EXISTS idx_documents_embedding_hnsw;")
