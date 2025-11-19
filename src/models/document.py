from sqlalchemy import Column, Integer, Text, String, DateTime, func
from pgvector.sqlalchemy import Vector
from src.models.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
