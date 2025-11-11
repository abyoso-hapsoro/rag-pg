# Environment imports
import os
from dotenv import load_dotenv
load_dotenv()

# Core imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Retrieve database URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required for ORM Session.")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, future=True, echo=False)

# Set the ORM session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# Dependency utility for application/services
@contextmanager
def get_session():
    """Context managed session for application code usage"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
