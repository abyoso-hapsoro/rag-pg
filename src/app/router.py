from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from io import StringIO

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db import get_session
from src.app.helper import unwrap_session
from src.models.document import Document
from src.ingestion.embedding import embed_text
from src.retrieval.search import synonym_vector_search, synonym_fuzzy_search


router = APIRouter()

@router.post("/rag")
async def rag_endpoint(
    session_cm = Depends(get_session),
    file: UploadFile | None = File(None),
    query: str | None = Form(None),
    limit: int = Form(5),
    threshold: float = Form(0.3),
    method: str = Form("vector")
):
    session: Session = unwrap_session(session_cm)
    ingest_flag = False

    if file is not None:
        if file.content_type != "text/csv":
            raise HTTPException(400, "File must be a CSV.")
        session.execute(text("TRUNCATE TABLE documents RESTART IDENTITY CASCADE;"))
        session.commit()
        raw = (await file.read()).decode("utf-8")
        normalized = raw.replace('""', '\\"')
        stream = StringIO(normalized)
        try:
            chunks = pd.read_csv(
                stream,
                chunksize=32,
                sep=",",
                quotechar='"',
                skipinitialspace=True,
                escapechar="\\",
                engine="python",
            )
        except Exception as e:
            raise HTTPException(400, f"CSV parsing error: {e}")
        for chunk in chunks:
            required = {"title", "content"}
            if not required.issubset(chunk.columns):
                raise HTTPException(400,
                    "CSV must contain exactly: title, content"
                )
            records = []
            for _, row in chunk.iterrows():
                title = str(row["title"])
                content = str(row["content"])
                embedding = embed_text(content.lower()).tolist()
                records.append(
                    Document(title=title, content=content, embedding=embedding)
                )
            session.add_all(records)
            session.commit()
        ingest_flag = True

    if query is not None:
        if method == "vector":
            results = synonym_vector_search(session, query, limit=limit)
            return [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "score": score
                }
                for doc, score in results
            ]
        elif method == "fuzzy":
            results = synonym_fuzzy_search(session, query, limit=limit, threshold=threshold)
            return [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "score": score,
                }
                for doc, score in results
            ]
        else:
            raise HTTPException(400, "method must be 'vector' or 'fuzzy'")

    if ingest_flag:
        return {"message": "File ingested."}

    raise HTTPException(400, "Provide a CSV file or a query")
