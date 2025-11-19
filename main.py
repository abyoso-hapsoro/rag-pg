from fastapi import FastAPI
from src.app.router import router

app = FastAPI(title="RAG-PG API")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
