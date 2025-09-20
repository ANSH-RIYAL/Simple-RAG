import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv

from src.rag.embeddings import MistralEmbeddingsClient
from src.rag.vector_store import SimpleVectorStore


PDF_DIR = Path("vector_db_files")
INDEX_DIR = Path("index_pdf")


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    texts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt:
            texts.append(txt)
    return "\n\n".join(texts)


def split_into_paragraphs(text: str) -> List[str]:
    blocks = [b.strip() for b in text.split("\n\n")]
    return [b for b in blocks if b]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    source_file: str
    chunk_index: int


app = FastAPI(title="Simple Vector DB API")

# Global state
_store: SimpleVectorStore | None = None
_files_indexed: List[str] = []
_embedder: MistralEmbeddingsClient | None = None


@app.on_event("startup")
def startup_event() -> None:
    global _store, _files_indexed, _embedder

    # Load .env from same directory as this file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))

    # Ensure API key is present (from .env or environment)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set. Create src/api/.env with MISTRAL_API_KEY=...")
    _embedder = MistralEmbeddingsClient(api_key=api_key)

    # Reset index directory
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Collect PDFs
    if not PDF_DIR.exists():
        PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted([p for p in PDF_DIR.iterdir() if p.suffix.lower() == ".pdf"])

    texts: List[str] = []
    ids: List[str] = []
    metas: List[dict] = []
    _files_indexed = []

    for pdf in pdfs:
        try:
            text = extract_pdf_text(pdf)
            paras = split_into_paragraphs(text)
            if not paras:
                continue
            _files_indexed.append(pdf.name)
            for i, para in enumerate(paras):
                ids.append(f"{pdf.name}:para:{i}")
                texts.append(para)
                metas.append({"source_file": str(pdf), "chunk_index": i})
        except Exception:
            # Skip malformed PDFs
            continue

    if not texts:
        # Initialize empty store with a default dim to avoid None; will error on search if empty
        _store = None
        return

    vectors = _embedder.embed_texts(texts, batch_size=24)
    dim = int(vectors.shape[1])
    _store = SimpleVectorStore(embedding_dim=dim)
    for i in range(len(texts)):
        _store.add(item_id=ids[i], text=texts[i], vector=vectors[i], metadata=metas[i])
    _store.save(str(INDEX_DIR))


@app.get("/files")
def list_files() -> dict:
    return {"processed_files": _files_indexed}


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest) -> List[SearchResult]:
    if _store is None or _embedder is None:
        raise HTTPException(status_code=503, detail="Index not ready or empty")
    qvec = _embedder.embed_texts([req.query])[0]
    results = _store.search(qvec, top_k=req.top_k)
    out: List[SearchResult] = []
    for r in results:
        out.append(SearchResult(
            id=r["id"],
            score=r["score"],
            text=r["text"],
            source_file=str(r["metadata"].get("source_file", "")),
            chunk_index=int(r["metadata"].get("chunk_index", -1)),
        ))
    return out
