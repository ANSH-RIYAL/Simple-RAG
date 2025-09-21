import os
import shutil
import json
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
import requests

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
_OPENAI_API_KEY: Optional[str] = None
_OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------- OpenAI helpers ---------------- #

def _openai_headers() -> dict:
    if not _OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in .env or environment")
    return {
        "Authorization": f"Bearer {_OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def _openai_chat_json(system_prompt: str, user_prompt: str) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": _OPENAI_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    resp = requests.post(url, headers=_openai_headers(), json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"OpenAI response not JSON: {content}") from e


def classify_intent_via_openai(query: str) -> dict:
    system = (
        "You are a classifier for routing queries in a RAG system."
        " Always return strict JSON only."
        " Fields: intent (one of: Chit-chat, KB_QA, List, Table),"
        " knowledge_base_requirement (0-10 integer), comments (string)."
    )
    user = (
        "Classify the user query."
        " Return JSON with keys: intent, knowledge_base_requirement, comments.\n\n"
        f"Query: {query}"
    )
    return _openai_chat_json(system, user)


def rewrite_query_via_openai(query: str) -> dict:
    system = (
        "You are a query rewriter to improve retrieval for a RAG system."
        " Normalize, keep key entities, expand with 2-6 high-value synonyms/aliases,"
        " remove fluff. Output concise retrieval query. Return strict JSON only with:"
        " rewritten_query (string), keywords (array of strings), notes (string)."
    )
    user = (
        "Rewrite the query for retrieval. JSON only. Fields: rewritten_query, keywords, notes.\n\n"
        f"Query: {query}"
    )
    return _openai_chat_json(system, user)


@app.on_event("startup")
def startup_event() -> None:
    global _store, _files_indexed, _embedder, _OPENAI_API_KEY

    # Load .env from same directory as this file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))

    # Ensure keys are present (from .env or environment)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set. Create src/api/.env with MISTRAL_API_KEY=...")
    _embedder = MistralEmbeddingsClient(api_key=api_key)

    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not _OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Add to src/api/.env as OPENAI_API_KEY=...")

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

    # LLM-based intent classification
    cls = classify_intent_via_openai(req.query)
    intent = str(cls.get("intent", "")).strip().lower()
    # parse knowledge_base_requirement (supports "7/10" or 7)
    kb_req_raw = cls.get("knowledge_base_requirement", 0)
    kb_score = 0
    try:
        if isinstance(kb_req_raw, str) and "/" in kb_req_raw:
            kb_score = int(kb_req_raw.split("/", 1)[0])
        else:
            kb_score = int(kb_req_raw)
    except Exception:
        kb_score = 0

    # If not suitable for KB search, return empty results
    if intent in ("chit-chat", "chitchat") or kb_score < 4:
        return []

    # LLM-based query rewrite
    rew = rewrite_query_via_openai(req.query)
    rewritten_query = str(rew.get("rewritten_query", req.query)) or req.query

    # Embed rewritten query and search
    qvec = _embedder.embed_texts([rewritten_query])[0]
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
