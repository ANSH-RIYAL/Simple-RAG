import os
import shutil
import json
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
import requests

from src.rag.embeddings import MistralEmbeddingsClient
from src.rag.vector_store import SimpleVectorStore
from src.rag.chunking import chunk_text_by_headers
from src.rag.retrieval import TFIDFIndex, hybrid_search


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


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    source_file: str
    chunk_index: int
    heading_path: str = ""


class ChatResponse(BaseModel):
    query: str
    response: str
    chunks_retrieved: List[SearchResult]


app = FastAPI(title="Simple RAG API")

# Global state
_store: SimpleVectorStore | None = None
_tfidf_index: TFIDFIndex | None = None
_chunk_texts: List[str] = []
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


def _openai_chat_json(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": _OPENAI_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
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


def _openai_chat_text(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": _OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=_openai_headers(), json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


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


def build_index_from_pdfs():
    """Build both vector and TF-IDF indices from PDFs."""
    global _store, _tfidf_index, _chunk_texts, _files_indexed, _embedder
    
    print(f"\n\n=== BUILDING INDEX ===")
    
    if not PDF_DIR.exists():
        PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted([p for p in PDF_DIR.iterdir() if p.suffix.lower() == ".pdf"])
    
    print(f"Found {len(pdfs)} PDF files: {[p.name for p in pdfs]}")
    
    all_chunks = []
    _files_indexed = []
    
    for pdf in pdfs:
        try:
            print(f"Processing {pdf.name}...")
            text = extract_pdf_text(pdf)
            if not text.strip():
                print(f"  No text extracted from {pdf.name}")
                continue
            chunks = chunk_text_by_headers(text, pdf.name)
            if chunks:
                print(f"  Created {len(chunks)} chunks from {pdf.name}")
                all_chunks.extend(chunks)
                _files_indexed.append(pdf.name)
            else:
                print(f"  No chunks created from {pdf.name}")
        except Exception as e:
            print(f"  Error processing {pdf.name}: {e}")
            continue
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    if not all_chunks:
        print("No chunks to index - setting stores to None")
        _store = None
        _tfidf_index = None
        _chunk_texts = []
        return
    
    # Build vector store
    texts = [chunk.text for chunk in all_chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    vectors = _embedder.embed_texts(texts, batch_size=24)
    dim = int(vectors.shape[1])
    _store = SimpleVectorStore(embedding_dim=dim)
    
    for i, chunk in enumerate(all_chunks):
        _store.add(
            item_id=chunk.id,
            text=chunk.text,
            vector=vectors[i],
            metadata=chunk.metadata
        )
    _store.save(str(INDEX_DIR))
    print(f"Vector store saved with {len(all_chunks)} chunks")
    
    # Build TF-IDF index
    _chunk_texts = texts
    _tfidf_index = TFIDFIndex()
    _tfidf_index.build_index(texts)
    print(f"TF-IDF index built with vocabulary size: {len(_tfidf_index.vocabulary)}")
    
    print(f"Index building complete! Files indexed: {_files_indexed}")


@app.on_event("startup")
def startup_event() -> None:
    global _embedder, _OPENAI_API_KEY
    
    # Load .env from same directory as this file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
    
    # Ensure keys are present
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
    
    # Build indices
    build_index_from_pdfs()


# Mount static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the main UI."""
    html_path = Path("src/api/templates/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>UI not found</h1>", status_code=404)


@app.get("/files")
def list_files() -> dict:
    return {"processed_files": _files_indexed, "total_count": len(_files_indexed)}


@app.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)) -> dict:
    """Upload PDFs and rebuild index."""
    uploaded = []
    
    for file in files:
        if not file.filename or not file.filename.endswith('.pdf'):
            continue
        
        # Save to vector_db_files
        file_path = PDF_DIR / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        uploaded.append(file.filename)
    
    # Rebuild indices
    build_index_from_pdfs()
    
    return {
        "uploaded_files": uploaded,
        "processed_files": _files_indexed,
        "total_count": len(_files_indexed)
    }


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest) -> List[SearchResult]:
    if _store is None or _embedder is None or _tfidf_index is None:
        raise HTTPException(status_code=503, detail="Index not ready or empty")
    
    print(f"\n\n=== SEARCH REQUEST ===")
    print(f"Query: {req.query}")
    print(f"Top-k: {req.top_k}")
    
    # LLM-based intent classification
    cls = classify_intent_via_openai(req.query)
    intent = str(cls.get("intent", "")).strip().lower()
    kb_req_raw = cls.get("knowledge_base_requirement", 0)
    kb_score = 0
    try:
        if isinstance(kb_req_raw, str) and "/" in kb_req_raw:
            kb_score = int(kb_req_raw.split("/", 1)[0])
        else:
            kb_score = int(kb_req_raw)
    except Exception:
        kb_score = 0

    print(f"\n\n=== INTENT CLASSIFICATION ===")
    print(f"Intent: {intent}")
    print(f"KB Score: {kb_score}")
    print(f"Comments: {cls.get('comments', 'N/A')}")

    # If not suitable for KB search, return empty results
    if intent in ("chit-chat", "chitchat") or kb_score < 4:
        print(f"\n\n=== SEARCH SKIPPED ===")
        print(f"Reason: Intent '{intent}' or KB score {kb_score} < 4")
        return []

    # LLM-based query rewrite
    rew = rewrite_query_via_openai(req.query)
    rewritten_query = str(rew.get("rewritten_query", req.query)) or req.query

    print(f"\n\n=== QUERY REWRITE ===")
    print(f"Original: {req.query}")
    print(f"Rewritten: {rewritten_query}")
    print(f"Keywords: {rew.get('keywords', [])}")

    # Semantic search
    qvec = _embedder.embed_texts([rewritten_query])[0]
    semantic_results = _store.search(qvec, top_k=50)  # Get more for fusion

    print(f"\n\n=== SEMANTIC SEARCH ===")
    print(f"Found {len(semantic_results)} semantic results")
    for i, r in enumerate(semantic_results[:5]):
        print(f"  {i+1}. Score: {r['score']:.4f} | {r['id']} | {r['text'][:100]}...")

    # TF-IDF search
    tfidf_results = _tfidf_index.search(rewritten_query, top_k=50)

    print(f"\n\n=== TF-IDF SEARCH ===")
    print(f"Found {len(tfidf_results)} TF-IDF results")
    for i, (idx, score) in enumerate(tfidf_results[:5]):
        chunk_text = _chunk_texts[idx] if idx < len(_chunk_texts) else "N/A"
        print(f"  {i+1}. Score: {score:.4f} | Chunk {idx} | {chunk_text[:100]}...")

    # Hybrid fusion with threshold
    fused_results = hybrid_search(
        semantic_results, tfidf_results, _chunk_texts, 
        req.query, threshold=0.3  # Lowered threshold
    )

    print(f"\n\n=== HYBRID FUSION ===")
    print(f"Threshold: 0.3")
    print(f"Results after fusion: {len(fused_results)}")
    for i, r in enumerate(fused_results[:5]):
        print(f"  {i+1}. Fused: {r['score']:.4f} | Sem: {r.get('semantic_score', 0):.4f} | TF-IDF: {r.get('tfidf_score', 0):.4f}")
        print(f"      {r['id']} | {r['text'][:100]}...")

    # Convert to response format
    out: List[SearchResult] = []
    for r in fused_results[:req.top_k]:
        out.append(SearchResult(
            id=r["id"],
            score=r["score"],
            text=r["text"],
            source_file=str(r["metadata"].get("source_file", "")),
            chunk_index=int(r["metadata"].get("chunk_index", -1)),
            heading_path=str(r["metadata"].get("heading_path", ""))
        ))
    
    print(f"\n\n=== FINAL RESULTS ===")
    print(f"Returning {len(out)} results to client")
    
    return out


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Chat endpoint with RAG generation."""
    
    # Get search results using same logic as /search
    search_req = SearchRequest(query=req.query, top_k=req.top_k)
    chunks = search(search_req)
    
    if not chunks:
        # No relevant chunks found
        response_text = "I don't have sufficient information in my knowledge base to answer this question. Please try rephrasing or ask about topics covered in the uploaded documents."
        return ChatResponse(
            query=req.query,
            response=response_text,
            chunks_retrieved=[]
        )
    
    # Generate response using OpenAI
    context_text = "\n\n".join([
        f"[Source: {chunk.source_file}]\n{chunk.text}" 
        for chunk in chunks
    ])
    
    system_prompt = (
        "You are a helpful assistant that answers questions based on provided context. "
        "Use only the information from the context to answer questions. "
        "If the context doesn't contain enough information, say so. "
        "Always cite sources in your response using [Source: filename] format."
    )
    
    user_prompt = f"""Context:
{context_text}

Question: {req.query}

Please provide a comprehensive answer based on the context above."""
    
    try:
        response_text = _openai_chat_text(system_prompt, user_prompt, max_tokens=800)
    except Exception as e:
        response_text = f"Error generating response: {str(e)}"
    
    return ChatResponse(
        query=req.query,
        response=response_text,
        chunks_retrieved=chunks
    )