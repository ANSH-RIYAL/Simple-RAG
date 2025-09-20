## Vector db + Mistral embeddings

### v1 - Current project structure

- .gitignore
- README.md
- requirements.txt
- project_structure.md
- index_pdf/
  - config.json
  - items.jsonl
  - embeddings.npy
- src/
  - api/
    - __init__.py
    - server.py
    - .env (MISTRAL_API_KEY=...)
  - rag/
    - __init__.py
    - embeddings.py
    - vector_store.py
- vector_db_files/
  - soc2_framework_doc.pdf


### v2 - Final planned structure (submission)

- .gitignore
- README.md
- requirements.txt
- project_structure.md
- index_pdf/
  - config.json
  - items.jsonl
  - embeddings.npy
- vector_db_files/
  - (user-uploaded PDFs live here before startup indexing)
- src/
  - api/
    - __init__.py
    - server.py  (startup indexes PDFs from vector_db_files; routes: GET /files, POST /search, POST /ingest, POST /chat)
    - templates/
      - index.html  (simple UI: upload control + chat pane + retrieved chunks panel)
    - static/
      - app.js      (handles uploads, chat requests, renders results/citations)
      - styles.css
    - .env         (MISTRAL_API_KEY=...)
  - rag/
    - __init__.py
    - embeddings.py      (Mistral embeddings client)
    - vector_store.py    (on-disk vector DB; cosine search)
    - chunking.py        (paragraph/heading-aware chunking utilities)
    - query.py           (intent detection, query rewrite)
    - retrieval.py       (hybrid semantic+keyword search, MMR rerank, scoring)
    - generation.py      (LLM prompt templates, response formatting, citations)
    - conversation.py    (Conversation class: intent check → RAG fetch → final LLM answer)
- tests/
  - api_smoke_test.http (sample curl/httpie requests)
  - sample_queries.txt
