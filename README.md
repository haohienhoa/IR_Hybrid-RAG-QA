# ViJuris

A Vietnamese legal question answering system built on the Zalo AI Challenge (ZAC) 2021 legal corpus.

ViJuris combines retrieval-augmented generation (RAG), hybrid search (BM25 + semantic retrieval), cross-encoder reranking, and LLM-based answer generation to provide grounded responses from Vietnamese legal documents.

## Tech Stack

* FastAPI
* LangGraph
* ChromaDB
* BM25
* keepitreal/vietnamese-sbert
* BAAI/bge-reranker-v2-m3
* Groq API

## Run

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

## API Documentation

```text
http://127.0.0.1:8000/docs
```
