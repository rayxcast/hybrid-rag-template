# Hybrid RAG Template

A modular, production-oriented Hybrid Retrieval-Augmented Generation (RAG) system built with:

- FastAPI
- Qdrant (vector store)
- Redis (semantic cache)
- OpenAI embeddings + LLM
- Cross-encoder reranker
- Dockerized infrastructure
- Offline evaluation framework

---

## Architecture

User Query
    ↓
Retriever (Vector + Sparse)
    ↓
Reranker (Cross-Encoder)
    ↓
LLM Generator
    ↓
Answer + Sources

Infrastructure:
- Qdrant (vector storage)
- Redis (cache)
- Docker Compose orchestration

---

## Features

- Hybrid retrieval (dense + sparse)
- Optional reranking
- Configurable RAG pipeline
- Deterministic evaluation module
- Cost tracking
- Modular structure for production use

---

## Project Structure
├── app/
│ ├── api/
│ ├── rag/
│ ├── utils/
│ ├── config.py
│ └── main.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt


---

## Quick Start

### 1. Clone

git clone https://github.com/rayxcast/hybrid-rag-template.git
cd hybrid-rag-template

### 2. Create .env

OPENAI_API_KEY=your_key_here

### 3. Run

docker compose up --build

- App runs at: http://localhost:8000
- FastAPI Swagger UI: http://localhost:8000/docs
- Quadrant dashboard: http://localhost:6333/dashboard

---

## API Endpoints

POST `/ingest`
POST `/query`

---

## Evaluation

docker compose run --rm eval 

## Evaluation Dataset Source

Evaluation questions are derived from the public 10-K filing of Alphabet Inc.

Official filing source: Alphabet 10-k Annual Report Feb 5, 2026
https://abc.xyz/investor/sec-filings/
(Download Anual Fillings for 2026)

The PDF is not included in this repository.

## Benchmark Results (v1.0)

Evaluation Set:
- 25 questions
  - 5 adversarial
  - 5 unanswerable
  - 5 paraphrased
  - 5 multi-hop
  - 5 numerical precision

Results:
- Accuracy: 96.00%
- Avg Retrieval Time: 0.67s
- Avg Rerank Time: 1.03s
- Avg Generation Time: 6.39s
- Avg Judge Time: 5.93s

---
