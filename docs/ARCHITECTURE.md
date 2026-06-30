# Architecture

This repository is a production-oriented hybrid RAG template. It is intentionally
small enough to understand, but shaped like a service that can be embedded into a
larger application.

## Runtime Components

- FastAPI app: exposes ingestion, query, status, and demo UI routes.
- Qdrant: stores dense and sparse vectors for document chunks.
- Redis Stack: stores semantic answer cache entries and the active index revision.
- Reranker service: runs FastEmbed cross-encoder reranking behind an internal HTTP API.
- LLM provider: called through LiteLLM or provider-specific LlamaIndex integrations.

Docker Compose keeps Qdrant, Redis, and the reranker service internal by default. The
FastAPI app is the only service exposed to the host.

## Query Flow

1. Validate the incoming query and optional API key.
2. Load the current cache scope from Redis.
3. Look up semantically similar cached answers for the same cache scope.
4. Retrieve candidate chunks from Qdrant when there is no valid cache hit.
5. Rerank candidates when reranking is enabled.
6. Generate a grounded answer from the final context chunks.
7. Store the answer in Redis with the current cache scope and TTL.
8. Return answer, sources, cache status, timings, and trace metadata.

## Ingestion Flow

1. Validate upload or opt-in path ingestion request.
2. Load supported documents from PDF, TXT, or Markdown inputs.
3. Split documents into chunks.
4. Create or reuse the configured Qdrant collection.
5. Embed and index chunks.
6. Bump the Redis-backed index revision only after successful indexing.
7. Return ingest counts, timings, and the new revision in the trace.

If ingestion fails, the index revision is not bumped. Existing cache entries remain
eligible only for the previous successful index state.

## Cache Correctness

Semantic caching is useful for cost and latency, but answer caches are only correct
for the corpus that produced them. This template scopes each answer cache entry by:

- collection name
- embedding provider
- embedding model
- embedding dimension
- index revision

The index revision is stored in Redis as:

```text
index_revision:{COLLECTION_NAME}
```

It defaults to `0` and increments after successful ingestion. Cache TTL is still used
to limit Redis growth, but index scope is the correctness boundary.

## Extension Points

- Add tenant and document authorization filters in retrieval before serving multi-tenant traffic.
- Replace Qdrant by implementing the vector store provider interface.
- Replace embedding or reranking providers behind the existing provider factories.
- Add production auth, rate limits, and observability in the API layer or upstream gateway.
- Extend evals with domain-specific golden cases before using this template in critical workflows.
