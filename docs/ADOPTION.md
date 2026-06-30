# Adoption Guide

This project is meant to be copied, forked, or used as a reference when adding hybrid
RAG to an existing system. Treat it as a strong starting point, not as a drop-in SaaS.

## What To Reuse

- FastAPI route structure for ingest, query, and status endpoints.
- Provider factories for embeddings, vector stores, and rerankers.
- Docker Compose topology for app, Qdrant, Redis, and reranker service.
- Request tracing shape for debugging retrieval, reranking, generation, and cache behavior.
- Cache scope design that prevents stale answers after new ingestion.
- Evaluation scaffolding for domain-specific RAG quality checks.

## What To Replace

- Authentication: integrate with your platform identity, gateway, or API-key system.
- Authorization: add tenant, user, and document-level filters before retrieval.
- Storage: connect ingestion to your canonical document store or event pipeline.
- Secrets: use your deployment platform's secret manager instead of local `.env` files.
- Observability: export logs, metrics, traces, and alerts to your production stack.
- Deployment: replace local Docker Compose with your production platform configuration.

## Safe Integration Path

1. Run the template locally with sample documents.
2. Disable demo-only features that do not fit your product.
3. Wire ingestion to your document source of truth.
4. Add metadata to chunks for tenant, document, source, and permission filtering.
5. Enforce authorization before retrieval.
6. Configure provider keys and model settings through your secret/config system.
7. Add domain-specific tests and evals before relying on answers in production.
8. Add dashboards and alerts for latency, errors, cache hit rate, and provider failures.

## Cache Integration Notes

The semantic cache is scoped to the active indexed corpus. If your system has multiple
tenants, collections, or permission scopes, extend the cache scope before enabling shared
cache use. At minimum, include tenant or authorization scope in the cache key or payload.

Do not share cached answers across users who do not have access to the same documents.

## Multi-Tenant Notes

Before using this template in a multi-tenant system:

- Store tenant and document identifiers in chunk metadata.
- Apply metadata filters in retrieval.
- Include tenant or access scope in semantic cache scope.
- Add tests proving users cannot retrieve or cache another tenant's data.
- Keep audit logs for ingestion and query access.

## What Not To Ship Unchanged

- Public unauthenticated ingestion or query endpoints.
- Path ingestion enabled in internet-facing deployments.
- Local `.env` secrets management.
- Demo UI as the only operational interface.
- One-size-fits-all eval cases for a regulated or high-impact domain.

## Reviewer Checklist

A careful adopter should be able to answer:

- How are documents ingested, chunked, and indexed?
- Which provider/model produced the embeddings?
- What document set was active when an answer was cached?
- Which chunks supported the final answer?
- How are user permissions applied before retrieval?
- What happens when Redis, Qdrant, the reranker, or an LLM provider fails?
- Which tests prove the behavior that matters for this product?
