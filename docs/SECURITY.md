# Security Model

This project is a production-oriented RAG template, not a turnkey secured SaaS
application. Its default posture is designed to be safe for local development and
straightforward to harden when embedded into a larger system.

## Current Protections

- Docker Compose exposes only the FastAPI app to the host by default.
- Qdrant, Redis, and the reranker service are internal Docker services by default.
- App and reranker containers run as non-root users.
- `.env` files are ignored by Git, and `.env.example` contains placeholders only.
- API keys, auth headers, tokens, secrets, and passwords are redacted from structured logs.
- Upload ingestion validates filename, extension, non-empty content, and size.
- Query input is rejected when blank or too large.
- Container path ingestion is disabled unless `ALLOW_PATH_INGEST=true`.
- Optional API-key authentication can protect `/ingest/` and `/query/`.

## Optional API-Key Authentication

API-key authentication is disabled by default so the template is easy to run locally.
Enable it before exposing the API beyond trusted local or internal networks.

```env
API_KEY_AUTH_ENABLED=true
API_KEYS=replace-with-long-random-key,replace-with-second-key
```

Clients may send either:

```http
X-API-Key: replace-with-long-random-key
```

or:

```http
Authorization: Bearer replace-with-long-random-key
```

When `API_KEY_AUTH_ENABLED=true` and `API_KEYS` is empty, protected endpoints fail
closed with a service-unavailable response. This prevents accidentally enabling auth
without configuring credentials.

Protected endpoints:

- `POST /ingest/`
- `POST /query/`

Public endpoints:

- `GET /`
- `GET /status/`
- `GET /docs`
- `GET /openapi.json`
- static demo UI assets

## Known Non-Goals

This template does not currently provide:

- User accounts, sessions, OAuth, SSO, JWT validation, or tenant identity.
- Per-document authorization or tenant-isolated retrieval filters.
- Redis-backed or distributed rate limiting.
- Malware scanning or MIME sniffing for uploaded files.
- Full prompt-injection prevention for retrieved documents.
- Secrets management beyond environment variables.
- Production TLS, ingress, WAF, or reverse-proxy configuration.

Teams adopting this template should add those controls in the surrounding platform or
extend the API layer before internet-facing production use.

## Deployment Guidance

- Put the API behind TLS and a trusted reverse proxy or cloud load balancer.
- Keep Qdrant, Redis, and the reranker service private to the application network.
- Use long, random API keys if the built-in API-key guard is enabled.
- Prefer external secret managers over checked-in or manually shared `.env` files.
- Add rate limits at the proxy, API gateway, or middleware layer.
- Use tenant and document metadata filters before serving multi-tenant traffic.
- Recreate the Qdrant collection and clear Redis when changing embedding providers,
  embedding models, or embedding dimensions.

## Reporting Security Issues

Do not open a public issue for vulnerabilities or leaked credentials. Contact the
maintainer privately with a concise description, affected version or commit, and
reproduction steps when possible.
