# Contributing

Thanks for helping improve this hybrid RAG template. The goal is to keep the project
small, understandable, and production-shaped enough that teams can safely adapt it.

## Local Setup

```bash
cp .env.example .env
uv sync --extra dev
```

For Docker-based development on macOS, OrbStack is recommended. Docker Desktop should
also work.

## Checks

Run the main local checks before opening a pull request:

```bash
make test
make lint
make compose-check
```

When Docker or OrbStack is available, also run:

```bash
make docker-build
make smoke
```

`make lint` is intentionally scoped to the production-critical API, cache, ingestion,
pipeline, and tests that are actively hardened. `make lint-all` is available to inspect
full-repo Ruff debt, but it is not the merge gate yet.

## Change Guidelines

- Keep diffs focused and easy to review.
- Prefer existing providers, factories, settings, and test style before adding new
  abstractions.
- Update tests when changing behavior.
- Update README or docs when changing setup, configuration, public routes, Docker
  behavior, or production assumptions.
- Do not commit secrets, provider keys, private documents, or generated local data.

## Security

This template includes optional API-key auth and safer defaults, but adopters still need
production identity, authorization, rate limiting, and upload hardening for internet-facing
deployments. See `docs/SECURITY.md` before making security-sensitive changes.

Please report security issues privately to the maintainer instead of opening a public
issue with exploit details.
