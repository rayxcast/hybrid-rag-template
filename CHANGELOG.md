# Changelog

All notable changes to this project will be documented here.

This project follows a simple keep-a-changelog style with an `Unreleased` section until
versioned releases are published.

## Unreleased

### Added

- Docker-backed health/readiness smoke test via `make smoke`.
- `/healthz` liveness and `/readyz` dependency readiness endpoints.
- Equality-only retrieval metadata filters on `POST /query/`.
- Metadata-filter-aware semantic cache scoping.
- `CONTRIBUTING.md` with local checks, Docker workflow, and contribution expectations.
- `make lint-all` to expose full-repo Ruff debt separately from the focused CI gate.

### Changed

- Docker app healthcheck now uses `/readyz`.
- CI uses Makefile targets for focused lint and Docker smoke validation.
- README and docs now cover health/readiness, metadata filters, smoke testing, and lint
  scope.

### Known Gaps

- Full-repo Ruff cleanup remains intentionally non-gating.
- Provider-backed ingest/query smoke tests are still manual until the template has a
  deterministic fake-provider mode.
