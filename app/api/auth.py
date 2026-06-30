import hmac
from typing import Annotated

from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import app_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def configured_api_keys(raw_keys: str | None = None) -> list[str]:
    keys = app_settings.API_KEYS if raw_keys is None else raw_keys
    return [key.strip() for key in keys.split(",") if key.strip()]


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None

    return token.strip()


def require_api_key_auth(
    *,
    api_key_auth_enabled: bool | None = None,
    api_keys: str | None = None,
    x_api_key: str | None = None,
    authorization: str | None = None,
) -> None:
    is_enabled = (
        app_settings.API_KEY_AUTH_ENABLED
        if api_key_auth_enabled is None
        else api_key_auth_enabled
    )
    if not is_enabled:
        return

    allowed_keys = configured_api_keys(api_keys)
    if not allowed_keys:
        raise HTTPException(
            status_code=503,
            detail="API key authentication is enabled but no API keys are configured.",
        )

    candidate = x_api_key or bearer_token(authorization)
    if not candidate:
        raise HTTPException(status_code=401, detail="Missing API key.")

    if not any(hmac.compare_digest(candidate, allowed_key) for allowed_key in allowed_keys):
        raise HTTPException(status_code=401, detail="Invalid API key.")


async def require_api_key(
    x_api_key: Annotated[str | None, Security(api_key_header)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    require_api_key_auth(x_api_key=x_api_key, authorization=authorization)
