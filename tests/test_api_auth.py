import pytest
from fastapi import HTTPException

from app.api.auth import bearer_token, configured_api_keys, require_api_key_auth

HTTP_401_UNAUTHORIZED = 401
HTTP_503_SERVICE_UNAVAILABLE = 503


def test_configured_api_keys_strips_empty_values() -> None:
    assert configured_api_keys(" first-key, ,second-key ") == ["first-key", "second-key"]


def test_bearer_token_extracts_valid_bearer_token() -> None:
    assert bearer_token("Bearer secret-key") == "secret-key"


def test_bearer_token_rejects_non_bearer_authorization() -> None:
    assert bearer_token("Basic secret-key") is None


def test_require_api_key_auth_allows_requests_when_disabled() -> None:
    require_api_key_auth(api_key_auth_enabled=False, api_keys="", x_api_key=None)


def test_require_api_key_auth_fails_closed_when_no_keys_configured() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_api_key_auth(api_key_auth_enabled=True, api_keys="", x_api_key="anything")

    assert exc_info.value.status_code == HTTP_503_SERVICE_UNAVAILABLE


def test_require_api_key_auth_rejects_missing_key() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_api_key_auth(api_key_auth_enabled=True, api_keys="expected-key")

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED


def test_require_api_key_auth_rejects_invalid_key() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_api_key_auth(
            api_key_auth_enabled=True,
            api_keys="expected-key",
            x_api_key="wrong-key",
        )

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED


def test_require_api_key_auth_accepts_x_api_key() -> None:
    require_api_key_auth(
        api_key_auth_enabled=True,
        api_keys="expected-key",
        x_api_key="expected-key",
    )


def test_require_api_key_auth_accepts_bearer_token() -> None:
    require_api_key_auth(
        api_key_auth_enabled=True,
        api_keys="expected-key",
        authorization="Bearer expected-key",
    )
