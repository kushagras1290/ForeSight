"""Google "Sign in with Google" - authorization-code flow.

Two HTTP calls to Google, wrapped so the route handler in ``service`` never
touches an HTTP client directly: exchange the authorization code for a token,
then use that token to fetch the signed-in user's profile.

Requires a real Google Cloud OAuth client (``google_client_id`` /
``google_client_secret`` in :class:`foresight.config.Settings`) - the project
ships with placeholder values so the route exists and can be exercised, but
:func:`build_authorize_url` refuses to run until they are replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from urllib.parse import urlencode

import httpx

from foresight.config import Settings
from foresight.exceptions import OAuthConfigurationError, OAuthExchangeError
from foresight.logging_setup import get_logger

__all__ = ["GoogleProfile", "build_authorize_url", "exchange_code_for_profile"]

log = get_logger(__name__)

_AUTHORIZE_URL: Final[str] = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL: Final[str] = "https://oauth2.googleapis.com/token"
_USERINFO_URL: Final[str] = "https://openidconnect.googleapis.com/v1/userinfo"
_SCOPE: Final[str] = "openid email profile"

#: Both Google's own endpoints and this service's uptime depend on this being
#: short enough that a hung request fails the login rather than hangs it.
_REQUEST_TIMEOUT_SECONDS: Final[float] = 10.0


@dataclass(frozen=True, slots=True)
class GoogleProfile:
    """The subset of Google's userinfo response this project needs."""

    sub: str
    email: str
    name: str


def build_authorize_url(settings: Settings, *, state: str) -> str:
    """Build the URL to send the browser to start the Google consent screen.

    Raises:
        OAuthConfigurationError: the placeholder client id/secret are still in
            place - sending a user to Google with them would just bounce back
            as an error from Google's own side, so fail before that happens.
    """
    if not settings.google_oauth_configured:
        raise OAuthConfigurationError(
            "Google sign-in is not configured. Set FORESIGHT_GOOGLE_CLIENT_ID and "
            "FORESIGHT_GOOGLE_CLIENT_SECRET to a real Google Cloud OAuth client."
        )

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": _SCOPE,
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_profile(settings: Settings, *, code: str) -> GoogleProfile:
    """Trade an authorization code for the signed-in user's Google profile.

    Raises:
        OAuthConfigurationError: Google sign-in is not configured.
        OAuthExchangeError: Google rejected the code, or the token/userinfo
            request otherwise failed.
    """
    if not settings.google_oauth_configured:
        raise OAuthConfigurationError(
            "Google sign-in is not configured. Set FORESIGHT_GOOGLE_CLIENT_ID and "
            "FORESIGHT_GOOGLE_CLIENT_SECRET to a real Google Cloud OAuth client."
        )

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        try:
            token_response = await client.post(
                _TOKEN_URL,
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": settings.google_redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]

            userinfo_response = await client.get(
                _USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}
            )
            userinfo_response.raise_for_status()
            userinfo = userinfo_response.json()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("google oauth exchange failed", extra={"context": {"error": str(exc)}})
            raise OAuthExchangeError("Google sign-in failed. Please try again.") from exc

    sub = userinfo.get("sub")
    email = userinfo.get("email")
    if not sub or not email:
        raise OAuthExchangeError("Google did not return a usable profile.")

    return GoogleProfile(sub=sub, email=email, name=userinfo.get("name") or email)
