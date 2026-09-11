"""Authentication routes: register, log in, log out, Google sign-in.

Kept in its own router rather than folded into ``service/main.py`` (already
large) - included there with ``app.include_router(auth_router)``. Session
state lives in :mod:`foresight.auth`; this module only translates HTTP in and
out of that store, matching the rest of the service's shape (routes are thin,
the library holds the logic).
"""

from __future__ import annotations

import secrets
from typing import Final

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse

from foresight.auth import (
    SECURITY_QUESTIONS,
    AuthUser,
    authenticate_user,
    change_password,
    create_session,
    delete_session,
    find_or_create_google_user,
    get_security_questions,
    register_user,
    reset_password_with_security_answers,
    resolve_session,
)
from foresight.config import Settings, get_settings
from foresight.exceptions import (
    InvalidCredentialsError,
    OAuthConfigurationError,
    OAuthExchangeError,
    SessionExpiredError,
    UserAlreadyExistsError,
)
from foresight.logging_setup import get_logger
from foresight.oauth import build_authorize_url, exchange_code_for_profile
from service.models import (
    ChangePasswordRequest,
    Envelope,
    ErrorCode,
    ForgotPasswordQuestionsRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    SecurityQuestionsResponse,
    UserResponse,
)

__all__ = ["auth_router", "get_current_user", "try_get_current_user"]

log = get_logger("service.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])
auth_router = router

_OAUTH_STATE_MAX_AGE_SECONDS: Final[int] = 600


def _oauth_state_cookie_name(settings: Settings) -> str:
    """Short-lived cookie proving the browser on the OAuth callback is the
    same one that started the flow - a minimal CSRF guard needing no
    server-side state.

    Derived from ``session_cookie_name`` rather than a fixed name: cookies
    are not port-scoped, so two instances of this service on the same
    machine (e.g. two localhost ports) would otherwise silently overwrite
    each other's OAuth state mid-flow whenever both were used from the same
    browser - not a hypothetical, this is exactly what tripped the state
    mismatch below the first time two instances shared one browser.
    """
    return f"{settings.session_cookie_name}_oauth_state"


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": ErrorCode.UNAUTHORIZED.value, "message": message},
    )


def try_get_current_user(request: Request) -> AuthUser | None:
    """Resolve the session cookie to a user, or ``None`` if absent/invalid.

    Used by the request-gating middleware, which needs to tell "not logged
    in" apart from every other kind of failure without raising.
    """
    settings = get_settings()
    raw_token = request.cookies.get(settings.session_cookie_name)
    if not raw_token:
        return None
    try:
        return resolve_session(settings, raw_token)
    except SessionExpiredError:
        return None


def get_current_user(request: Request) -> AuthUser:
    """FastAPI dependency: the signed-in user, or a 401."""
    user = try_get_current_user(request)
    if user is None:
        raise _unauthorized("Sign in required.")
    return user


def _set_session_cookie(response: Response, raw_token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        settings.session_cookie_name,
        raw_token,
        max_age=settings.session_ttl_days * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/",
    )


def _user_response(user: AuthUser) -> UserResponse:
    return UserResponse(
        id=user.id, email=user.email, username=user.username, display_name=user.display_name
    )


@router.get("/security-questions", response_model=Envelope[list[str]])
async def security_questions() -> Envelope[list[str]]:
    """The fixed list of security questions a registration form picks two from."""
    return Envelope(data=list(SECURITY_QUESTIONS))


@router.post("/register", response_model=Envelope[UserResponse])
async def register(payload: RegisterRequest, response: Response) -> Envelope[UserResponse]:
    """Create a password-based account and sign the caller in immediately."""
    settings = get_settings()
    try:
        user = register_user(
            settings,
            email=payload.email,
            username=payload.username,
            password=payload.password,
            display_name=payload.display_name,
            security_questions=(payload.security_question_1, payload.security_question_2),
            security_answers=(payload.security_answer_1, payload.security_answer_2),
        )
    except UserAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": ErrorCode.ALREADY_EXISTS.value, "message": str(exc)},
        ) from exc

    _set_session_cookie(response, create_session(settings, user.id))
    return Envelope(data=_user_response(user))


@router.post("/login", response_model=Envelope[UserResponse])
async def login(payload: LoginRequest, response: Response) -> Envelope[UserResponse]:
    """Sign in with an email-or-username and password."""
    settings = get_settings()
    try:
        user = authenticate_user(settings, identifier=payload.identifier, password=payload.password)
    except InvalidCredentialsError as exc:
        raise _unauthorized(str(exc)) from exc

    _set_session_cookie(response, create_session(settings, user.id))
    return Envelope(data=_user_response(user))


@router.post("/logout", response_model=Envelope[None])
async def logout(request: Request, response: Response) -> Envelope[None]:
    """End the current session. A no-op if there wasn't one."""
    settings = get_settings()
    raw_token = request.cookies.get(settings.session_cookie_name)
    if raw_token:
        delete_session(settings, raw_token)
    response.delete_cookie(settings.session_cookie_name, path="/")
    return Envelope(data=None)


@router.get("/me", response_model=Envelope[UserResponse])
async def me(request: Request) -> Envelope[UserResponse]:
    """The signed-in account. 401 if there isn't one.

    This is the one route a not-yet-authenticated frontend is expected to
    call, so it can tell whether to render the app or redirect to /login.
    """
    return Envelope(data=_user_response(get_current_user(request)))


@router.post("/change-password", response_model=Envelope[None])
async def change_password_route(payload: ChangePasswordRequest, request: Request) -> Envelope[None]:
    """Change the signed-in account's password. Requires the current one."""
    user = get_current_user(request)
    try:
        change_password(
            get_settings(),
            user_id=user.id,
            current_password=payload.current_password,
            new_password=payload.new_password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.VALIDATION_ERROR.value, "message": str(exc)},
        ) from exc
    return Envelope(data=None)


@router.post("/forgot-password/questions", response_model=Envelope[SecurityQuestionsResponse])
async def forgot_password_questions(
    payload: ForgotPasswordQuestionsRequest,
) -> Envelope[SecurityQuestionsResponse]:
    """Step 1 of recovery: look up the two questions on file for an account."""
    try:
        questions = get_security_questions(get_settings(), identifier=payload.identifier)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.NOT_FOUND.value, "message": str(exc)},
        ) from exc
    return Envelope(data=SecurityQuestionsResponse(questions=list(questions)))


@router.post("/forgot-password/reset", response_model=Envelope[UserResponse])
async def forgot_password_reset(
    payload: ResetPasswordRequest, response: Response
) -> Envelope[UserResponse]:
    """Step 2 of recovery: answer both questions correctly to set a new password.

    Signs the caller in immediately on success, the same as register/login -
    a successful reset is itself proof of identity.
    """
    settings = get_settings()
    try:
        user = reset_password_with_security_answers(
            settings,
            identifier=payload.identifier,
            security_answers=(payload.security_answer_1, payload.security_answer_2),
            new_password=payload.new_password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": ErrorCode.VALIDATION_ERROR.value, "message": str(exc)},
        ) from exc

    _set_session_cookie(response, create_session(settings, user.id))
    return Envelope(data=_user_response(user))


@router.get("/google/start", include_in_schema=True)
async def google_start() -> RedirectResponse:
    """Redirect the browser to Google's consent screen."""
    settings = get_settings()
    try:
        state = secrets.token_urlsafe(24)
        url = build_authorize_url(settings, state=state)
    except OAuthConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": ErrorCode.OAUTH_ERROR.value, "message": str(exc)},
        ) from exc

    redirect = RedirectResponse(url, status_code=status.HTTP_302_FOUND)
    redirect.set_cookie(
        _oauth_state_cookie_name(settings),
        state,
        max_age=_OAUTH_STATE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/api/auth/google",
    )
    return redirect


@router.get("/google/callback", include_in_schema=True)
async def google_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
) -> RedirectResponse:
    """Google redirects here with a code after the user grants consent."""
    settings = get_settings()
    expected_state = request.cookies.get(_oauth_state_cookie_name(settings))
    if not expected_state or not secrets.compare_digest(expected_state, state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.OAUTH_ERROR.value,
                "message": "OAuth state mismatch. Please try signing in again.",
            },
        )

    try:
        profile = await exchange_code_for_profile(settings, code=code)
    except (OAuthConfigurationError, OAuthExchangeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": ErrorCode.OAUTH_ERROR.value, "message": str(exc)},
        ) from exc

    user = find_or_create_google_user(
        settings, google_sub=profile.sub, email=profile.email, display_name=profile.name
    )

    redirect = RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    redirect.delete_cookie(_oauth_state_cookie_name(settings), path="/api/auth/google")
    _set_session_cookie(redirect, create_session(settings, user.id))
    return redirect
