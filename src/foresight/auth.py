"""User accounts and sessions.

A small SQLite store, deliberately: this is a single-team internal tool, not a
multi-tenant product, and the rest of the project already keeps its state in
flat files (parquet, JSON) rather than a server database. SQLite gives durable,
concurrent-safe storage with zero extra infrastructure to run.

Passwords are hashed with :func:`hashlib.scrypt` (stdlib, no extra dependency)
under a random per-user salt; only the hash and salt are stored, never the
password itself. Sessions are opaque random tokens handed to the browser as a
cookie - only their SHA-256 hash is kept server-side, so reading the database
does not hand out usable session tokens.

Google sign-in is a separate concern, in :mod:`foresight.oauth`; this module
only knows how to attach or create the local account a Google identity maps
to.

Forgotten-password recovery uses a security question rather than email
delivery, because this project has no outbound mail transport configured.
The trade-off is deliberate and worth naming: revealing which question is on
file for an identifier confirms that identifier is registered, which a
"we emailed you a reset link either way" flow would not. That is accepted
here for a single-team internal tool with no mail infrastructure; it would
not be the right choice for a public-facing product.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from foresight.config import Settings
from foresight.exceptions import (
    InvalidCredentialsError,
    SessionExpiredError,
    UserAlreadyExistsError,
)
from foresight.logging_setup import get_logger

__all__ = [
    "AuthUser",
    "SECURITY_QUESTIONS",
    "authenticate_user",
    "change_password",
    "create_session",
    "delete_session",
    "find_or_create_google_user",
    "get_security_questions",
    "init_auth_db",
    "register_user",
    "reset_password_with_security_answers",
    "resolve_session",
]

log = get_logger(__name__)

# --- Password hashing --------------------------------------------------- #
# scrypt cost parameters. N is the dominant work factor (memory and CPU both
# scale with it); 2**14 takes low tens of milliseconds on ordinary hardware,
# slow enough to make offline guessing expensive without making login feel slow.
_SCRYPT_N: Final[int] = 2**14
_SCRYPT_R: Final[int] = 8
_SCRYPT_P: Final[int] = 1
_SCRYPT_DKLEN: Final[int] = 32
_SALT_BYTES: Final[int] = 16

# --- Sessions ------------------------------------------------------------- #
_SESSION_TOKEN_BYTES: Final[int] = 32

# --- Security questions ---------------------------------------------------- #
# A fixed list rather than a free-text question the user writes themselves:
# self-authored questions are reliably weaker ("what's my favourite colour?")
# and a fixed list keeps the answer's normalisation (below) predictable. Two
# independent questions, both required to reset a password, rather than one -
# a single question is one guess (or one data-broker lookup) away from a
# takeover.
SECURITY_QUESTIONS: Final[tuple[str, ...]] = (
    "What was the name of your first pet?",
    "What city were you born in?",
    "What was your childhood nickname?",
    "What is your mother's maiden name?",
    "What was the model of your first car?",
)


@dataclass(frozen=True, slots=True)
class AuthUser:
    """A registered account, without anything secret attached."""

    id: int
    email: str
    username: str
    display_name: str


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )


def _hash_session_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


@contextmanager
def _connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_auth_db(settings: Settings) -> None:
    """Create the users/sessions tables if they do not already exist."""
    with _connection(settings.auth_db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                email                   TEXT NOT NULL UNIQUE,
                username                TEXT NOT NULL UNIQUE,
                password_hash           BLOB,
                password_salt           BLOB,
                google_sub              TEXT UNIQUE,
                display_name            TEXT NOT NULL,
                security_question_1     TEXT,
                security_answer_hash_1  BLOB,
                security_answer_salt_1  BLOB,
                security_question_2     TEXT,
                security_answer_hash_2  BLOB,
                security_answer_salt_2  BLOB,
                created_at              TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
            """
        )
    log.info("auth store ready", extra={"context": {"path": str(settings.auth_db_path)}})


def _row_to_user(row: sqlite3.Row) -> AuthUser:
    return AuthUser(
        id=row["id"], email=row["email"], username=row["username"], display_name=row["display_name"]
    )


def _normalise_answer(answer: str) -> str:
    """Fold a security answer so trivial formatting differences still match."""
    return " ".join(answer.strip().casefold().split())


def register_user(
    settings: Settings,
    *,
    email: str,
    username: str,
    password: str,
    display_name: str,
    security_questions: tuple[str, str],
    security_answers: tuple[str, str],
) -> AuthUser:
    """Create a new password-based account.

    Raises:
        UserAlreadyExistsError: the email or username is already registered.
    """
    normalised_email = email.strip().lower()
    normalised_username = username.strip().lower()
    salt = secrets.token_bytes(_SALT_BYTES)
    password_hash = _hash_password(password, salt)

    answer_salt_1 = secrets.token_bytes(_SALT_BYTES)
    answer_hash_1 = _hash_password(_normalise_answer(security_answers[0]), answer_salt_1)
    answer_salt_2 = secrets.token_bytes(_SALT_BYTES)
    answer_hash_2 = _hash_password(_normalise_answer(security_answers[1]), answer_salt_2)

    created_at = dt.datetime.now(tz=dt.UTC).isoformat()

    with _connection(settings.auth_db_path) as connection:
        try:
            cursor = connection.execute(
                """
                INSERT INTO users (
                    email, username, password_hash, password_salt, display_name,
                    security_question_1, security_answer_hash_1, security_answer_salt_1,
                    security_question_2, security_answer_hash_2, security_answer_salt_2,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalised_email,
                    normalised_username,
                    password_hash,
                    salt,
                    display_name.strip(),
                    security_questions[0],
                    answer_hash_1,
                    answer_salt_1,
                    security_questions[1],
                    answer_hash_2,
                    answer_salt_2,
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise UserAlreadyExistsError(
                "An account with that email or username already exists."
            ) from exc

        row = connection.execute(
            "SELECT id, email, username, display_name FROM users WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()

    log.info("user registered", extra={"context": {"user_id": row["id"]}})
    return _row_to_user(row)


def authenticate_user(settings: Settings, *, identifier: str, password: str) -> AuthUser:
    """Verify an email-or-username + password pair.

    Raises:
        InvalidCredentialsError: no account matches, or the password is wrong.
            Both cases raise the same error with the same message on purpose.
    """
    normalised = identifier.strip().lower()
    with _connection(settings.auth_db_path) as connection:
        row = connection.execute(
            """
            SELECT id, email, username, display_name, password_hash, password_salt
            FROM users WHERE email = ? OR username = ?
            """,
            (normalised, normalised),
        ).fetchone()

    if row is None or row["password_hash"] is None or row["password_salt"] is None:
        # No account (or a Google-only account with no password set) - reject
        # identically to a wrong password, so neither can be distinguished.
        raise InvalidCredentialsError("Incorrect email/username or password.")

    candidate = _hash_password(password, bytes(row["password_salt"]))
    if not hmac.compare_digest(candidate, bytes(row["password_hash"])):
        raise InvalidCredentialsError("Incorrect email/username or password.")

    return _row_to_user(row)


def change_password(
    settings: Settings, *, user_id: int, current_password: str, new_password: str
) -> None:
    """Replace a password-account's password after verifying the current one.

    Raises:
        InvalidCredentialsError: the current password is wrong, or this
            account has no password to change (it signs in with Google only -
            the message says so explicitly, since there is no enumeration
            risk in telling a user who is already signed in about their own
            account).
    """
    with _connection(settings.auth_db_path) as connection:
        row = connection.execute(
            "SELECT password_hash, password_salt FROM users WHERE id = ?", (user_id,)
        ).fetchone()

        if row is None or row["password_hash"] is None or row["password_salt"] is None:
            raise InvalidCredentialsError(
                "This account has no password - it signs in with Google only."
            )

        candidate = _hash_password(current_password, bytes(row["password_salt"]))
        if not hmac.compare_digest(candidate, bytes(row["password_hash"])):
            raise InvalidCredentialsError("The current password is incorrect.")

        new_salt = secrets.token_bytes(_SALT_BYTES)
        new_hash = _hash_password(new_password, new_salt)
        connection.execute(
            "UPDATE users SET password_hash = ?, password_salt = ? WHERE id = ?",
            (new_hash, new_salt, user_id),
        )

    log.info("password changed", extra={"context": {"user_id": user_id}})


def get_security_questions(settings: Settings, *, identifier: str) -> tuple[str, str]:
    """Return both security questions on file for an email or username.

    Raises:
        InvalidCredentialsError: no password account matches. (See the module
            docstring: revealing the questions does confirm the account
            exists, an accepted trade-off for this project.)
    """
    normalised = identifier.strip().lower()
    with _connection(settings.auth_db_path) as connection:
        row = connection.execute(
            "SELECT security_question_1, security_question_2 FROM users WHERE email = ? OR username = ?",
            (normalised, normalised),
        ).fetchone()

    if row is None or row["security_question_1"] is None or row["security_question_2"] is None:
        raise InvalidCredentialsError("No account with security questions matches that identifier.")

    return (str(row["security_question_1"]), str(row["security_question_2"]))


def reset_password_with_security_answers(
    settings: Settings,
    *,
    identifier: str,
    security_answers: tuple[str, str],
    new_password: str,
) -> AuthUser:
    """Reset a password after verifying **both** security answers on file.

    Raises:
        InvalidCredentialsError: no account matches, or either answer is
            wrong. All three cases raise the same error, for the same reason
            as :func:`authenticate_user` - and because revealing which one of
            two answers was wrong halves the effort of guessing the other.
    """
    normalised = identifier.strip().lower()
    with _connection(settings.auth_db_path) as connection:
        row = connection.execute(
            """
            SELECT id, email, username, display_name,
                   security_answer_hash_1, security_answer_salt_1,
                   security_answer_hash_2, security_answer_salt_2
            FROM users WHERE email = ? OR username = ?
            """,
            (normalised, normalised),
        ).fetchone()

        if (
            row is None
            or row["security_answer_hash_1"] is None
            or row["security_answer_hash_2"] is None
        ):
            raise InvalidCredentialsError("Incorrect identifier or security answers.")

        candidate_1 = _hash_password(
            _normalise_answer(security_answers[0]), bytes(row["security_answer_salt_1"])
        )
        candidate_2 = _hash_password(
            _normalise_answer(security_answers[1]), bytes(row["security_answer_salt_2"])
        )
        # Both comparisons always run - a short-circuiting `and` would make the
        # first answer being wrong take measurably less time than the second
        # being wrong, a timing side-channel that leaks which one to retry.
        first_matches = hmac.compare_digest(candidate_1, bytes(row["security_answer_hash_1"]))
        second_matches = hmac.compare_digest(candidate_2, bytes(row["security_answer_hash_2"]))
        if not (first_matches and second_matches):
            raise InvalidCredentialsError("Incorrect identifier or security answers.")

        new_salt = secrets.token_bytes(_SALT_BYTES)
        new_hash = _hash_password(new_password, new_salt)
        connection.execute(
            "UPDATE users SET password_hash = ?, password_salt = ? WHERE id = ?",
            (new_hash, new_salt, row["id"]),
        )

    log.info("password reset via security answers", extra={"context": {"user_id": row["id"]}})
    return _row_to_user(row)


def find_or_create_google_user(
    settings: Settings, *, google_sub: str, email: str, display_name: str
) -> AuthUser:
    """Look up the account linked to a Google identity, creating one if needed.

    An existing password account with the same email is linked rather than
    duplicated, so a user who registered with a password and later uses
    "Sign in with Google" lands on the same account.
    """
    normalised_email = email.strip().lower()

    with _connection(settings.auth_db_path) as connection:
        row = connection.execute(
            "SELECT id, email, username, display_name FROM users WHERE google_sub = ?",
            (google_sub,),
        ).fetchone()
        if row is not None:
            return _row_to_user(row)

        existing_by_email = connection.execute(
            "SELECT id FROM users WHERE email = ?", (normalised_email,)
        ).fetchone()
        if existing_by_email is not None:
            connection.execute(
                "UPDATE users SET google_sub = ? WHERE id = ?",
                (google_sub, existing_by_email["id"]),
            )
            row = connection.execute(
                "SELECT id, email, username, display_name FROM users WHERE id = ?",
                (existing_by_email["id"],),
            ).fetchone()
            log.info(
                "linked google identity to existing account",
                extra={"context": {"user_id": row["id"]}},
            )
            return _row_to_user(row)

        # New account. The username has to be unique but Google does not give
        # us one, so derive one from the email's local part and disambiguate
        # on collision rather than fail the sign-in.
        base_username = normalised_email.split("@", 1)[0] or "user"
        username = base_username
        suffix = 1
        while connection.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            suffix += 1
            username = f"{base_username}{suffix}"

        created_at = dt.datetime.now(tz=dt.UTC).isoformat()
        cursor = connection.execute(
            """
            INSERT INTO users (email, username, google_sub, display_name, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (normalised_email, username, google_sub, display_name or username, created_at),
        )
        row = connection.execute(
            "SELECT id, email, username, display_name FROM users WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()

    log.info("user created via google sign-in", extra={"context": {"user_id": row["id"]}})
    return _row_to_user(row)


def create_session(settings: Settings, user_id: int) -> str:
    """Start a session for ``user_id`` and return the raw token for the cookie.

    Only the token's hash is persisted; the raw value returned here is the
    only time it exists outside the caller's browser.
    """
    raw_token = secrets.token_urlsafe(_SESSION_TOKEN_BYTES)
    now = dt.datetime.now(tz=dt.UTC)
    expires_at = now + dt.timedelta(days=settings.session_ttl_days)

    with _connection(settings.auth_db_path) as connection:
        connection.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (_hash_session_token(raw_token), user_id, now.isoformat(), expires_at.isoformat()),
        )
    return raw_token


def resolve_session(settings: Settings, raw_token: str) -> AuthUser:
    """Return the user a session token belongs to.

    Raises:
        SessionExpiredError: the token is unknown, expired, or was revoked.
    """
    token_hash = _hash_session_token(raw_token)
    with _connection(settings.auth_db_path) as connection:
        row = connection.execute(
            """
            SELECT u.id, u.email, u.username, u.display_name, s.expires_at
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()

        if row is None:
            raise SessionExpiredError("Session not found. Please sign in again.")

        expires_at = dt.datetime.fromisoformat(row["expires_at"])
        if expires_at <= dt.datetime.now(tz=dt.UTC):
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            raise SessionExpiredError("Session expired. Please sign in again.")

    return _row_to_user(row)


def delete_session(settings: Settings, raw_token: str) -> None:
    """Log out: revoke one session. Safe to call with an unknown token."""
    with _connection(settings.auth_db_path) as connection:
        connection.execute(
            "DELETE FROM sessions WHERE token_hash = ?", (_hash_session_token(raw_token),)
        )
