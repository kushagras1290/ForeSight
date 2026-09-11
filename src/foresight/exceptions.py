"""Exception hierarchy for Project FORESIGHT.

Every failure mode in the pipeline maps to a specific exception type so callers
can react precisely and logs can be filtered by class. Nothing in this codebase
raises or catches bare ``Exception``.

Hierarchy::

    ForesightError
    +-- ConfigurationError        invalid or missing configuration
    +-- DataError                 anything wrong with the data itself
    |   +-- MissingDataFileError      an expected extract is absent
    |   +-- SchemaValidationError     columns/dtypes do not match the contract
    |   +-- DataQualityError          data is present but unusable
    +-- ModelError                anything wrong with the model lifecycle
    |   +-- InsufficientHistoryError  not enough weeks to train or backtest
    |   +-- LeakageError              a feature was built from future data
    |   +-- ModelNotFittedError       predict() called before fit()
    |   +-- ArtifactNotFoundError     a trained artifact is missing on disk
    +-- RiskScoringError          risk layer could not produce a decision
    +-- AuthError                 sign-in / session handling failed
        +-- InvalidCredentialsError   identifier or password did not match
        +-- UserAlreadyExistsError    email or username already registered
        +-- SessionExpiredError       session token unknown, expired or revoked
        +-- OAuthConfigurationError   Google sign-in not configured (placeholder creds)
        +-- OAuthExchangeError        Google rejected the code/token exchange
"""

from __future__ import annotations

__all__ = [
    "ArtifactNotFoundError",
    "AuthError",
    "ConfigurationError",
    "DataError",
    "DataQualityError",
    "ForesightError",
    "InsufficientHistoryError",
    "InvalidCredentialsError",
    "LeakageError",
    "MissingDataFileError",
    "ModelError",
    "ModelNotFittedError",
    "OAuthConfigurationError",
    "OAuthExchangeError",
    "RiskScoringError",
    "SchemaValidationError",
    "SessionExpiredError",
    "UserAlreadyExistsError",
]


class ForesightError(Exception):
    """Base class for every error raised by this project."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(ForesightError):
    """Configuration is missing, malformed, or internally inconsistent."""


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class DataError(ForesightError):
    """Base class for data-related failures."""


class MissingDataFileError(DataError):
    """An expected client extract could not be found on disk."""

    def __init__(self, path: object, hint: str = "") -> None:
        message = f"Required data file not found: {path}"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)
        self.path = path


class SchemaValidationError(DataError):
    """A dataframe does not satisfy its declared column contract."""

    def __init__(self, table: str, problems: list[str]) -> None:
        joined = "; ".join(problems)
        super().__init__(f"Schema validation failed for '{table}': {joined}")
        self.table = table
        self.problems = problems


class DataQualityError(DataError):
    """Data loaded and matched its schema but is not fit for use."""


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
class ModelError(ForesightError):
    """Base class for modelling failures."""


class InsufficientHistoryError(ModelError):
    """Too few observations to train or to run a rolling-origin backtest."""


class LeakageError(ModelError):
    """A feature was constructed from information unavailable at forecast time.

    Raised by the leakage guard in :mod:`foresight.features`. This is a hard
    failure by design: brief section 7.1 makes leak-free features the
    non-negotiable rule of the engagement, so a leak must break the build rather
    than quietly inflate the reported accuracy.
    """


class ModelNotFittedError(ModelError):
    """``predict`` was called before ``fit``."""


class ArtifactNotFoundError(ModelError):
    """A trained artifact required by the service or dashboard is missing."""

    def __init__(self, path: object, hint: str = "") -> None:
        message = f"Required artifact not found: {path}"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)
        self.path = path


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
class RiskScoringError(ForesightError):
    """The risk layer could not turn a forecast into a decision."""


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
class AuthError(ForesightError):
    """Base class for sign-in and session failures."""


class InvalidCredentialsError(AuthError):
    """The identifier/password pair did not match any account.

    Deliberately raised for both "no such user" and "wrong password" with the
    same message, so a caller cannot enumerate which registered emails exist
    by watching which error comes back.
    """


class UserAlreadyExistsError(AuthError):
    """Registration was attempted with an email or username already taken."""


class SessionExpiredError(AuthError):
    """The session token is unknown to the store, expired, or was revoked."""


class OAuthConfigurationError(AuthError):
    """Google sign-in was attempted without real OAuth credentials configured."""


class OAuthExchangeError(AuthError):
    """Google rejected the authorization code or the userinfo request."""
