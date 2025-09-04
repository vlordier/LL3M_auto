"""Authentication and authorization system for LL3M API."""

import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

# Security constants
MIN_SECRET_KEY_LENGTH = 32


class AuthConfig(BaseModel):
    """Authentication configuration."""

    SECRET_KEY: str = Field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""),
        description="JWT secret key - MUST be set in production",
    )
    ALGORITHM: str = Field(
        default=os.getenv("JWT_ALGORITHM", "HS256"), description="JWT signing algorithm"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
        description="Access token expiration in minutes",
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")),
        description="Refresh token expiration in days",
    )
    PASSWORD_MIN_LENGTH: int = Field(
        default=int(os.getenv("PASSWORD_MIN_LENGTH", "8")),
        description="Minimum password length",
    )

    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        # Allow empty secret key for testing environments
        if not self.SECRET_KEY:
            env = os.getenv("ENVIRONMENT", "development")
            if env.lower() in ("test", "testing"):
                # Generate a test key for testing
                self.SECRET_KEY = (
                    "test-secret-key-for-testing-only-not-for-production-use-32chars"  # nosec B105 # noqa: S105
                )
            else:
                raise ValueError("JWT_SECRET_KEY environment variable must be set")
        if len(self.SECRET_KEY) < MIN_SECRET_KEY_LENGTH:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")


class AuthUser(BaseModel):
    """Authenticated user information."""

    id: UUID
    email: str
    name: str
    subscription_tier: str
    is_active: bool = True
    permissions: list[str] = []


class AuthManager:
    """Authentication manager for JWT tokens and password hashing."""

    def __init__(self, config: AuthConfig | None = None):
        """Initialize authentication manager."""
        self.config = config or AuthConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return str(self.pwd_context.hash(password))

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bool(self.pwd_context.verify(plain_password, hashed_password))

    def create_access_token(
        self,
        user_id: UUID,
        email: str,
        subscription_tier: str = "free",
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a JWT access token."""
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(
                minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES
            )

        to_encode = {
            "sub": str(user_id),
            "email": email,
            "subscription_tier": subscription_tier,
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "access",
        }

        return str(
            jwt.encode(
                to_encode, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM
            )
        )

    def create_refresh_token(self, user_id: UUID) -> str:
        """Create a JWT refresh token."""
        expire = datetime.now(UTC) + timedelta(
            days=self.config.REFRESH_TOKEN_EXPIRE_DAYS
        )

        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "refresh",
        }

        return str(
            jwt.encode(
                to_encode, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM
            )
        )

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM]
            )
            return dict(payload)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None  # Don't leak internal details
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None  # Don't leak internal details

    async def get_current_user(
        self, credentials: HTTPAuthorizationCredentials
    ) -> AuthUser:
        """Extract current user from JWT token."""
        token = credentials.credentials
        payload = self.verify_token(token)

        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
            )

        user_id = payload.get("sub")
        email = payload.get("email")
        subscription_tier = payload.get("subscription_tier", "free")

        if not user_id or not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
            )

        # In production, you would fetch user from database here
        # For now, we'll create from token payload
        return AuthUser(
            id=UUID(user_id),
            email=email,
            name="User",  # Would be fetched from DB
            subscription_tier=subscription_tier,
            is_active=True,
            permissions=self._get_user_permissions(subscription_tier),
        )

    def _get_user_permissions(self, subscription_tier: str) -> list[str]:
        """Get user permissions based on subscription tier."""
        base_permissions = ["asset:create", "asset:read", "asset:export"]

        if subscription_tier == "pro":
            return base_permissions + [
                "asset:batch",
                "asset:advanced_export",
                "analytics:basic",
            ]
        elif subscription_tier == "enterprise":
            return base_permissions + [
                "asset:batch",
                "asset:advanced_export",
                "analytics:full",
                "admin:users",
                "plugin:install",
            ]

        return base_permissions


class RoleChecker:
    """Dependency for checking user roles and permissions."""

    def __init__(self, required_permissions: list[str]):
        """Initialize role checker with required permissions."""
        self.required_permissions = required_permissions

    def __call__(self, user: AuthUser) -> AuthUser:
        """Check if user has required permissions."""
        missing_permissions = set(self.required_permissions) - set(user.permissions)

        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(missing_permissions)}",
            )

        return user


class APIKeyManager:
    """Manager for API key authentication."""

    def __init__(self):
        """Initialize API key manager."""
        self.api_keys: dict[str, AuthUser] = {}  # In production, use database

    def create_api_key(self, user: AuthUser) -> str:
        """Create a new API key for user."""
        api_key = f"ll3m_{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = user
        return api_key

    def verify_api_key(self, api_key: str) -> AuthUser | None:
        """Verify and return user for API key."""
        return self.api_keys.get(api_key)


# Global instances - lazy loaded to avoid import-time configuration issues
_auth_manager = None
_api_key_manager = None


def get_auth_manager() -> AuthManager:
    """Get or create the global auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def get_api_key_manager() -> APIKeyManager:
    """Get or create the global API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


# For backward compatibility, provide module-level access
auth_manager = get_auth_manager
api_key_manager = get_api_key_manager

http_bearer = HTTPBearer()
http_bearer_optional = HTTPBearer(auto_error=False)
# Pre-computed dependency instances to avoid B008 linting errors
_http_bearer_dep = Depends(http_bearer)
_http_bearer_optional_dep = Depends(http_bearer_optional)


# Common permission dependencies
require_asset_create = RoleChecker(["asset:create"])
require_asset_batch = RoleChecker(["asset:batch"])
require_analytics = RoleChecker(["analytics:basic"])
require_admin = RoleChecker(["admin:users"])


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = _http_bearer_dep,
) -> AuthUser:
    """Get current authenticated user."""
    return await auth_manager().get_current_user(credentials)


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = _http_bearer_optional_dep,
) -> AuthUser | None:
    """Get current user if authenticated, otherwise None."""
    if not credentials:
        return None

    try:
        return await auth_manager().get_current_user(credentials)
    except HTTPException:
        return None


def create_test_user() -> AuthUser:
    """Create a test user for development."""
    return AuthUser(
        id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        email="test@example.com",
        name="Test User",
        subscription_tier="pro",
        is_active=True,
        permissions=auth_manager()._get_user_permissions("pro"),
    )
