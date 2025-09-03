"""Authentication and authorization system for LL3M API."""

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from jose import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel


class AuthConfig(BaseModel):
    """Authentication configuration."""

    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 8


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
        self.config = config or AuthConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

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

        return jwt.encode(
            to_encode, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM
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

        return jwt.encode(
            to_encode, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM
        )

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_current_user(
        self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
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
        self.required_permissions = required_permissions

    def __call__(
        self, user: AuthUser = Depends(AuthManager().get_current_user)
    ) -> AuthUser:
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
        self.api_keys: dict[str, AuthUser] = {}  # In production, use database

    def create_api_key(self, user: AuthUser) -> str:
        """Create a new API key for user."""
        api_key = f"ll3m_{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = user
        return api_key

    def verify_api_key(self, api_key: str) -> AuthUser | None:
        """Verify and return user for API key."""
        return self.api_keys.get(api_key)


# Global instances
auth_manager = AuthManager()
api_key_manager = APIKeyManager()


# Common permission dependencies
require_asset_create = RoleChecker(["asset:create"])
require_asset_batch = RoleChecker(["asset:batch"])
require_analytics = RoleChecker(["analytics:basic"])
require_admin = RoleChecker(["admin:users"])


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> AuthUser:
    """Get current authenticated user."""
    return await auth_manager.get_current_user(credentials)


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> AuthUser | None:
    """Get current user if authenticated, otherwise None."""
    if not credentials:
        return None

    try:
        return await auth_manager.get_current_user(credentials)
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
        permissions=auth_manager._get_user_permissions("pro"),
    )
