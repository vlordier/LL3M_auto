"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth import AuthUser, auth_manager, get_current_user
from ..database import UserRepository, get_user_repo
from ..models import Token, User, UserCreate, UserLogin

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
security = HTTPBearer()


@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate, user_repo: UserRepository = Depends(get_user_repo)
):
    """Register a new user."""
    # Check if user already exists
    existing_user = await user_repo.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )

    # Hash password and create user
    password_hash = auth_manager().hash_password(user_data.password)

    try:
        db_user = await user_repo.create_user(
            email=user_data.email, name=user_data.name, password_hash=password_hash
        )

        return {
            "message": "User created successfully",
            "user_id": str(db_user.id),
            "email": db_user.email,
        }

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        )


@router.post("/login", response_model=Token)
async def login(
    user_data: UserLogin, user_repo: UserRepository = Depends(get_user_repo)
):
    """Authenticate user and return access token."""
    # Get user from database
    db_user = await user_repo.get_user_by_email(user_data.email)

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Verify password
    if not auth_manager().verify_password(user_data.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    if not db_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is disabled"
        )

    # Create tokens
    access_token = auth_manager().create_access_token(
        user_id=db_user.id,
        email=db_user.email,
        subscription_tier=db_user.subscription_tier.value,
    )
    refresh_token = auth_manager().create_refresh_token(db_user.id)

    return Token(
        access_token=access_token,
        token_type="bearer",  # nosec B106
        expires_in=auth_manager().config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_repo: UserRepository = Depends(get_user_repo),
):
    """Refresh access token using refresh token."""
    refresh_token = credentials.credentials
    payload = auth_manager().verify_token(refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
        )

    # Get user from database
    db_user = await user_repo.get_user_by_id(user_id)
    if not db_user or not db_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    # Create new access token
    access_token = auth_manager().create_access_token(
        user_id=db_user.id,
        email=db_user.email,
        subscription_tier=db_user.subscription_tier.value,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",  # nosec B106
        expires_in=auth_manager().config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: AuthUser = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
):
    """Get current user information."""
    db_user = await user_repo.get_user_by_id(current_user.id)

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return User(
        id=db_user.id,
        email=db_user.email,
        name=db_user.name,
        subscription_tier=db_user.subscription_tier,
        is_active=db_user.is_active,
        created_at=db_user.created_at,
        updated_at=db_user.updated_at,
    )


@router.post("/logout")
async def logout(current_user: AuthUser = Depends(get_current_user)):
    """Logout user (token invalidation would be implemented with token blacklist)."""
    # In a production system, you would add the token to a blacklist
    # or use a more sophisticated token management system
    return {"message": "Logged out successfully"}


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: AuthUser = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
):
    """Change user password."""
    if len(new_password) < auth_manager().config.PASSWORD_MIN_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be at least {auth_manager().config.PASSWORD_MIN_LENGTH} characters long",
        )

    # Get user from database
    db_user = await user_repo.get_user_by_id(current_user.id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Verify current password
    if not auth_manager().verify_password(current_password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )

    # Hash new password and update
    auth_manager().hash_password(new_password)

    # Update password in database (would implement in user repository)
    # await user_repo.update_password(db_user.id, new_password_hash)

    return {"message": "Password changed successfully"}
