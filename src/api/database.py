"""Database models and connection management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func

from .models import AssetStatus, ExportFormat, SubscriptionTier


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class DBUser(Base):
    """User database model."""

    __tablename__ = "users"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    subscription_tier: Column[SubscriptionTier] = Column(
        SQLEnum(SubscriptionTier), default=SubscriptionTier.FREE, nullable=False
    )
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    assets = relationship(
        "DBAsset", back_populates="user", cascade="all, delete-orphan"
    )
    generation_jobs = relationship("DBGenerationJob", back_populates="user")


class DBAsset(Base):
    """Asset database model."""

    __tablename__ = "assets"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    status: Column[AssetStatus] = Column(
        SQLEnum(AssetStatus), default=AssetStatus.PENDING, nullable=False
    )
    blender_file_url = Column(Text)
    preview_image_url = Column(Text)
    asset_metadata = Column(JSON)
    file_size = Column(BigInteger)
    polygon_count = Column(Integer)
    generation_time = Column(Integer)  # seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user = relationship("DBUser", back_populates="assets")
    generation_jobs = relationship("DBGenerationJob", back_populates="asset")
    exports = relationship("DBAssetExport", back_populates="asset")


class DBGenerationJob(Base):
    """Generation job database model."""

    __tablename__ = "generation_jobs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    asset_id = Column(PGUUID(as_uuid=True), ForeignKey("assets.id"), nullable=False)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status: Column[AssetStatus] = Column(
        SQLEnum(AssetStatus), default=AssetStatus.PENDING, nullable=False
    )
    progress = Column(Integer, default=0)
    current_step = Column(String(255))
    error_message = Column(Text)
    execution_log = Column(Text)
    generated_code = Column(Text)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    estimated_completion = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    asset = relationship("DBAsset", back_populates="generation_jobs")
    user = relationship("DBUser", back_populates="generation_jobs")


class DBAssetExport(Base):
    """Asset export database model."""

    __tablename__ = "asset_exports"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    asset_id = Column(PGUUID(as_uuid=True), ForeignKey("assets.id"), nullable=False)
    format: Column[ExportFormat] = Column(SQLEnum(ExportFormat), nullable=False)
    file_url = Column(Text, nullable=False)
    file_size = Column(BigInteger)
    quality = Column(String(50))
    settings = Column(JSON)
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    asset = relationship("DBAsset", back_populates="exports")


class DBBatch(Base):
    """Batch processing database model."""

    __tablename__ = "batches"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")
    total_assets = Column(Integer, nullable=False)
    completed_assets = Column(Integer, default=0)
    failed_assets = Column(Integer, default=0)
    priority = Column(Integer, default=1)
    notify_on_completion = Column(Boolean, default=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    estimated_completion = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("DBUser")


class DatabaseManager:
    """Database connection and session management."""

    def __init__(self, database_url: str):
        """Initialize database manager with connection URL."""
        self.engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL logging
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager."""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def create_tables(self):
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self):
        """Drop all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self):
        """Close database connection."""
        await self.engine.dispose()


class UserRepository:
    """Repository for user operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize user repository with database manager."""
        self.db = db_manager

    async def create_user(
        self,
        email: str,
        name: str,
        password_hash: str,
        subscription_tier: SubscriptionTier = SubscriptionTier.FREE,
    ) -> DBUser:
        """Create a new user."""
        async with self.db.get_session() as session:
            user = DBUser(
                email=email,
                name=name,
                password_hash=password_hash,
                subscription_tier=subscription_tier,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def get_user_by_id(self, user_id: UUID) -> DBUser | None:
        """Get user by ID."""
        async with self.db.get_session() as session:
            result: DBUser | None = await session.get(DBUser, user_id)
            return result

    async def get_user_by_email(self, email: str) -> DBUser | None:
        """Get user by email."""
        async with self.db.get_session() as session:
            from sqlalchemy import select

            result = await session.execute(select(DBUser).where(DBUser.email == email))
            user: DBUser | None = result.scalar_one_or_none()
            return user


class AssetRepository:
    """Repository for asset operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize asset repository with database manager."""
        self.db = db_manager

    async def create_asset(
        self, user_id: UUID, name: str, prompt: str, metadata: dict | None = None
    ) -> DBAsset:
        """Create a new asset."""
        async with self.db.get_session() as session:
            asset = DBAsset(
                user_id=user_id, name=name, prompt=prompt, asset_metadata=metadata or {}
            )
            session.add(asset)
            await session.commit()
            await session.refresh(asset)
            return asset

    async def get_asset_by_id(self, asset_id: UUID) -> DBAsset | None:
        """Get asset by ID."""
        async with self.db.get_session() as session:
            result: DBAsset | None = await session.get(DBAsset, asset_id)
            return result

    async def update_asset_status(
        self,
        asset_id: UUID,
        status: AssetStatus,
        blender_file_url: str | None = None,
        preview_image_url: str | None = None,
    ) -> DBAsset | None:
        """Update asset status and file URLs."""
        async with self.db.get_session() as session:
            asset: DBAsset | None = await session.get(DBAsset, asset_id)
            if asset:
                asset.status = status
                if blender_file_url:
                    asset.blender_file_url = blender_file_url
                if preview_image_url:
                    asset.preview_image_url = preview_image_url
                await session.commit()
                await session.refresh(asset)
            return asset

    async def get_user_assets(
        self, user_id: UUID, limit: int = 50, offset: int = 0
    ) -> list[DBAsset]:
        """Get user's assets with pagination."""
        async with self.db.get_session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(DBAsset)
                .where(DBAsset.user_id == user_id)
                .order_by(DBAsset.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())


# Global database instance (will be initialized in app startup)
db_manager: DatabaseManager | None = None
user_repo: UserRepository | None = None
asset_repo: AssetRepository | None = None


def get_db_manager() -> DatabaseManager:
    """Get global database manager."""
    if db_manager is None:
        raise RuntimeError("Database not initialized")
    return db_manager


def get_user_repo() -> UserRepository:
    """Get user repository."""
    if user_repo is None:
        raise RuntimeError("User repository not initialized")
    return user_repo


def get_asset_repo() -> AssetRepository:
    """Get asset repository."""
    if asset_repo is None:
        raise RuntimeError("Asset repository not initialized")
    return asset_repo
