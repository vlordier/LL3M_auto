"""Dependency injection container for API components.

Replaces global variables with a proper dependency injection system
for better testability and modularity.
"""

from fastapi import HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from .database import AssetRepository, DatabaseManager, UserRepository


class DatabaseContainer:
    """Container for database-related dependencies."""

    def __init__(self) -> None:
        """Initialize database container with lazy-loaded dependencies."""
        self._db_manager: DatabaseManager | None = None
        self._user_repo: UserRepository | None = None
        self._asset_repo: AssetRepository | None = None

    def initialize(self, database_url: str) -> None:
        """Initialize database components with the given URL."""
        self._db_manager = DatabaseManager(database_url)
        self._user_repo = UserRepository(self._db_manager)
        self._asset_repo = AssetRepository(self._db_manager)

    async def create_tables(self) -> None:
        """Create database tables if they don't exist."""
        if self._db_manager is None:
            raise RuntimeError("Database container not initialized")
        await self._db_manager.create_tables()

    async def close(self) -> None:
        """Close database connections."""
        if self._db_manager:
            await self._db_manager.close()

    def get_db_manager(self) -> DatabaseManager:
        """Get the database manager instance."""
        if self._db_manager is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not initialized")
        return self._db_manager

    def get_user_repository(self) -> UserRepository:
        """Get the user repository instance."""
        if self._user_repo is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="User repository not initialized"
            )
        return self._user_repo

    def get_asset_repository(self) -> AssetRepository:
        """Get the asset repository instance."""
        if self._asset_repo is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Asset repository not initialized"
            )
        return self._asset_repo


class ApplicationContainer:
    """Main dependency injection container for the application."""

    def __init__(self) -> None:
        """Initialize application container with core dependencies."""
        self.database = DatabaseContainer()
        self.rate_limiter = Limiter(key_func=get_remote_address)
        self._initialized = False

    async def initialize(self, database_url: str) -> None:
        """Initialize all application dependencies."""
        self.database.initialize(database_url)
        await self.database.create_tables()
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown and cleanup all dependencies."""
        await self.database.close()
        self._initialized = False

    def ensure_initialized(self) -> None:
        """Ensure the container is properly initialized."""
        if not self._initialized:
            raise RuntimeError("Application container not initialized")


# Global container instance - this is the only global we need
_app_container: ApplicationContainer | None = None


def get_app_container() -> ApplicationContainer:
    """Get the global application container."""
    global _app_container
    if _app_container is None:
        _app_container = ApplicationContainer()
    return _app_container


def set_app_container(container: ApplicationContainer) -> None:
    """Set the global application container (for testing)."""
    global _app_container
    _app_container = container


# FastAPI dependency providers
def get_database_manager() -> DatabaseManager:
    """FastAPI dependency provider for database manager."""
    return get_app_container().database.get_db_manager()


def get_user_repository() -> UserRepository:
    """FastAPI dependency provider for user repository."""
    return get_app_container().database.get_user_repository()


def get_asset_repository() -> AssetRepository:
    """FastAPI dependency provider for asset repository."""
    return get_app_container().database.get_asset_repository()


def get_rate_limiter() -> Limiter:
    """FastAPI dependency provider for rate limiter."""
    return get_app_container().rate_limiter
