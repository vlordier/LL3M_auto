"""Service provider for dependency injection."""

from .blender_service import BlenderService
from .blender_service_impl import BlenderServiceImpl
from .knowledge_service import KnowledgeService
from .knowledge_service_impl import Context7KnowledgeService


class ServiceProvider:
    """Manages service instances."""

    def __init__(self) -> None:
        """Initialize the service provider."""
        self._blender_service = BlenderServiceImpl()
        self._knowledge_service = Context7KnowledgeService()

    @property
    def blender_service(self) -> BlenderService:
        """Get the Blender service instance."""
        return self._blender_service

    @property
    def knowledge_service(self) -> KnowledgeService:
        """Get the knowledge service instance."""
        return self._knowledge_service


# Global service provider instance
services = ServiceProvider()
