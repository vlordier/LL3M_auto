"""Mock service provider for dependency injection in tests."""

from src.services.blender_service import BlenderService
from src.services.knowledge_service import KnowledgeService

from .mock_blender_service import MockBlenderService
from .mock_knowledge_service import MockKnowledgeService


class MockServiceProvider:
    """Manages mock service instances for testing."""

    def __init__(self) -> None:
        """Initialize the mock service provider."""
        self._blender_service = MockBlenderService()
        self._knowledge_service = MockKnowledgeService()

    @property
    def blender_service(self) -> BlenderService:
        """Get the mock Blender service instance."""
        return self._blender_service

    @property
    def knowledge_service(self) -> KnowledgeService:
        """Get the mock knowledge service instance."""
        return self._knowledge_service


# Global mock service provider instance
mock_services = MockServiceProvider()
