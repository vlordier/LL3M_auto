"""Mock implementation of the knowledge service for testing."""

from src.services.knowledge_service import KnowledgeService
from src.utils.types import AgentResponse, AgentType


class MockKnowledgeService(KnowledgeService):
    """Mock implementation of the knowledge service for testing."""

    def __init__(self) -> None:
        """Initialize the mock knowledge service."""
        self.should_succeed = True
        self.docs = "Sample documentation."

    async def resolve_library_id(self, library_name: str) -> str | None:  # noqa: ARG002
        """Simulate resolving a library ID."""
        if self.should_succeed:
            return "/blender/python-api"
        else:
            return None

    async def get_library_docs(self, library_id: str, topic: str | None = None, tokens: int = 10000) -> str | None:  # noqa: ARG002
        """Simulate fetching library documentation."""
        if self.should_succeed:
            return self.docs
        else:
            return None

    async def search(self, query: str, library_id: str) -> AgentResponse:  # noqa: ARG002
        """Simulate searching for information."""
        if self.should_succeed:
            return AgentResponse(
                agent_type=AgentType.RETRIEVAL,
                success=True,
                data=self.docs,
                message="Documentation found.",
                execution_time=0.1,
            )
        else:
            return AgentResponse(
                agent_type=AgentType.RETRIEVAL,
                success=False,
                data=None,
                message="No relevant documentation found.",
                execution_time=0.1,
            )
