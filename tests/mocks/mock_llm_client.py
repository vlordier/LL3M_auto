from typing import Any


class MockLLMClient:
    """A mock LLM client for testing purposes."""

    def __init__(
        self,
        api_key: str | None = "mock_api_key",
        base_url: str | None = "http://localhost:1234/v1",
    ):
        self.should_succeed = True
        self.mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": """
{
    "asset_analysis": {
        "complexity_level": 2,
        "estimated_components": "1",
        "primary_challenges": ["Creating a primitive shape"],
        "recommended_approach": "Use bpy.ops.mesh.primitive_cube_add."
    },
    "tasks": [
        {
            "id": "create_cube",
            "type": "geometry",
            "description": "Create a simple red cube.",
            "priority": 1,
            "dependencies": [],
            "parameters": {
                "shape": "cube",
                "size": 2,
                "location": [0, 0, 0],
                "material": {
                    "name": "RedMaterial",
                    "color": [1.0, 0.0, 0.0, 1.0]
                }
            },
            "validation_criteria": ["A red cube exists at the origin"],
            "estimated_complexity": "low"
        }
    ]
}
""",
                    }
                }
            ],
            "usage": {"total_tokens": 123},
        }

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock chat completion method."""
        if self.should_succeed:
            return self.mock_response
        else:
            raise Exception("Mock LLM client failed as requested.")

    async def close(self) -> None:
        """Mock close method."""
        pass
