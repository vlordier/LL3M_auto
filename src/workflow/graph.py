"""Shim module to support tests that patch workflow graph helpers.

This provides minimal implementations used in tests:
- `_save_checkpoint` async function (no-op), patched in tests to assert calls
- `EnhancedBlenderExecutor` re-export for patching
"""

from collections.abc import Awaitable
from typing import Any

# Re-export the executor so tests can patch it via this path
try:
    from src.blender.enhanced_executor import EnhancedBlenderExecutor  # type: ignore
except Exception:  # pragma: no cover - in test contexts, just provide a placeholder
    class EnhancedBlenderExecutor:  # type: ignore
        async def execute_code(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401, ANN401
            return {"success": False, "errors": ["executor unavailable"]}


async def _save_checkpoint(_state: Any, *_args: Any, **_kwargs: Any) -> Awaitable[None]:
    """No-op checkpoint saver used for testing."""
    return None

