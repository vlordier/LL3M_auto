"""Quality assessment workflow stubs for testing.

This module exposes factory functions referenced by tests for patching.
"""

from collections.abc import Callable


def create_quality_assessment_workflow() -> Callable:  # pragma: no cover - stub
    async def _noop(state):
        return state

    return _noop

