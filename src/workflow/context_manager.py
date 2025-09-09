"""Context management utilities for LangGraph workflows to prevent token limit errors."""

import json
from typing import Any

from ..utils.types import WorkflowState


class ContextCompactor:
    """Manages context compression and token monitoring for LangGraph workflows."""

    def __init__(
        self, max_tokens: int = 100000, compression_threshold: float = 0.85
    ):  # Adjusted for GPT-5's 128k context
        """Initialize context compactor.

        Args:
            max_tokens: Maximum tokens before compression is triggered
            compression_threshold: Fraction of max_tokens that triggers compression
        """
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.compression_trigger = int(max_tokens * compression_threshold)

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 characters)."""
        return len(text) // 4

    def should_compress(self, state: WorkflowState) -> bool:
        """Check if state needs compression based on token count."""
        total_tokens = self._calculate_state_tokens(state)
        return total_tokens > self.compression_trigger

    def _calculate_state_tokens(self, state: WorkflowState) -> int:
        """Calculate approximate total tokens in state."""
        token_count = 0

        # Core content
        token_count += self.estimate_tokens(state.prompt or "")
        token_count += self.estimate_tokens(state.generated_code or "")
        token_count += self.estimate_tokens(state.documentation or "")

        # Analysis results
        if state.critic_analysis:
            token_count += self.estimate_tokens(json.dumps(state.critic_analysis, default=str))

        if state.verification_result:
            token_count += self.estimate_tokens(json.dumps(state.verification_result, default=str))

        # Subtasks
        if state.subtasks:
            for subtask in state.subtasks:
                token_count += self.estimate_tokens(subtask.description)
                token_count += self.estimate_tokens(json.dumps(subtask.parameters, default=str))

        return token_count

    def compress_state(self, state: WorkflowState) -> WorkflowState:
        """Compress state by summarizing and truncating content."""
        # Create compressed copy
        compressed_state = state.model_copy()

        # Compress documentation (keep only essential parts)
        if state.documentation:
            compressed_state.documentation = self._compress_documentation(state.documentation)

        # Compress generated code (keep main functions only)
        if state.generated_code:
            compressed_state.generated_code = self._compress_code(state.generated_code)

        # Compress analysis results
        if state.critic_analysis:
            compressed_state.critic_analysis = self._compress_analysis(state.critic_analysis)

        if state.verification_result:
            compressed_state.verification_result = self._compress_analysis(state.verification_result)

        # Keep only high-priority subtasks
        if state.subtasks:
            compressed_state.subtasks = self._compress_subtasks(state.subtasks)

        return compressed_state

    def _compress_documentation(self, doc: str) -> str:
        """Compress documentation by keeping only key sections."""
        lines = doc.split("\n")

        # Keep headers and first paragraph of each section
        compressed_lines = []
        keep_next = True

        for line in lines:
            if line.startswith("#") or line.startswith("##"):
                compressed_lines.append(line)
                keep_next = True
            elif keep_next and line.strip():
                compressed_lines.append(line)
                keep_next = False
            elif not line.strip():
                keep_next = True

        return "\n".join(compressed_lines[:100])  # Limit to 100 lines

    def _compress_code(self, code: str) -> str:
        """Compress code by keeping main functions and removing comments."""
        lines = code.split("\n")
        compressed_lines = []

        for line in lines:
            # Keep imports and function definitions
            if (
                line.strip().startswith("import ")
                or line.strip().startswith("from ")
                or line.strip().startswith("def ")
                or line.strip().startswith("class ")
                or "bpy.ops" in line
                or "bpy.context" in line
            ):
                compressed_lines.append(line)

        return "\n".join(compressed_lines[:50])  # Limit to 50 key lines

    def _compress_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Compress analysis by keeping only essential fields."""
        essential_fields = {
            "overall_score",
            "needs_refinement",
            "critical_issues",
            "quality_score",
            "recommendations",
            "improvement_suggestions",
        }

        return {k: v for k, v in analysis.items() if k in essential_fields}

    def _compress_subtasks(self, subtasks: list[Any]) -> list[Any]:
        """Keep only high-priority subtasks."""
        # Sort by priority and keep top 3
        sorted_tasks = sorted(subtasks, key=lambda x: getattr(x, "priority", 0), reverse=True)
        return sorted_tasks[:3]

    def monitor_and_compress(self, state: WorkflowState) -> WorkflowState:
        """Monitor state size and compress if needed."""
        if self.should_compress(state):
            print(f"🔧 Context compression triggered (>{self.compression_trigger} tokens)")
            compressed_state = self.compress_state(state)

            original_tokens = self._calculate_state_tokens(state)
            compressed_tokens = self._calculate_state_tokens(compressed_state)

            print(
                f"📉 Reduced from ~{original_tokens} to ~{compressed_tokens} tokens "
                f"({compressed_tokens / original_tokens:.1%} of original)"
            )

            return compressed_state

        return state


class ContextMonitor:
    """Monitors context usage and provides warnings."""

    def __init__(self, max_tokens: int = 128000):  # GPT-5 context window
        self.max_tokens = max_tokens
        self.compactor = ContextCompactor(max_tokens)

    def check_and_warn(self, state: WorkflowState, stage: str = "") -> None:
        """Check context size and warn if approaching limits."""
        tokens = self.compactor._calculate_state_tokens(state)
        percentage = (tokens / self.max_tokens) * 100

        if percentage > 80:
            print(f"⚠️  Context warning at {stage}: {tokens}/{self.max_tokens} tokens ({percentage:.1f}%)")
        elif percentage > 60:
            print(f"📊 Context info at {stage}: {tokens}/{self.max_tokens} tokens ({percentage:.1f}%)")
