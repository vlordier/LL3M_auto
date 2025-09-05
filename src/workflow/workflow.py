"""Unified LangGraph workflow implementation for LL3M multi-agent system."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..agents import get_agent_factory
from ..agents.factory import AgentFactory
from ..blender.enhanced_executor import EnhancedBlenderExecutor
from ..utils.config import Settings, get_settings
from ..utils.types import AssetMetadata, WorkflowState

# Shared instances to reduce duplication
_agent_factory: AgentFactory | None = None
_settings: Settings | None = None


def _get_agent_factory() -> AgentFactory:
    """Get shared agent factory instance."""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = get_agent_factory()
    return _agent_factory


def _get_settings() -> Settings:
    """Get shared settings instance."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _generate_asset_name() -> str:
    """Generate unique asset name."""
    return f"asset_{int(asyncio.get_event_loop().time())}"


def _generate_asset_id() -> str:
    """Generate unique asset ID."""
    return f"asset_{int(time.time())}"


def _add_linear_edges(workflow: StateGraph[WorkflowState], nodes: list[str]) -> None:
    """Add linear conditional edges between nodes."""
    for i in range(len(nodes) - 1):
        workflow.add_conditional_edges(nodes[i], should_continue_main, {"continue": nodes[i + 1], "end": END})


# Core workflow nodes
async def planner_node(state: WorkflowState) -> WorkflowState:
    """Execute planner agent."""
    if not state.original_prompt:
        state.original_prompt = state.prompt

    planner = _get_agent_factory().create_planner()
    response = await planner.process(state)

    if response.success:
        state.subtasks = response.data
        await _save_checkpoint(state, "planner_completed")
    else:
        state.error_message = response.message
        state.should_continue = False

    return state


async def retrieval_node(state: WorkflowState) -> WorkflowState:
    """Execute retrieval agent."""
    retrieval = _get_agent_factory().create_retrieval()
    response = await retrieval.process(state)

    if response.success:
        state.documentation = response.data
        await _save_checkpoint(state, "retrieval_completed")
    else:
        state.error_message = response.message
        state.should_continue = False

    return state


async def coding_node(state: WorkflowState) -> WorkflowState:
    """Execute coding agent."""
    coding = _get_agent_factory().create_coding()
    response = await coding.process(state)

    if response.success:
        state.generated_code = response.data
        await _save_checkpoint(state, "coding_completed")
    else:
        state.error_message = response.message
        state.should_continue = False

    return state


async def execution_node(state: WorkflowState) -> WorkflowState:
    """Execute generated code using enhanced Blender executor."""
    executor = EnhancedBlenderExecutor()

    try:
        # Store previous screenshot for comparison if this is a refinement
        if hasattr(state, "execution_result") and state.execution_result:
            state.previous_screenshot_path = state.execution_result.screenshot_path

        result = await executor.execute_code(
            state.generated_code,
            asset_name=_generate_asset_name(),
            export_formats=["blend", "obj"],
            validate_code=True,
            quality_settings={
                "render_engine": "EEVEE",
                "render_samples": 64,
                "screenshot_width": 1024,
                "screenshot_height": 1024,
            },
        )

        state.execution_result = result

        if result.success:
            state.asset_metadata = AssetMetadata(
                id=_generate_asset_id(),
                prompt=state.original_prompt or state.prompt,
                file_path=result.asset_path or "",
                screenshot_path=result.screenshot_path,
                subtasks=state.subtasks,
                quality_score=None,
            )
            await _save_checkpoint(state, "execution_completed")
        else:
            state.error_message = f"Execution failed: {'; '.join(result.errors)}"

    except Exception as e:
        state.error_message = f"Execution error: {e!s}"
        state.should_continue = False

    return state


# Quality assessment nodes (only used in full workflow)
async def critic_node(state: WorkflowState) -> WorkflowState:
    """Execute critic agent for visual analysis."""
    critic = _get_agent_factory().create_critic()
    response = await critic.process(state)

    if response.success:
        state.critic_analysis = response.data
        state.needs_refinement = response.metadata.get("needs_refinement", False)
        state.refinement_priority = response.metadata.get("refinement_priority", "low")
        await _save_checkpoint(state, "critic_completed")
    else:
        state.error_message = f"Critic analysis failed: {response.message}"
        state.should_continue = False

    return state


async def verification_node(state: WorkflowState) -> WorkflowState:
    """Execute verification agent for quality assessment."""
    verification = _get_agent_factory().create_verification()
    response = await verification.process(state)

    if response.success:
        state.verification_result = response.data

        # Combine verification and critic results for refinement decision
        verification_score = response.data.get("quality_score", 10.0)
        has_critical_issues = response.metadata.get("critical_issues", 0) > 0

        # Update refinement need based on verification
        settings = _get_settings()
        if verification_score < settings.app.min_verification_score or has_critical_issues:
            state.needs_refinement = True
            state.refinement_priority = 1.0 if has_critical_issues else 0.5

        await _save_checkpoint(state, "verification_completed")
    else:
        state.error_message = f"Verification failed: {response.message}"
        state.should_continue = False

    return state


async def quality_assessment_node(state: WorkflowState) -> WorkflowState:
    """Combined quality assessment using both critic and verification results."""
    critic_analysis = getattr(state, "critic_analysis", {})
    verification_result = getattr(state, "verification_result", {})

    # Generate refinement recommendations
    refinement_suggestions = []

    if critic_analysis:
        critic_suggestions = critic_analysis.get("improvement_suggestions", [])
        refinement_suggestions.extend(critic_suggestions)

    if verification_result:
        verification_suggestions = verification_result.get("recommendations", [])
        refinement_suggestions.extend(verification_suggestions)

    # Set refinement request based on combined analysis
    if refinement_suggestions:
        state.refinement_request = (
            f"Quality assessment identified the following improvements needed: "
            f"{'; '.join(refinement_suggestions[:3])}"  # Limit to top 3 suggestions
        )

    await _save_checkpoint(state, "quality_assessment_completed")
    return state


async def user_proxy_node(state: WorkflowState) -> WorkflowState:
    """Execute user proxy agent for human-in-the-loop feedback processing."""
    user_proxy = _get_agent_factory().create_user_proxy()
    response = await user_proxy.process(state)

    if response.success:
        proxy_data = response.data
        workflow_action = proxy_data.get("workflow_action", {})

        # Update state based on user proxy decision
        action = workflow_action.get("action", "wait_for_input")

        if action == "complete":
            state.should_continue = False
            state.user_accepted = True
        elif action == "terminate":
            state.should_continue = False
            state.user_terminated = True
        elif action == "refine":
            # Extract refinement instructions
            workflow_action.get("instructions", [])
            refinement_summary = workflow_action.get("reason", "User requested refinements")

            # Update refinement request with structured feedback
            state.refinement_request = refinement_summary
            state.needs_refinement = True
            state.refinement_priority = 1.0 if workflow_action.get("priority") == "high" else 0.5

        # Store user prompt for API responses
        state.user_prompt = proxy_data.get("user_prompt", "")

        await _save_checkpoint(state, "user_proxy_completed")
    else:
        state.error_message = f"User proxy failed: {response.message}"
        state.should_continue = False

    return state


# Refinement nodes
async def refinement_node(state: WorkflowState) -> WorkflowState:
    """Handle refinement requests and iterative improvements."""
    settings = _get_settings()
    if not state.refinement_request or state.refinement_iterations >= settings.app.max_refinement_iterations:
        state.should_continue = False
        return state

    # Increment refinement counter
    state.refinement_iterations += 1

    # Reset error state for retry
    state.error_message = ""
    state.should_continue = True

    # Update prompt with refinement request
    state.prompt = f"{state.original_prompt}\n\nRefinement request: {state.refinement_request}"

    return state


# Decision functions
def should_continue_main(state: WorkflowState) -> Literal["end", "continue"]:
    """Determine if main workflow should continue."""
    if not state.should_continue or state.error_message:
        return "end"
    return "continue"


def should_assess_quality(state: WorkflowState) -> Literal["assess", "end"]:
    """Determine if quality assessment should be performed."""
    if not state.execution_result or not state.execution_result.success:
        return "end"
    return "assess"


def should_refine_enhanced(state: WorkflowState) -> Literal["refine", "complete", "end"]:
    """Enhanced refinement decision based on quality assessment."""
    if state.error_message:
        return "end"

    # Check if refinement is needed and under iteration limit
    settings = _get_settings()
    if state.needs_refinement and state.refinement_iterations < settings.app.max_refinement_iterations:
        # Consider refinement priority
        priority = getattr(state, "refinement_priority", 0.0)
        if priority > 0.0 or state.refinement_iterations == 0:
            return "refine"

    return "complete"


# Workflow creation functions
def create_ll3m_workflow(_config: dict[str, Any] | None = None) -> CompiledStateGraph[WorkflowState]:
    """Create the main LL3M workflow - the most thorough version with full quality assessment.

    Args:
        _config: Optional configuration dict (reserved for future use)
    """
    return _create_full_workflow()


def _create_full_workflow() -> CompiledStateGraph[WorkflowState]:
    """Create the full workflow with comprehensive quality assessment and user feedback."""
    workflow = StateGraph(WorkflowState)

    # Add all workflow nodes for comprehensive processing
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("quality_assessment", quality_assessment_node)
    workflow.add_node("user_proxy", user_proxy_node)
    workflow.add_node("refinement", refinement_node)

    workflow.set_entry_point("planner")

    # Main workflow edges
    _add_linear_edges(workflow, ["planner", "retrieval", "coding", "execution"])

    # Comprehensive quality assessment pipeline
    workflow.add_conditional_edges("execution", should_assess_quality, {"assess": "critic", "end": END})
    workflow.add_edge("critic", "verification")
    workflow.add_edge("verification", "quality_assessment")
    workflow.add_edge("quality_assessment", "user_proxy")
    workflow.add_conditional_edges(
        "user_proxy",
        should_refine_enhanced,
        {"refine": "refinement", "complete": END, "end": END},
    )

    # Refinement loop back to planner
    workflow.add_edge("refinement", "planner")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Checkpoint management
async def _save_checkpoint(state: WorkflowState, checkpoint_name: str) -> None:
    """Save workflow state checkpoint to disk."""
    try:
        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)

        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "timestamp": time.time(),
            "state": state.model_dump(),
        }

        checkpoint_file = checkpoints_dir / f"checkpoint_{checkpoint_name}_{int(time.time())}.json"

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

    except Exception as e:
        # Don't fail workflow on checkpoint save error
        print(f"Warning: Failed to save checkpoint {checkpoint_name}: {e}")


async def _load_checkpoint(checkpoint_file: str) -> WorkflowState:
    """Load workflow state from checkpoint file."""
    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)

    return WorkflowState(**checkpoint_data["state"])


# Legacy compatibility aliases
def create_enhanced_workflow() -> CompiledStateGraph[WorkflowState]:
    """Legacy alias for the full workflow."""
    return _create_full_workflow()


def create_simple_workflow() -> CompiledStateGraph[WorkflowState]:
    """Legacy alias for the full workflow (now redirects to full workflow)."""
    return _create_full_workflow()


def create_fast_workflow() -> CompiledStateGraph[WorkflowState]:
    """Legacy alias for the full workflow (now redirects to full workflow)."""
    return _create_full_workflow()


create_workflow_with_config = create_ll3m_workflow
