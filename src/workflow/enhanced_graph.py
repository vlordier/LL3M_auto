"""Enhanced LangGraph workflow with CriticAgent and VerificationAgent integration."""

import asyncio
import json
import time
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..agents.coding import CodingAgent
from ..agents.critic import CriticAgent
from ..agents.planner import PlannerAgent
from ..agents.retrieval import RetrievalAgent
from ..agents.verification import VerificationAgent
from ..blender.enhanced_executor import EnhancedBlenderExecutor
from ..utils.config import get_settings
from ..utils.types import AssetMetadata, WorkflowState

# Workflow constants
MIN_VERIFICATION_SCORE = 7.0
MAX_REFINEMENT_ITERATIONS = 3


async def planner_node(state: WorkflowState) -> WorkflowState:
    """Execute planner agent."""
    # Initialize original_prompt on first run
    if not state.original_prompt:
        state.original_prompt = state.prompt

    planner = PlannerAgent(get_settings().get_agent_config("planner"))
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
    retrieval = RetrievalAgent(get_settings().get_agent_config("retrieval"))
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
    coding = CodingAgent(get_settings().get_agent_config("coding"))
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
            asset_name=f"asset_{int(asyncio.get_event_loop().time())}",
            export_formats=["blend", "obj"],
            validate_code=True,
            quality_settings={
                "render_engine": "EEVEE",
                "render_samples": 64,
                "screenshot_resolution": [1024, 1024],
            },
        )

        state.execution_result = result

        if result.success:
            state.asset_metadata = AssetMetadata(
                id=f"asset_{int(time.time())}",
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
        state.error_message = f"Execution error: {str(e)}"
        state.should_continue = False

    return state


async def critic_node(state: WorkflowState) -> WorkflowState:
    """Execute critic agent for visual analysis."""
    critic = CriticAgent(get_settings().get_agent_config("critic"))
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
    verification = VerificationAgent(get_settings().get_agent_config("verification"))
    response = await verification.process(state)

    if response.success:
        state.verification_result = response.data

        # Combine verification and critic results for refinement decision
        verification_score = response.data.get("quality_score", 10.0)
        has_critical_issues = response.metadata.get("critical_issues", 0) > 0

        # Update refinement need based on verification
        if verification_score < MIN_VERIFICATION_SCORE or has_critical_issues:
            state.needs_refinement = True
            state.refinement_priority = "high" if has_critical_issues else "medium"

        await _save_checkpoint(state, "verification_completed")
    else:
        state.error_message = f"Verification failed: {response.message}"
        state.should_continue = False

    return state


async def quality_assessment_node(state: WorkflowState) -> WorkflowState:
    """Combined quality assessment using both critic and verification results."""
    # This node combines results from critic and verification agents
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


async def refinement_node(state: WorkflowState) -> WorkflowState:
    """Handle refinement requests and iterative improvements."""
    if not state.refinement_request or state.refinement_iterations >= MAX_REFINEMENT_ITERATIONS:
        state.should_continue = False
        return state

    # Increment refinement counter
    state.refinement_iterations += 1

    # Reset error state for retry
    state.error_message = ""
    state.should_continue = True

    # Update prompt with refinement request
    state.prompt = (
        f"{state.original_prompt}\n\nRefinement request: {state.refinement_request}"
    )

    return state


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


def should_refine_enhanced(
    state: WorkflowState,
) -> Literal["refine", "complete", "end"]:
    """Enhanced refinement decision based on quality assessment."""
    if state.error_message:
        return "end"

    # Check if refinement is needed and under iteration limit
    if state.needs_refinement and state.refinement_iterations < MAX_REFINEMENT_ITERATIONS:
        # Consider refinement priority
        priority = getattr(state, "refinement_priority", "low")
        if priority in ["high", "medium"] or state.refinement_iterations == 0:
            return "refine"

    return "complete"


def create_enhanced_workflow() -> StateGraph:
    """Create the enhanced workflow with critic and verification agents."""
    workflow = StateGraph(WorkflowState)

    # Add main workflow nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("execution", execution_node)

    # Add quality assessment nodes
    workflow.add_node("critic", critic_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("quality_assessment", quality_assessment_node)

    # Add refinement node
    workflow.add_node("refinement", refinement_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Main workflow edges
    workflow.add_conditional_edges(
        "planner", should_continue_main, {"continue": "retrieval", "end": END}
    )

    workflow.add_conditional_edges(
        "retrieval", should_continue_main, {"continue": "coding", "end": END}
    )

    workflow.add_conditional_edges(
        "coding", should_continue_main, {"continue": "execution", "end": END}
    )

    # Quality assessment pipeline
    workflow.add_conditional_edges(
        "execution", should_assess_quality, {"assess": "critic", "end": END}
    )

    workflow.add_edge("critic", "verification")
    workflow.add_edge("verification", "quality_assessment")

    # Refinement decision
    workflow.add_conditional_edges(
        "quality_assessment",
        should_refine_enhanced,
        {"refine": "refinement", "complete": END, "end": END},
    )

    # Refinement loop back to planner
    workflow.add_edge("refinement", "planner")

    # Add memory saver for state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def create_fast_workflow() -> StateGraph:
    """Create a faster workflow without quality assessment for development."""
    workflow = StateGraph(WorkflowState)

    # Add main workflow nodes only
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("execution", execution_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Simple linear workflow
    workflow.add_conditional_edges(
        "planner", should_continue_main, {"continue": "retrieval", "end": END}
    )

    workflow.add_conditional_edges(
        "retrieval", should_continue_main, {"continue": "coding", "end": END}
    )

    workflow.add_conditional_edges(
        "coding", should_continue_main, {"continue": "execution", "end": END}
    )

    workflow.add_edge("execution", END)

    # Add memory saver for state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def create_comparison_workflow() -> StateGraph:
    """Create a workflow specifically for comparing refinement iterations."""
    workflow = StateGraph(WorkflowState)

    # Add analysis nodes only
    workflow.add_node("critic", critic_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("quality_assessment", quality_assessment_node)

    # Set entry point
    workflow.set_entry_point("critic")

    # Analysis pipeline
    workflow.add_edge("critic", "verification")
    workflow.add_edge("verification", "quality_assessment")
    workflow.add_edge("quality_assessment", END)

    # Add memory saver for state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


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

        checkpoint_file = (
            checkpoints_dir / f"checkpoint_{checkpoint_name}_{int(time.time())}.json"
        )

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


# Default exports
create_ll3m_workflow = create_enhanced_workflow
create_workflow_with_config = create_enhanced_workflow
