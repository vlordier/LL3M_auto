"""LangGraph workflow implementation for LL3M multi-agent system."""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

<<<<<<< HEAD
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
=======
import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel
>>>>>>> origin/master

from ..agents.coding import CodingAgent
from ..agents.planner import PlannerAgent
from ..agents.retrieval import RetrievalAgent
from ..blender.executor import BlenderExecutor
from ..utils.config import settings
from ..utils.types import AssetMetadata, WorkflowState

logger = structlog.get_logger(__name__)


async def planner_node(state: WorkflowState) -> WorkflowState:
    """Execute planner agent."""
    # Initialize original_prompt on first run
    if not state.original_prompt:
        state.original_prompt = state.prompt

    planner = PlannerAgent(settings.get_agent_config("planner"))
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
    retrieval = RetrievalAgent(settings.get_agent_config("retrieval"))
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
    coding = CodingAgent(settings.get_agent_config("coding"))
    response = await coding.process(state)

    if response.success:
        state.generated_code = response.data
        await _save_checkpoint(state, "coding_completed")
    else:
        state.error_message = response.message
        state.should_continue = False

    return state


async def execution_node(state: WorkflowState) -> WorkflowState:
    """Execute generated code in Blender."""
    executor = BlenderExecutor()

    try:
        result = await executor.execute_code(
            state.generated_code,
            asset_name=f"asset_{uuid.uuid4()}",
        )

        state.execution_result = result

        if result.success:
            state.asset_metadata = AssetMetadata(
                id=f"asset_{uuid.uuid4()}",
                prompt=state.original_prompt or state.prompt,
                file_path=result.asset_path or "unknown",
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


async def refinement_node(state: WorkflowState) -> WorkflowState:
    """Handle refinement requests and iterative improvements."""
    if not state.refinement_request or state.refinement_count >= 3:
        state.should_continue = False
        return state

    # Increment refinement counter
<<<<<<< HEAD
    state.refinement_iterations += 1
=======
    state.refinement_count += 1
>>>>>>> origin/master

    # Reset error state for retry
    state.error_message = ""
    state.should_continue = True

    # Update prompt with refinement request
    state.prompt = (
        f"{state.original_prompt}\n\nRefinement request: {state.refinement_request}"
    )

    return state


async def validation_node(state: WorkflowState) -> WorkflowState:
    """Validate execution results and determine if refinement is needed."""
    if not state.execution_result:
        state.error_message = "No execution result to validate"
        state.should_continue = False
        return state

    # Check for common failure patterns
    validation_issues = []

    if state.execution_result and not state.execution_result.success:
        validation_issues.extend(state.execution_result.errors)

    # Add validation logic for asset quality
    if state.asset_metadata:
        if not state.asset_metadata.file_path:
            validation_issues.append("No asset file generated")
        if not state.asset_metadata.screenshot_path:
            validation_issues.append("No screenshot generated")

    if validation_issues:
        state.refinement_request = f"Fix issues: {'; '.join(validation_issues)}"
        state.needs_refinement = True
    else:
        state.needs_refinement = False
        state.should_continue = False

    return state


def should_continue_main(state: WorkflowState) -> Literal["end", "continue"]:
    """Determine if main workflow should continue."""
    if not state.should_continue or state.error_message:
        return "end"
    return "continue"


def should_refine(state: WorkflowState) -> Literal["refine", "complete", "end"]:
    """Determine if refinement is needed after execution."""
    if state.error_message:
        return "end"

<<<<<<< HEAD
    if state.needs_refinement and state.refinement_iterations < 3:
=======
    if state.needs_refinement and state.refinement_count < 3:
>>>>>>> origin/master
        return "refine"

    return "complete"


def create_initial_workflow() -> Pregel[WorkflowState, None, Any]:
    """Create the initial creation workflow."""
<<<<<<< HEAD
    workflow = StateGraph(WorkflowState)

    # Add main workflow nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("execution", execution_node)

    # Add refinement nodes
    workflow.add_node("validation", validation_node)
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

    # Post-execution validation and refinement
    workflow.add_edge("execution", "validation")
    workflow.add_conditional_edges(
        "validation",
        should_refine,
        {
            "refine": "refinement",
            "complete": END,
            "end": END
        }
    )

    # Refinement loop back to planner
    workflow.add_edge("refinement", "planner")

    # Add memory saver for state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
=======
    return _create_workflow_internal({"enable_refinement": True})
>>>>>>> origin/master


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
        logger.warning(
            "Failed to save checkpoint", checkpoint_name=checkpoint_name, error=str(e)
        )


async def _load_checkpoint(checkpoint_file: str) -> WorkflowState:
    """Load workflow state from checkpoint file."""
    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)

    return WorkflowState(**checkpoint_data["state"])


def create_ll3m_workflow() -> Pregel[WorkflowState, None, Any]:
    """Create the main LL3M workflow (alias for initial workflow)."""
    return create_initial_workflow()


def create_workflow_with_config(
    config: dict[str, Any],
) -> Pregel[WorkflowState, None, Any]:
    """Create workflow with custom configuration."""
    return _create_workflow_internal(config)


def _create_workflow_internal(
    config: dict[str, Any],
) -> Pregel[WorkflowState, None, Any]:
    """Internal function to create workflow with configuration."""
    workflow = StateGraph(WorkflowState)

    # Add main workflow nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("execution", execution_node)

    # Add refinement nodes
    workflow.add_node("validation", validation_node)
    workflow.add_node("refinement", refinement_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Configure edges based on config
    if config.get("enable_refinement", True):
        # Full workflow with refinement
        workflow.add_conditional_edges(
            "planner", should_continue_main, {"continue": "retrieval", "end": END}
        )
        workflow.add_conditional_edges(
            "retrieval", should_continue_main, {"continue": "coding", "end": END}
        )
        workflow.add_conditional_edges(
            "coding", should_continue_main, {"continue": "execution", "end": END}
        )
        workflow.add_edge("execution", "validation")
        workflow.add_conditional_edges(
            "validation",
            should_refine,
<<<<<<< HEAD
            {
                "refine": "refinement",
                "complete": END,
                "end": END
            }
=======
            {"refine": "refinement", "complete": END, "end": END},
>>>>>>> origin/master
        )
        workflow.add_edge("refinement", "planner")
    else:
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
<<<<<<< HEAD
    return workflow.compile(checkpointer=memory)
=======
    compiled = workflow.compile(checkpointer=memory)
    return compiled
>>>>>>> origin/master
