"""Unified LangGraph workflow implementation for LL3M multi-agent system."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..agents import get_agent_factory
from ..agents.factory import AgentFactory
from ..services import services
from ..utils.config import Settings, get_settings
from ..utils.types import AssetMetadata, WorkflowState
from ..services.artifact_manager import ArtifactManager
from .context_manager import ContextCompactor, ContextMonitor
from .error_recovery import with_error_recovery
from .retry_handler import log_workflow_state, with_retry
from ..blender.enhanced_executor import EnhancedBlenderExecutor

# Initialize logger
logger = structlog.get_logger(__name__)

# Constants for quality assessment
QUALITY_THRESHOLD = 3.0
CRITICAL_ISSUES_THRESHOLD = 3

# Shared instances to reduce duplication
_agent_factory: AgentFactory | None = None
_settings: Settings | None = None
_context_monitor: ContextMonitor | None = None
_context_compactor: ContextCompactor | None = None
_artifact_manager: ArtifactManager | None = None


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


def _get_context_monitor() -> ContextMonitor:
    """Get shared context monitor instance."""
    global _context_monitor
    if _context_monitor is None:
        _context_monitor = ContextMonitor(max_tokens=128000)  # GPT-5 limit
    return _context_monitor


def _get_context_compactor() -> ContextCompactor:
    """Get shared context compactor instance."""
    global _context_compactor
    if _context_compactor is None:
        _context_compactor = ContextCompactor(max_tokens=100000, compression_threshold=0.85)  # Higher threshold for GPT-5
    return _context_compactor


def _get_artifact_manager() -> ArtifactManager:
    """Get shared artifact manager instance."""
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = ArtifactManager()
    return _artifact_manager


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
    monitor = _get_context_monitor()
    compactor = _get_context_compactor()

    # Monitor context before processing
    monitor.check_and_warn(state, "planner_start")

    # Compress state if needed before processing
    state = compactor.monitor_and_compress(state)

    if not state.original_prompt:
        state.original_prompt = state.prompt

    # Initialize workflow ID if not set
    if not hasattr(state, 'workflow_id') or not state.workflow_id:
        import uuid
        state.workflow_id = str(uuid.uuid4())

    planner = _get_agent_factory().create_planner()
    response = await planner.process(state)

    if response.success:
        state.subtasks = response.data
        await _save_checkpoint(state, "planner_completed")
    else:
        state.error_message = response.message
        state.should_continue = False

    # Monitor context after processing
    monitor.check_and_warn(state, "planner_end")
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


@with_retry(max_retries=2, delay=2.0)
@with_error_recovery("coding")
async def coding_node(state: WorkflowState) -> WorkflowState:
    """Execute coding agent."""
    monitor = _get_context_monitor()
    compactor = _get_context_compactor()

    # Monitor and compress context before coding (often the largest node)
    monitor.check_and_warn(state, "coding_start")
    state = compactor.monitor_and_compress(state)

    coding = _get_agent_factory().create_coding()
    response = await coding.process(state)

    if response.success:
        state.generated_code = response.data
        await _save_checkpoint(state, "coding_completed")
    else:
        state.error_message = response.message
        state.should_continue = False

    monitor.check_and_warn(state, "coding_end")
    return state


@with_retry(max_retries=2, delay=1.0)
async def execution_node(state: WorkflowState) -> WorkflowState:
    """Execute generated code using enhanced Blender executor."""
    log_workflow_state(state, "execution_start")
    executor = EnhancedBlenderExecutor()

    try:
        # Store previous screenshot for comparison if this is a refinement
        if hasattr(state, "execution_result") and state.execution_result:
            state.previous_screenshot_path = state.execution_result.screenshot_path

        result = await executor.execute_code(
            state.generated_code,
            asset_name=_generate_asset_name(),
            export_formats=["blend", "obj"],
            validate_code=False,  # Temporarily disable validation to test execution
            quality_settings={
                "render_engine": "BLENDER_EEVEE_NEXT",
                "render_samples": 64,
                "screenshot_width": 1024,
                "screenshot_height": 1024,
            },
        )

        state.execution_result = result

        # For testing: inject some errors to test retry mechanisms
        # This should be removed in production
        # Disabled for production runs - enable for testing retry mechanisms
        # if state.refinement_iterations == 0 and result.success:  # Only on first iteration and success
        #     test_errors = [
        #         "Test geometry error: Invalid mesh topology detected",
        #         "Test material error: Missing texture nodes"
        #     ]
        #     inject_execution_errors(state, test_errors)
        #     logger.info("Injected test errors for retry mechanism testing")

        if result.success:
            # Store assets using artifact manager
            artifact_manager = _get_artifact_manager()
            stored_artifacts = []
            
            # Store blend file if available
            if result.asset_path and Path(result.asset_path).exists():
                blend_artifact_id = await artifact_manager.store_artifact(
                    workflow_id=state.workflow_id,
                    artifact_type="blend_file",
                    source_path=result.asset_path,
                    metadata={"asset_name": _generate_asset_name(), "execution_step": "initial"}
                )
                stored_artifacts.append(blend_artifact_id)
            
            # Store screenshot if available
            if result.screenshot_path and Path(result.screenshot_path).exists():
                screenshot_artifact_id = await artifact_manager.store_artifact(
                    workflow_id=state.workflow_id,
                    artifact_type="screenshot",
                    source_path=result.screenshot_path,
                    metadata={"asset_name": _generate_asset_name(), "execution_step": "initial"}
                )
                stored_artifacts.append(screenshot_artifact_id)
            
            # Store generated code as artifact
            if state.generated_code:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(state.generated_code)
                    code_file_path = f.name
                
                code_artifact_id = await artifact_manager.store_artifact(
                    workflow_id=state.workflow_id,
                    artifact_type="code",
                    source_path=code_file_path,
                    metadata={"execution_step": "initial", "code_type": "blender_script"}
                )
                stored_artifacts.append(code_artifact_id)
                
                # Clean up temporary file
                Path(code_file_path).unlink(missing_ok=True)
            
            state.asset_metadata = AssetMetadata(
                id=_generate_asset_id(),
                prompt=state.original_prompt or state.prompt,
                file_path=result.asset_path or "",
                screenshot_path=result.screenshot_path,
                subtasks=state.subtasks,
                quality_score=None,
            )
            
            # Add artifact references to state
            if not hasattr(state, 'stored_artifacts'):
                state.stored_artifacts = []
            state.stored_artifacts.extend(stored_artifacts)
            
            log_workflow_state(state, "execution_completed")
            await _save_checkpoint(state, "execution_completed")
        else:
            state.error_message = f"Execution failed: {'; '.join(result.errors)}"
            log_workflow_state(state, "execution_failed")

    except Exception as e:
        state.error_message = f"Execution error: {e!s}"
        state.should_continue = False
        log_workflow_state(state, "execution_exception")

    return state


# Quality assessment nodes (only used in full workflow)
async def critic_node(state: WorkflowState) -> WorkflowState:
    """Execute critic agent for visual analysis."""
    log_workflow_state(state, "critic_node_start")
    critic = _get_agent_factory().create_critic()
    response = await critic.process(state)

    if response.success:
        state.critic_analysis = response.data
        state.needs_refinement = response.metadata.get("needs_refinement", False)

        # Convert refinement priority from string to float
        priority_str = response.metadata.get("refinement_priority", "low")
        priority_map = {"high": 1.0, "medium": 0.7, "low": 0.3}
        state.refinement_priority = priority_map.get(priority_str, 0.3)

        await _save_checkpoint(state, "critic_completed")
        log_workflow_state(state, "critic_completed")
    else:
        state.error_message = f"Critic analysis failed: {response.message}"
        state.should_continue = False
        log_workflow_state(state, "critic_failed")

    return state


async def verification_node(state: WorkflowState) -> WorkflowState:
    """Execute verification agent for quality assessment."""
    log_workflow_state(state, "verification_node_start")
    verification = _get_agent_factory().create_verification()
    response = await verification.process(state)

    if response.success:
        state.verification_result = response.data

        # Combine verification and critic results for refinement decision
        verification_score = response.data.get("quality_score", 10.0)
        has_critical_issues = response.metadata.get("critical_issues", 0) > 0

        # Store verification data for quality_assessment_node to make final decision
        # Don't override needs_refinement here - let quality_assessment_node decide
        state.verification_score = verification_score
        state.has_critical_issues = has_critical_issues

        await _save_checkpoint(state, "verification_completed")
        log_workflow_state(state, "verification_completed")
    else:
        state.error_message = f"Verification failed: {response.message}"
        state.should_continue = False
        log_workflow_state(state, "verification_failed")

    return state


async def quality_assessment_node(state: WorkflowState) -> WorkflowState:
    """Combined quality assessment using both critic and verification results."""
    settings = _get_settings()
    log_workflow_state(state, "quality_assessment_start")
    logger.info(f"Quality assessment: iteration={state.refinement_iterations}, max={settings.app.max_refinement_iterations}")

    critic_analysis = getattr(state, "critic_analysis", {})
    verification_result = getattr(state, "verification_result", {})

    # Generate refinement recommendations
    refinement_suggestions = []

    if critic_analysis:
        critic_suggestions = critic_analysis.get("improvement_suggestions", [])
        refinement_suggestions.extend(critic_suggestions)
        # Check if critic suggests continuing refinement
        continue_refinement = critic_analysis.get("continue_refinement", False)
        logger.info(f"Critic continue_refinement: {continue_refinement}")

    if verification_result:
        verification_suggestions = verification_result.get("recommendations", [])
        refinement_suggestions.extend(verification_suggestions)

    # Only set needs_refinement if we haven't reached max iterations and quality is poor
    should_refine = False

    # Check if we've already hit max refinements
    if state.refinement_iterations >= settings.app.max_refinement_iterations:
        logger.info(f"Max refinements reached ({state.refinement_iterations}/{settings.app.max_refinement_iterations}), not requesting more refinement")
        should_refine = False
    # Hard stop after 2 refinements for successful executions (unless critical issues)
    elif (state.refinement_iterations >= 2 and 
          state.execution_result and 
          state.execution_result.success and
          not getattr(state, "has_critical_issues", False)):
        logger.info(f"Hard stop after {state.refinement_iterations} refinements with successful execution")
        should_refine = False
    elif refinement_suggestions:
        # Use stored verification data for decision
        quality_score = getattr(state, "verification_score", 10.0)
        has_critical_issues = getattr(state, "has_critical_issues", False)
        critical_issues = len(verification_result.get("issues_found", [])) if verification_result else 0

        # Only refine for truly poor quality or critical issues
        if quality_score < QUALITY_THRESHOLD or critical_issues > CRITICAL_ISSUES_THRESHOLD or has_critical_issues:
            should_refine = True
            logger.info(f"Quality needs improvement: score={quality_score}, issues={critical_issues}, critical={has_critical_issues}")
        else:
            should_refine = False
            logger.info(f"Quality acceptable: score={quality_score}, issues={critical_issues}, critical={has_critical_issues}")
    else:
        # No refinement suggestions means quality is acceptable
        should_refine = False
        logger.info("No refinement suggestions - quality is acceptable")

    state.needs_refinement = should_refine

    # Set refinement request based on combined analysis
    if refinement_suggestions and should_refine:
        state.refinement_request = (
            f"Quality assessment identified the following improvements needed: "
            f"{'; '.join(refinement_suggestions[:3])}"  # Limit to top 3 suggestions
        )
    else:
        state.refinement_request = ""

    await _save_checkpoint(state, "quality_assessment_completed")
    log_workflow_state(state, "quality_assessment_completed")
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
            # Check if we can still refine (respect both limits)
            settings = _get_settings()
            max_reached = state.refinement_iterations >= settings.app.max_refinement_iterations
            hard_stop = (state.refinement_iterations >= 2 and 
                        state.execution_result and 
                        state.execution_result.success and
                        not getattr(state, "has_critical_issues", False))
            
            if not max_reached and not hard_stop:
                # Extract refinement instructions
                workflow_action.get("instructions", [])
                refinement_summary = workflow_action.get("reason", "User requested refinements")

                # Update refinement request with structured feedback
                state.refinement_request = refinement_summary
                state.needs_refinement = True
                state.refinement_priority = 1.0 if workflow_action.get("priority") == "high" else 0.5
            else:
                # Max refinements reached, complete instead
                logger.info(f"User requested refinement but limits reached (iter={state.refinement_iterations}, max={max_reached}, hard_stop={hard_stop})")
                state.should_continue = False
                state.user_accepted = True

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
    logger.info(
        f"Refinement decision check: iterations={state.refinement_iterations}, "
        f"needs_refinement={state.needs_refinement}, error_message={bool(state.error_message)}"
    )

    if state.error_message:
        logger.info("Ending due to error message")
        return "end"

    # Check if refinement is needed and under iteration limit
    settings = _get_settings()
    max_iterations = settings.app.max_refinement_iterations

    # Hard stop after max iterations or 2 iterations for successful execution (unless critical issues)
    if state.refinement_iterations >= max_iterations or (
        state.refinement_iterations >= 2 and 
        state.execution_result and 
        state.execution_result.success and
        not getattr(state, "has_critical_issues", False)
    ):
        logger.info(f"Max refinement iterations reached: {state.refinement_iterations}/{max_iterations}")
        return "complete"

    # If we have execution errors, stop refining
    if (state.execution_result and
        state.execution_result.errors and
        len(state.execution_result.errors) > 0):
        logger.info(f"Stopping refinement due to execution errors: {state.execution_result.errors}")
        return "complete"

    # Only refine if explicitly needed and we haven't exceeded iterations
    if state.needs_refinement:
        # Consider refinement priority
        priority = getattr(state, "refinement_priority", 0.0)
        if priority > 0.0 or state.refinement_iterations == 0:
            logger.info(f"Continuing refinement: iteration {state.refinement_iterations + 1}/{max_iterations}")
            return "refine"
        else:
            logger.info("Skipping refinement due to low priority")
            return "complete"

    logger.info("No refinement needed, completing")
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

    # Use artifact manager-compatible memory saver with persistent storage
    memory = MemorySaver()
    compiled_workflow = workflow.compile(checkpointer=memory)
    
    # Add workflow metadata for tracking
    if hasattr(compiled_workflow, 'workflow_metadata'):
        compiled_workflow.workflow_metadata = {
            'supports_interruption': True,
            'supports_resumption': True,
            'artifact_management': True,
            'checkpoint_steps': [
                'planner_completed', 'retrieval_completed', 'coding_completed',
                'execution_completed', 'critic_completed', 'verification_completed',
                'quality_assessment_completed', 'user_proxy_completed'
            ]
        }
    
    return compiled_workflow


# Enhanced checkpoint management with artifact integration
async def _save_checkpoint(state: WorkflowState, checkpoint_name: str) -> None:
    """Save workflow state checkpoint using artifact manager."""
    try:
        # Ensure workflow has an ID
        if not hasattr(state, 'workflow_id') or not state.workflow_id:
            import uuid
            state.workflow_id = str(uuid.uuid4())
        
        artifact_manager = _get_artifact_manager()
        
        # Create execution context for the checkpoint
        execution_context = {
            "checkpoint_name": checkpoint_name,
            "timestamp": time.time(),
            "healing_iterations": getattr(state, 'healing_iterations', 0),
            "total_tokens_used": getattr(state, 'total_tokens_used', 0),
            "performance_metrics": getattr(state, 'performance_metrics', {}),
            "refinement_iterations": getattr(state, 'refinement_iterations', 0)
        }
        
        # Save checkpoint using artifact manager
        checkpoint_id = await artifact_manager.create_checkpoint(
            workflow_id=state.workflow_id,
            state=state,
            step_name=checkpoint_name,
            execution_context=execution_context
        )
        
        # Store checkpoint ID in state for reference
        if not hasattr(state, 'checkpoint_history'):
            state.checkpoint_history = []
        state.checkpoint_history.append({
            "checkpoint_id": checkpoint_id,
            "step_name": checkpoint_name,
            "timestamp": time.time()
        })
        
        logger.info(f"Checkpoint saved: {checkpoint_name} (ID: {checkpoint_id})")
        
    except Exception as e:
        # Don't fail workflow on checkpoint save error
        logger.warning(f"Failed to save checkpoint {checkpoint_name}: {e}")


async def _load_checkpoint(checkpoint_id: str) -> WorkflowState:
    """Load workflow state from checkpoint ID using artifact manager."""
    try:
        artifact_manager = _get_artifact_manager()
        checkpoint = await artifact_manager.restore_from_checkpoint(checkpoint_id)
        
        logger.info(f"Restored workflow from checkpoint {checkpoint_id} at step {checkpoint.step_name}")
        return checkpoint.state
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
        raise


async def resume_workflow_from_latest(workflow_id: str) -> tuple[WorkflowState, str] | None:
    """Resume workflow from its latest checkpoint."""
    try:
        artifact_manager = _get_artifact_manager()
        latest_checkpoint = await artifact_manager.get_latest_checkpoint(workflow_id)
        
        if not latest_checkpoint:
            logger.warning(f"No checkpoints found for workflow {workflow_id}")
            return None
            
        logger.info(f"Resuming workflow {workflow_id} from step {latest_checkpoint.step_name}")
        return latest_checkpoint.state, latest_checkpoint.step_name
        
    except Exception as e:
        logger.error(f"Failed to resume workflow {workflow_id}: {e}")
        return None


async def get_workflow_progress(workflow_id: str) -> dict[str, Any] | None:
    """Get detailed progress information for a workflow."""
    try:
        artifact_manager = _get_artifact_manager()
        summary = await artifact_manager.get_workflow_summary(workflow_id)
        
        # Get latest checkpoint info
        latest_checkpoint = await artifact_manager.get_latest_checkpoint(workflow_id)
        if latest_checkpoint:
            summary['current_step'] = latest_checkpoint.step_name
            summary['last_checkpoint'] = latest_checkpoint.created_at.isoformat()
            summary['completion_percentage'] = _estimate_completion_percentage(latest_checkpoint.step_name)
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get workflow progress for {workflow_id}: {e}")
        return None


def _estimate_completion_percentage(step_name: str) -> float:
    """Estimate workflow completion percentage based on current step."""
    step_progress = {
        "planner_completed": 10.0,
        "retrieval_completed": 20.0,
        "coding_completed": 40.0,
        "execution_completed": 60.0,
        "critic_completed": 70.0,
        "verification_completed": 80.0,
        "quality_assessment_completed": 90.0,
        "user_proxy_completed": 95.0,
        "workflow_completed": 100.0
    }
    return step_progress.get(step_name, 0.0)


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
