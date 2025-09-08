"""Enhanced planning nodes with optional MCTS integration for LL3M workflow."""

import asyncio
from typing import Literal

import structlog

from ..agents import get_agent_factory
from ..utils.config import get_settings
from ..utils.types import WorkflowState

logger = structlog.get_logger(__name__)


async def adaptive_planner_node(state: WorkflowState) -> WorkflowState:
    """Enhanced planner node that adaptively chooses between standard and MCTS planning."""
    if not state.original_prompt:
        state.original_prompt = state.prompt
    
    settings = get_settings()
    factory = get_agent_factory()
    
    # Determine planning strategy based on task complexity
    planning_strategy = _determine_planning_strategy(state)
    
    if planning_strategy == "mcts":
        logger.info("Using MCTS planning for complex task", prompt=state.prompt[:100])
        planner = factory.create_mcts()
    else:
        logger.info("Using standard planning for simple task", prompt=state.prompt[:100])
        planner = factory.create_planner()
    
    response = await planner.process(state)
    
    if response.success:
        state.subtasks = response.data
        # Store planning metadata for analysis
        if hasattr(response, 'metadata') and response.metadata:
            if not hasattr(state, 'planning_metadata'):
                state.planning_metadata = {}
            state.planning_metadata.update({
                'strategy_used': planning_strategy,
                'agent_type': str(response.agent_type),
                **response.metadata
            })
        await _save_checkpoint(state, "adaptive_planning_completed")
    else:
        state.error_message = response.message
        state.should_continue = False
        logger.error("Planning failed", error=response.message, strategy=planning_strategy)
    
    return state


async def hybrid_planner_node(state: WorkflowState) -> WorkflowState:
    """Hybrid planner that uses both standard and MCTS planning, then selects the best result."""
    if not state.original_prompt:
        state.original_prompt = state.prompt
    
    factory = get_agent_factory()
    
    # Run both planners concurrently
    logger.info("Running hybrid planning (standard + MCTS)", prompt=state.prompt[:100])
    
    standard_planner = factory.create_planner()
    mcts_planner = factory.create_mcts()
    
    try:
        # Execute both planners concurrently
        standard_task = asyncio.create_task(standard_planner.process(state))
        mcts_task = asyncio.create_task(mcts_planner.process(state))
        
        standard_response, mcts_response = await asyncio.gather(
            standard_task, mcts_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(standard_response, Exception):
            logger.warning("Standard planner failed", error=str(standard_response))
            standard_response = None
            
        if isinstance(mcts_response, Exception):
            logger.warning("MCTS planner failed", error=str(mcts_response))
            mcts_response = None
        
        # Select the best response
        selected_response = _select_best_planning_response(
            standard_response, mcts_response, state
        )
        
        if selected_response and selected_response.success:
            state.subtasks = selected_response.data
            # Store hybrid planning metadata
            if not hasattr(state, 'planning_metadata'):
                state.planning_metadata = {}
            state.planning_metadata.update({
                'strategy_used': 'hybrid',
                'selected_planner': str(selected_response.agent_type),
                'standard_success': standard_response.success if standard_response else False,
                'mcts_success': mcts_response.success if mcts_response else False,
                **(selected_response.metadata if hasattr(selected_response, 'metadata') and selected_response.metadata else {})
            })
            await _save_checkpoint(state, "hybrid_planning_completed")
        else:
            state.error_message = "Both planning strategies failed"
            state.should_continue = False
            logger.error("All planning strategies failed")
            
    except Exception as e:
        state.error_message = f"Hybrid planning failed: {str(e)}"
        state.should_continue = False
        logger.error("Hybrid planning exception", error=str(e))
    
    return state


async def mcts_enhanced_refinement_node(state: WorkflowState) -> WorkflowState:
    """Refinement planning node that uses MCTS for sophisticated refinement strategies."""
    factory = get_agent_factory()
    
    # Check if refinement is needed
    if not state.needs_refinement or not state.refinement_request:
        return state
    
    logger.info("Using MCTS for refinement planning", 
                refinement_request=state.refinement_request[:100])
    
    # Create a modified state for MCTS refinement planning
    refinement_state = WorkflowState(
        prompt=f"Refine the current asset: {state.refinement_request}",
        original_prompt=state.original_prompt,
        user_prompt=state.user_prompt,
        user_feedback=state.user_feedback,
        documentation=state.documentation,
        generated_code=state.generated_code,
        execution_result=state.execution_result,
        refinement_request=state.refinement_request,
        asset_metadata=state.asset_metadata,
        error_message=state.error_message,
        previous_screenshot_path=state.previous_screenshot_path,
        critic_analysis=state.critic_analysis,
        verification_result=state.verification_result,
    )
    
    mcts_planner = factory.create_mcts()
    response = await mcts_planner.process(refinement_state)
    
    if response.success:
        # Merge refinement subtasks with existing ones
        refinement_subtasks = response.data
        state.subtasks.extend(refinement_subtasks)
        
        # Update refinement metadata
        if not hasattr(state, 'refinement_metadata'):
            state.refinement_metadata = {}
        state.refinement_metadata.update({
            'mcts_refinement_subtasks': len(refinement_subtasks),
            'refinement_strategy': 'mcts_enhanced',
            **(response.metadata if hasattr(response, 'metadata') and response.metadata else {})
        })
        
        await _save_checkpoint(state, "mcts_refinement_completed")
        logger.info("MCTS refinement planning completed", 
                   new_subtasks=len(refinement_subtasks))
    else:
        logger.warning("MCTS refinement planning failed", error=response.message)
        # Continue without MCTS refinement
        
    return state


def _determine_planning_strategy(state: WorkflowState) -> Literal["standard", "mcts"]:
    """Determine whether to use standard or MCTS planning based on task characteristics."""
    prompt = state.prompt.lower()
    
    # Factors that suggest MCTS would be beneficial
    mcts_indicators = [
        # Complexity indicators
        len(prompt.split()) > 20,  # Long, complex prompts
        "complex" in prompt or "sophisticated" in prompt or "advanced" in prompt,
        "multiple" in prompt or "several" in prompt,
        
        # Quality indicators
        "realistic" in prompt or "photorealistic" in prompt,
        "professional" in prompt or "high-quality" in prompt,
        "detailed" in prompt or "intricate" in prompt,
        
        # Scene complexity
        "scene" in prompt and ("lighting" in prompt or "environment" in prompt),
        "animation" in prompt or "rigging" in prompt,
        "procedural" in prompt or "parametric" in prompt,
        
        # Multi-object scenarios
        " and " in prompt and prompt.count(" and ") >= 2,
        "with" in prompt and prompt.count("with") >= 2,
    ]
    
    # Use MCTS if multiple indicators are present
    mcts_score = sum(mcts_indicators)
    
    if mcts_score >= 3:
        return "mcts"
    else:
        return "standard"


def _select_best_planning_response(standard_response, mcts_response, state: WorkflowState):
    """Select the best planning response from standard and MCTS planners."""
    # If only one succeeded, use that one
    if standard_response and standard_response.success and not (mcts_response and mcts_response.success):
        return standard_response
    elif mcts_response and mcts_response.success and not (standard_response and standard_response.success):
        return mcts_response
    elif not (standard_response and standard_response.success) and not (mcts_response and mcts_response.success):
        return None
    
    # Both succeeded - select based on quality heuristics
    standard_subtasks = standard_response.data if standard_response else []
    mcts_subtasks = mcts_response.data if mcts_response else []
    
    # Scoring criteria
    scores = {
        'standard': _score_planning_quality(standard_subtasks, state),
        'mcts': _score_planning_quality(mcts_subtasks, state)
    }
    
    # MCTS gets bonus for metadata richness
    if mcts_response and hasattr(mcts_response, 'metadata') and mcts_response.metadata:
        scores['mcts'] += 0.1
    
    # Select the higher scoring response
    if scores['mcts'] > scores['standard']:
        return mcts_response
    else:
        return standard_response


def _score_planning_quality(subtasks, state: WorkflowState) -> float:
    """Score the quality of a planning result."""
    if not subtasks:
        return 0.0
    
    score = 0.0
    
    # Task count (moderate number is better)
    task_count = len(subtasks)
    if 3 <= task_count <= 8:
        score += 0.3
    elif task_count > 8:
        score += 0.2  # Many tasks can be good but may be over-complex
    else:
        score += 0.1  # Too few tasks might be under-planned
    
    # Task type diversity
    task_types = {task.type for task in subtasks}
    diversity_score = len(task_types) / 5.0  # Normalize by max expected types
    score += diversity_score * 0.2
    
    # Priority distribution (should have varied priorities)
    priorities = [task.priority for task in subtasks]
    if len(set(priorities)) > 1:
        score += 0.2
    
    # Dependency complexity (some dependencies indicate good planning)
    total_deps = sum(len(task.dependencies) for task in subtasks)
    if total_deps > 0:
        score += min(0.2, total_deps / len(subtasks) * 0.1)
    
    # Description quality (longer, more detailed descriptions)
    avg_desc_length = sum(len(task.description.split()) for task in subtasks) / len(subtasks)
    if avg_desc_length > 8:
        score += 0.1
    
    return min(1.0, score)  # Cap at 1.0


async def _save_checkpoint(state: WorkflowState, checkpoint_name: str):
    """Save workflow checkpoint (placeholder for actual implementation)."""
    # This would integrate with the actual checkpoint saving system
    logger.debug("Checkpoint saved", name=checkpoint_name)