"""Refinement components for workflow improvement."""

from typing import Callable, Literal
import structlog
from ...utils.types import WorkflowState
from .workflow_builder import WorkflowConfig, RefinementMode

logger = structlog.get_logger(__name__)


class RefinementStrategy:
    """Individual refinement strategy."""
    
    def __init__(self, name: str, condition_func: Callable, refinement_func: Callable):
        self.name = name
        self.condition_func = condition_func
        self.refinement_func = refinement_func
        
    async def should_refine(self, state: WorkflowState) -> bool:
        """Check if this strategy should trigger refinement."""
        try:
            return await self.condition_func(state)
        except Exception as e:
            logger.error(f"Refinement condition check failed for {self.name}: {e}")
            return False
            
    async def refine(self, state: WorkflowState) -> WorkflowState:
        """Apply refinement."""
        try:
            logger.info(f"Applying refinement strategy: {self.name}")
            return await self.refinement_func(state)
        except Exception as e:
            logger.error(f"Refinement strategy {self.name} failed: {e}")
            state.error_message = f"Refinement failed: {e}"
            return state


class RefinementEngine:
    """Configurable refinement engine."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        
    def create_limited_refinement(self) -> Callable:
        """Create limited refinement (up to max_refinements)."""
        async def limited_refinement(state: WorkflowState) -> WorkflowState:
            """Apply limited refinement strategy."""
            logger.info("Running limited refinement")
            
            # Check basic refinement conditions
            if not self._should_attempt_refinement(state):
                return state
                
            try:
                # Run refinement agent
                from ...agents.refinement import create_refinement_workflow
                refinement_node = create_refinement_workflow()
                state = await refinement_node(state)
                
                # Increment refinement counter
                state.refinement_iterations = getattr(state, 'refinement_iterations', 0) + 1
                
                return state
                
            except Exception as e:
                logger.error(f"Limited refinement failed: {e}")
                state.error_message = f"Refinement error: {e}"
                return state
                
        return limited_refinement
        
    def create_adaptive_refinement(self) -> Callable:
        """Create adaptive refinement based on quality analysis."""
        async def adaptive_refinement(state: WorkflowState) -> WorkflowState:
            """Apply adaptive refinement based on quality issues."""
            logger.info("Running adaptive refinement")
            
            if not self._should_attempt_refinement(state):
                return state
                
            try:
                # Select appropriate refinement strategy based on issues
                strategies = [
                    RefinementStrategy(
                        "error_focused",
                        self._has_execution_errors,
                        self._create_error_focused_refinement()
                    ),
                    RefinementStrategy(
                        "quality_focused", 
                        self._has_quality_issues,
                        self._create_quality_focused_refinement()
                    ),
                    RefinementStrategy(
                        "output_focused",
                        self._has_output_issues,
                        self._create_output_focused_refinement()
                    ),
                    RefinementStrategy(
                        "general",
                        lambda s: True,  # Fallback strategy
                        self._create_general_refinement()
                    )
                ]
                
                # Apply first matching strategy
                for strategy in strategies:
                    if await strategy.should_refine(state):
                        state = await strategy.refine(state)
                        break
                        
                # Increment refinement counter
                state.refinement_iterations = getattr(state, 'refinement_iterations', 0) + 1
                
                return state
                
            except Exception as e:
                logger.error(f"Adaptive refinement failed: {e}")
                state.error_message = f"Adaptive refinement error: {e}"
                return state
                
        return adaptive_refinement
    
    def _should_attempt_refinement(self, state: WorkflowState) -> bool:
        """Check if refinement should be attempted."""
        # Check iteration limits
        if state.max_iterations and state.iterations >= state.max_iterations:
            logger.info("Max iterations reached, skipping refinement")
            return False
            
        refinement_count = getattr(state, 'refinement_iterations', 0)
        if refinement_count >= self.config.max_refinements:
            logger.info("Max refinements reached, skipping refinement")
            return False
            
        # Check if there's something to refine
        has_errors = bool(state.error_message)
        has_quality_issues = hasattr(state, 'quality_score') and state.quality_score < self.config.quality_threshold
        has_critical_issues = (hasattr(state, 'quality_issues') and 
                             len(state.quality_issues) > self.config.critical_issues_threshold)
        
        should_refine = has_errors or has_quality_issues or has_critical_issues
        
        if not should_refine:
            logger.info("No refinement needed - quality sufficient")
            
        return should_refine
        
    async def _has_execution_errors(self, state: WorkflowState) -> bool:
        """Check if state has execution errors."""
        return bool(state.error_message or 
                   (state.execution_result and not state.execution_result.success))
                   
    async def _has_quality_issues(self, state: WorkflowState) -> bool:
        """Check if state has quality issues."""
        return (hasattr(state, 'quality_score') and 
                state.quality_score < self.config.quality_threshold)
                
    async def _has_output_issues(self, state: WorkflowState) -> bool:
        """Check if state has output issues."""
        return not (state.blend_file_path and state.screenshot_path)
    
    def _create_error_focused_refinement(self) -> Callable:
        """Create error-focused refinement function."""
        async def error_focused_refinement(state: WorkflowState) -> WorkflowState:
            """Refinement focused on fixing execution errors."""
            logger.info("Applying error-focused refinement")
            
            # Enhance refinement prompt with error context
            if hasattr(state, 'refinement_prompt'):
                error_context = f"Focus on fixing these errors: {state.error_message}"
                if state.execution_result and state.execution_result.errors:
                    error_context += f"\nExecution errors: {state.execution_result.errors}"
                state.refinement_prompt += f"\n\n{error_context}"
            
            # Apply standard refinement
            from ...agents.refinement import create_refinement_workflow
            refinement_node = create_refinement_workflow()
            return await refinement_node(state)
            
        return error_focused_refinement
        
    def _create_quality_focused_refinement(self) -> Callable:
        """Create quality-focused refinement function."""
        async def quality_focused_refinement(state: WorkflowState) -> WorkflowState:
            """Refinement focused on improving code quality."""
            logger.info("Applying quality-focused refinement")
            
            # Enhance refinement prompt with quality context
            if hasattr(state, 'refinement_prompt'):
                quality_context = f"Current quality score: {getattr(state, 'quality_score', 0)}"
                if hasattr(state, 'quality_issues'):
                    quality_context += f"\nQuality issues: {state.quality_issues}"
                if hasattr(state, 'quality_feedback'):
                    quality_context += f"\nQuality feedback: {state.quality_feedback}"
                state.refinement_prompt += f"\n\n{quality_context}"
            
            # Apply standard refinement
            from ...agents.refinement import create_refinement_workflow
            refinement_node = create_refinement_workflow()
            return await refinement_node(state)
            
        return quality_focused_refinement
        
    def _create_output_focused_refinement(self) -> Callable:
        """Create output-focused refinement function."""
        async def output_focused_refinement(state: WorkflowState) -> WorkflowState:
            """Refinement focused on generating required outputs."""
            logger.info("Applying output-focused refinement")
            
            # Enhance refinement prompt with output requirements
            if hasattr(state, 'refinement_prompt'):
                missing_outputs = []
                if not state.blend_file_path:
                    missing_outputs.append("blend file")
                if not state.screenshot_path:
                    missing_outputs.append("screenshot")
                    
                if missing_outputs:
                    output_context = f"Ensure these outputs are generated: {', '.join(missing_outputs)}"
                    state.refinement_prompt += f"\n\n{output_context}"
            
            # Apply standard refinement
            from ...agents.refinement import create_refinement_workflow  
            refinement_node = create_refinement_workflow()
            return await refinement_node(state)
            
        return output_focused_refinement
        
    def _create_general_refinement(self) -> Callable:
        """Create general refinement function."""
        async def general_refinement(state: WorkflowState) -> WorkflowState:
            """General refinement strategy."""
            logger.info("Applying general refinement")
            
            from ...agents.refinement import create_refinement_workflow
            refinement_node = create_refinement_workflow()
            return await refinement_node(state)
            
        return general_refinement


def create_simple_refinement_check(config: WorkflowConfig) -> Callable:
    """Create simple refinement check for fast workflows."""
    def simple_refinement_check(state: WorkflowState) -> Literal["refine", "end"]:
        """Simple check - refine if there are errors and we haven't hit limits."""
        if not state.error_message:
            return "end"
            
        refinement_count = getattr(state, 'refinement_iterations', 0)
        if refinement_count >= config.max_refinements:
            return "end"
            
        if state.max_iterations and state.iterations >= state.max_iterations:
            return "end"
            
        return "refine"
        
    return simple_refinement_check


def should_refine_enhanced(state: WorkflowState) -> Literal["refine", "end"]:
    """Enhanced refinement decision with comprehensive stop conditions."""
    logger.debug("Evaluating refinement decision")
    
    # Hard stop conditions
    if state.error_message and "max_iterations" in state.error_message.lower():
        logger.info("Stopping: Max iterations reached")
        return "end"
        
    if state.max_iterations and state.iterations >= state.max_iterations:
        logger.info("Stopping: Iteration limit reached")
        return "end"
        
    refinement_count = getattr(state, 'refinement_iterations', 0)
    if refinement_count >= 2:  # Hard limit
        logger.info("Stopping: Max refinements reached")
        return "end"
        
    # Success conditions  
    if (state.execution_result and 
        state.execution_result.success and 
        not state.execution_result.errors and
        refinement_count >= 1):
        logger.info("Stopping: Successful execution after refinement")
        return "end"
        
    # No refinement needed conditions
    if not state.error_message and refinement_count > 0:
        logger.info("Stopping: No errors after refinement")
        return "end"
        
    # Quality-based refinement decision
    if hasattr(state, 'quality_score') and state.quality_score >= 2.0:
        logger.info("Stopping: Quality threshold met")
        return "end"
        
    # Refine if we have errors and haven't hit limits
    if state.error_message or (hasattr(state, 'quality_score') and state.quality_score < 2.0):
        logger.info("Continuing: Refinement needed")
        return "refine"
        
    logger.info("Stopping: Default - no refinement needed")
    return "end"