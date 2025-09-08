"""Core workflow components for the LL3M system."""

from collections.abc import Callable

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ...utils.types import WorkflowState
from .workflow_builder import WorkflowConfig

logger = structlog.get_logger(__name__)


class ExecutionPipeline:
    """Core execution pipeline for the main workflow steps."""

    def __init__(self, config: WorkflowConfig):
        self.config = config

    def create_planner_node(self) -> Callable:
        """Create the planner node."""
        from ...agents.factory import get_agent_factory
        
        async def planner_node(state: WorkflowState) -> WorkflowState:
            """Execute planner agent."""
            planner = get_agent_factory().create_planner()
            response = await planner.process(state)
            if response.success:
                state.subtasks = response.data
            else:
                state.error_message = response.message or "Planner execution failed"
            return state
            
        return planner_node

    def create_retrieval_node(self) -> Callable:
        """Create the retrieval node."""
        from ...agents.factory import get_agent_factory
        
        async def retrieval_node(state: WorkflowState) -> WorkflowState:
            """Execute retrieval agent."""
            retrieval = get_agent_factory().create_retrieval()
            response = await retrieval.process(state)
            if response.success:
                state.documentation = response.data
            else:
                state.error_message = response.message or "Retrieval execution failed"
            return state
            
        return retrieval_node

    def create_coding_node(self) -> Callable:
        """Create the coding node."""
        from ...agents.factory import get_agent_factory
        
        async def coding_node(state: WorkflowState) -> WorkflowState:
            """Execute coding agent."""
            coding = get_agent_factory().create_coding()
            response = await coding.process(state)
            if response.success:
                state.generated_code = response.data
            else:
                state.error_message = response.message or "Coding execution failed"
            return state
            
        return coding_node

    def create_execution_node(self) -> Callable:
        """Create the execution node."""
        from ...blender.enhanced_executor import EnhancedBlenderExecutor
        
        async def execution_node(state: WorkflowState) -> WorkflowState:
            """Execute Blender code."""
            executor = EnhancedBlenderExecutor()
            result = await executor.execute_code(
                code=state.generated_code,
                prompt=state.prompt,
                output_dir=state.output_directory or "outputs"
            )
            state.execution_result = result
            if result.asset_path:
                state.blend_file_path = result.asset_path
            if result.screenshot_path:
                state.screenshot_path = result.screenshot_path
            if not result.success:
                error_msg = "; ".join(result.errors) if result.errors else "Execution failed"
                state.error_message = error_msg
            return state
            
        return execution_node

    def create_should_continue_condition(self) -> Callable:
        """Create the should_continue_main condition."""
        def should_continue_main(state: WorkflowState) -> str:
            """Determine if main workflow should continue."""
            if state.error_message:
                return "end"
            # Check if refinement limit reached (WorkflowState uses refinement_iterations)
            if state.refinement_iterations >= state.max_refinements:
                return "end"
            return "continue"
        return should_continue_main


class CoreWorkflowBuilder:
    """Core workflow builder that handles the basic execution pipeline."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.graph = StateGraph(WorkflowState)
        self.pipeline = ExecutionPipeline(config)

    def add_execution_pipeline(self):
        """Add the core execution pipeline (planner -> retrieval -> coding -> execution)."""
        logger.info("Adding core execution pipeline")

        # Add core nodes
        self.graph.add_node("planner", self.pipeline.create_planner_node())
        self.graph.add_node("retrieval", self.pipeline.create_retrieval_node())
        self.graph.add_node("coding", self.pipeline.create_coding_node())
        self.graph.add_node("execution", self.pipeline.create_execution_node())

        # Set entry point
        self.graph.set_entry_point("planner")

        # Add conditional edges for main flow
        should_continue = self.pipeline.create_should_continue_condition()

        self.graph.add_conditional_edges(
            "planner",
            should_continue,
            {"continue": "retrieval", "end": END}
        )
        self.graph.add_conditional_edges(
            "retrieval",
            should_continue,
            {"continue": "coding", "end": END}
        )
        self.graph.add_conditional_edges(
            "coding",
            should_continue,
            {"continue": "execution", "end": END}
        )

    def add_quick_quality_gate(self):
        """Add a quick quality assessment after execution."""
        from .quality_components import create_quick_quality_check

        self.graph.add_node("quick_quality", create_quick_quality_check(self.config))
        from .quality_components import should_assess_quality
        
        self.graph.add_conditional_edges(
            "execution",
            should_assess_quality,
            {"assess": "quick_quality", "end": END}
        )
        self.graph.add_edge("quick_quality", END)

    def add_simple_refinement(self):
        """Add simple refinement loop back to coding."""
        from .refinement_components import create_simple_refinement_check

        refinement_check = create_simple_refinement_check(self.config)
        
        # Add refinement node
        from .refinement_components import create_simple_refinement_node
        refinement_node = create_simple_refinement_node(self.config)
        self.graph.add_node("simple_refinement", refinement_node)
        
        # Execution -> refinement check -> refinement/end
        self.graph.add_conditional_edges(
            "execution",
            refinement_check,
            {"refine": "simple_refinement", "end": END}
        )
        
        # Refinement -> coding (loop back)
        self.graph.add_edge("simple_refinement", "coding")

    def add_quality_gate(self, quality_assessment: Callable):
        """Add a quality assessment pipeline after execution."""
        self.graph.add_node("quality_assessment", quality_assessment)
        self.graph.add_conditional_edges(
            "execution",
            self.pipeline.create_should_continue_condition(),
            {"continue": "quality_assessment", "end": END}
        )
        self.graph.add_edge("quality_assessment", END)

    def add_refinement(self, refinement_logic: Callable):
        """Add refinement logic with loop back to coding."""
        from .refinement_components import should_refine_enhanced
        
        self.graph.add_node("refinement", refinement_logic)
        
        # Quality assessment -> refinement decision
        self.graph.add_conditional_edges(
            "quality_assessment",
            should_refine_enhanced,
            {"refine": "refinement", "end": END}
        )
        
        # Refinement -> coding (loop back to target point)
        self.graph.add_edge("refinement", "coding")

    def create_refinement_condition(self) -> Callable:
        """Create refinement condition function."""
        def should_refine_condition(state: WorkflowState) -> str:
            """Enhanced refinement decision with configuration awareness."""
            # WorkflowState doesn't have max_iterations field
            if getattr(state, 'refinement_iterations', 0) >= self.config.max_refinements:
                return "end"
                
            has_errors = bool(state.error_message)
            has_quality_issues = (hasattr(state, 'quality_score') and 
                                state.quality_score < self.config.quality_threshold)
            
            if has_errors or has_quality_issues:
                return "refine"
            
            return "end"
            
        return should_refine_condition

    def compile(self):
        """Compile the workflow graph."""
        memory = MemorySaver() if self.config.enable_artifact_management else None

        return self.graph.compile(checkpointer=memory)
