"""Core workflow components for the LL3M system."""

from typing import Any, Callable, Dict, List
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
        from ...agents.planner import create_planner_workflow
        return create_planner_workflow()
        
    def create_retrieval_node(self) -> Callable:
        """Create the retrieval node."""
        from ...agents.retrieval import create_retrieval_workflow
        return create_retrieval_workflow()
        
    def create_coding_node(self) -> Callable:
        """Create the coding node."""
        from ...agents.coding import create_coding_workflow
        return create_coding_workflow()
        
    def create_execution_node(self) -> Callable:
        """Create the execution node."""
        from ...agents.execution import create_execution_workflow
        return create_execution_workflow()
        
    def create_should_continue_condition(self) -> Callable:
        """Create the should_continue_main condition."""
        def should_continue_main(state: WorkflowState) -> str:
            """Determine if main workflow should continue."""
            if state.error_message:
                return "end"
            if state.max_iterations and state.iterations >= state.max_iterations:
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
        self.graph.add_conditional_edges(
            "execution",
            self.pipeline.create_should_continue_condition(),
            {"continue": "quick_quality", "end": END}
        )
        self.graph.add_edge("quick_quality", END)
        
    def add_simple_refinement(self):
        """Add simple refinement loop back to coding."""
        from .refinement_components import create_simple_refinement_check
        
        refinement_check = create_simple_refinement_check(self.config)
        self.graph.add_conditional_edges(
            "execution",
            refinement_check,
            {"refine": "coding", "end": END}
        )
        
    def add_quality_gate(self, quality_assessment: Callable):
        """Add a quality assessment pipeline after execution."""
        self.graph.add_node("quality_assessment", quality_assessment)
        self.graph.add_edge("execution", "quality_assessment")
        self.graph.add_edge("quality_assessment", END)
        
    def add_refinement(self, refinement_logic: Callable):
        """Add refinement logic with loop back to coding."""
        self.graph.add_node("refinement", refinement_logic)
        self.graph.add_conditional_edges(
            "quality_assessment",
            lambda state: "refine" if self._should_refine(state) else "end",
            {"refine": "refinement", "end": END}
        )
        self.graph.add_edge("refinement", "coding")
        
    def _should_refine(self, state: WorkflowState) -> bool:
        """Determine if refinement is needed."""
        if state.max_iterations and state.iterations >= state.max_iterations:
            return False
        if state.refinement_iterations >= self.config.max_refinements:
            return False
        return bool(state.error_message or (hasattr(state, 'quality_score') and 
                                          state.quality_score < self.config.quality_threshold))
        
    def compile(self):
        """Compile the workflow graph."""
        memory = MemorySaver() if self.config.enable_artifact_management else None
        
        return self.graph.compile(
            checkpointer=memory,
            recursion_limit=self.config.recursion_limit
        )