"""Modular workflow components for LL3M system."""

from .composition import (
    CompiledWorkflow,
    WorkflowComposer,
    WorkflowTemplate,
    create_development_workflow,
    create_production_workflow,
    create_prototype_workflow,
    create_quality_workflow,
    create_research_workflow,
    get_template_info,
    list_workflow_templates,
    workflow_composer,
)

# Legacy components (maintained for backward compatibility)
from .core_components import CoreWorkflowBuilder, ExecutionPipeline
from .graph_builder import GraphBuilder
from .graph_builder import create_standard_workflow as streamlined_create_standard_workflow
from .node_factory import ConditionFactory, NodeFactory
from .quality_components import QualityAssessmentPipeline, QualityGate, should_assess_quality
from .refinement_components import (
    RefinementEngine,
    RefinementStrategy,
    create_simple_refinement_node,
    should_refine_enhanced,
)

# New streamlined components (recommended)
from .streamlined_core import StreamlinedWorkflowBuilder, WorkflowTemplates
from .streamlined_quality import QualityAssessment, create_quick_quality_check
from .streamlined_refinement import should_refine_enhanced as streamlined_should_refine
from .workflow_builder import (
    QualityLevel,
    RefinementMode,
    WorkflowBuilder,
    WorkflowConfig,
    WorkflowMode,
    create_fast_workflow,
    create_standard_workflow,
    create_thorough_workflow,
    create_workflow,
)

__all__ = [
    # Core components
    "CoreWorkflowBuilder",
    "ExecutionPipeline",
    "QualityAssessmentPipeline",
    "QualityGate",
    "RefinementEngine",
    "RefinementStrategy",
    "create_simple_refinement_node",
    # Graph building
    "GraphBuilder",
    "streamlined_create_standard_workflow",
    # Factories
    "ConditionFactory",
    "NodeFactory",
    # Streamlined components
    "StreamlinedWorkflowBuilder",
    "WorkflowTemplates",
    "QualityAssessment",
    "create_quick_quality_check",
    "streamlined_should_refine",
    # Workflow builders
    "WorkflowBuilder",
    "WorkflowConfig",
    "WorkflowMode",
    "QualityLevel",
    "RefinementMode",
    "create_workflow",
    "create_fast_workflow",
    "create_standard_workflow",
    "create_thorough_workflow",
    # Conditions
    "should_assess_quality",
    "should_refine_enhanced",
    # Composition utilities
    "WorkflowComposer",
    "WorkflowTemplate",
    "CompiledWorkflow",
    "workflow_composer",
    "create_development_workflow",
    "create_production_workflow",
    "create_research_workflow",
    "create_prototype_workflow",
    "create_quality_workflow",
    "list_workflow_templates",
    "get_template_info",
]
