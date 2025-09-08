"""Workflow composition utilities for flexible workflow construction."""

from typing import Dict, List, Callable, Optional, Any
import structlog
from dataclasses import dataclass

from ...utils.types import WorkflowState
from .workflow_builder import WorkflowConfig, WorkflowMode, QualityLevel, RefinementMode
from .core_components import CoreWorkflowBuilder, ExecutionPipeline
from .quality_components import QualityAssessmentPipeline
from .refinement_components import RefinementEngine

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowTemplate:
    """Template for creating preconfigured workflows."""
    name: str
    description: str
    config: WorkflowConfig
    custom_nodes: Optional[Dict[str, Callable]] = None
    custom_conditions: Optional[Dict[str, Callable]] = None


class WorkflowComposer:
    """Utility for composing workflows from components."""
    
    def __init__(self):
        self.templates = self._create_default_templates()
        
    def _create_default_templates(self) -> Dict[str, WorkflowTemplate]:
        """Create default workflow templates."""
        return {
            "development": WorkflowTemplate(
                name="Development Workflow",
                description="Fast workflow for development and testing",
                config=WorkflowConfig(
                    mode=WorkflowMode.FAST,
                    quality_level=QualityLevel.BASIC,
                    refinement_mode=RefinementMode.SINGLE,
                    max_refinements=1,
                    quality_threshold=1.5,
                    recursion_limit=15
                )
            ),
            
            "production": WorkflowTemplate(
                name="Production Workflow", 
                description="Balanced workflow for production use",
                config=WorkflowConfig(
                    mode=WorkflowMode.STANDARD,
                    quality_level=QualityLevel.STANDARD,
                    refinement_mode=RefinementMode.LIMITED,
                    max_refinements=2,
                    quality_threshold=2.5,
                    critical_issues_threshold=3,
                    recursion_limit=30
                )
            ),
            
            "research": WorkflowTemplate(
                name="Research Workflow",
                description="Comprehensive workflow for research and experimentation",
                config=WorkflowConfig(
                    mode=WorkflowMode.THOROUGH,
                    quality_level=QualityLevel.COMPREHENSIVE,
                    refinement_mode=RefinementMode.ADAPTIVE,
                    max_refinements=3,
                    quality_threshold=3.0,
                    critical_issues_threshold=2,
                    recursion_limit=40
                )
            ),
            
            "quick_prototype": WorkflowTemplate(
                name="Quick Prototype",
                description="Minimal workflow for rapid prototyping",
                config=WorkflowConfig(
                    mode=WorkflowMode.FAST,
                    quality_level=QualityLevel.NONE,
                    refinement_mode=RefinementMode.NONE,
                    max_refinements=0,
                    skip_perfect_executions=True,
                    recursion_limit=10
                )
            ),
            
            "quality_focused": WorkflowTemplate(
                name="Quality Focused",
                description="Quality-first workflow with extensive assessment",
                config=WorkflowConfig(
                    mode=WorkflowMode.THOROUGH,
                    quality_level=QualityLevel.COMPREHENSIVE,
                    refinement_mode=RefinementMode.ADAPTIVE,
                    max_refinements=4,
                    quality_threshold=4.0,
                    critical_issues_threshold=1,
                    skip_perfect_executions=False,
                    recursion_limit=50
                )
            )
        }
        
    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template by name."""
        return self.templates.get(name)
        
    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())
        
    def create_from_template(self, template_name: str, **overrides) -> 'CompiledWorkflow':
        """Create a workflow from a template with optional overrides."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
            
        # Apply overrides to config
        config_dict = template.config.__dict__.copy()
        config_dict.update(overrides)
        config = WorkflowConfig(**config_dict)
        
        logger.info(f"Creating workflow from template: {template_name}")
        
        from .workflow_builder import WorkflowBuilder
        builder = WorkflowBuilder(config)
        compiled_workflow = builder.build()
        
        return CompiledWorkflow(
            name=template.name,
            description=template.description,
            config=config,
            workflow=compiled_workflow
        )
        
    def create_hybrid_workflow(self, 
                             templates: List[str], 
                             merge_strategy: str = "balanced") -> 'CompiledWorkflow':
        """Create a hybrid workflow by merging multiple templates."""
        if not templates:
            raise ValueError("At least one template required for hybrid workflow")
            
        logger.info(f"Creating hybrid workflow from templates: {templates}")
        
        # Get all template configs
        template_objs = []
        for template_name in templates:
            template = self.get_template(template_name)
            if not template:
                raise ValueError(f"Unknown template: {template_name}")
            template_objs.append(template)
            
        # Merge configurations based on strategy
        if merge_strategy == "balanced":
            config = self._merge_balanced(template_objs)
        elif merge_strategy == "conservative":
            config = self._merge_conservative(template_objs)
        elif merge_strategy == "aggressive":
            config = self._merge_aggressive(template_objs)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            
        # Build hybrid workflow
        from .workflow_builder import WorkflowBuilder
        builder = WorkflowBuilder(config)
        compiled_workflow = builder.build()
        
        return CompiledWorkflow(
            name=f"Hybrid ({', '.join(templates)})",
            description=f"Hybrid workflow using {merge_strategy} merge strategy",
            config=config,
            workflow=compiled_workflow
        )
        
    def _merge_balanced(self, templates: List[WorkflowTemplate]) -> WorkflowConfig:
        """Merge templates using balanced strategy (averages)."""
        configs = [t.config for t in templates]
        
        return WorkflowConfig(
            mode=WorkflowMode.STANDARD,  # Default to standard
            quality_level=self._most_common([c.quality_level for c in configs]),
            refinement_mode=self._most_common([c.refinement_mode for c in configs]),
            max_refinements=int(sum(c.max_refinements for c in configs) / len(configs)),
            quality_threshold=sum(c.quality_threshold for c in configs) / len(configs),
            critical_issues_threshold=int(sum(c.critical_issues_threshold for c in configs) / len(configs)),
            recursion_limit=int(sum(c.recursion_limit for c in configs) / len(configs)),
            enable_caching=any(c.enable_caching for c in configs),
            enable_artifact_management=any(c.enable_artifact_management for c in configs),
            skip_perfect_executions=any(c.skip_perfect_executions for c in configs)
        )
        
    def _merge_conservative(self, templates: List[WorkflowTemplate]) -> WorkflowConfig:
        """Merge templates using conservative strategy (safer limits)."""
        configs = [t.config for t in templates]
        
        return WorkflowConfig(
            mode=WorkflowMode.THOROUGH,
            quality_level=max([c.quality_level for c in configs], key=lambda x: x.value),
            refinement_mode=max([c.refinement_mode for c in configs], key=lambda x: x.value),
            max_refinements=max(c.max_refinements for c in configs),
            quality_threshold=max(c.quality_threshold for c in configs),
            critical_issues_threshold=min(c.critical_issues_threshold for c in configs),
            recursion_limit=max(c.recursion_limit for c in configs),
            enable_caching=all(c.enable_caching for c in configs),
            enable_artifact_management=all(c.enable_artifact_management for c in configs),
            skip_perfect_executions=False  # Conservative - always assess
        )
        
    def _merge_aggressive(self, templates: List[WorkflowTemplate]) -> WorkflowConfig:
        """Merge templates using aggressive strategy (faster execution)."""
        configs = [t.config for t in templates]
        
        return WorkflowConfig(
            mode=WorkflowMode.FAST,
            quality_level=min([c.quality_level for c in configs], key=lambda x: x.value),
            refinement_mode=min([c.refinement_mode for c in configs], key=lambda x: x.value),
            max_refinements=min(c.max_refinements for c in configs),
            quality_threshold=min(c.quality_threshold for c in configs),
            critical_issues_threshold=max(c.critical_issues_threshold for c in configs),
            recursion_limit=min(c.recursion_limit for c in configs),
            enable_caching=any(c.enable_caching for c in configs),
            enable_artifact_management=any(c.enable_artifact_management for c in configs),
            skip_perfect_executions=True  # Aggressive - skip when possible
        )
        
    def _most_common(self, items: List[Any]) -> Any:
        """Return the most common item in a list."""
        from collections import Counter
        return Counter(items).most_common(1)[0][0]
        
    def create_custom_workflow(self,
                             name: str,
                             description: str,
                             base_template: str = "production",
                             **config_overrides) -> 'CompiledWorkflow':
        """Create a custom workflow based on a template with overrides."""
        logger.info(f"Creating custom workflow: {name} based on {base_template}")
        
        template = self.get_template(base_template)
        if not template:
            raise ValueError(f"Unknown base template: {base_template}")
            
        # Apply overrides
        config_dict = template.config.__dict__.copy()
        config_dict.update(config_overrides)
        config = WorkflowConfig(**config_dict)
        
        from .workflow_builder import WorkflowBuilder
        builder = WorkflowBuilder(config)
        compiled_workflow = builder.build()
        
        return CompiledWorkflow(
            name=name,
            description=description,
            config=config,
            workflow=compiled_workflow
        )


@dataclass
class CompiledWorkflow:
    """A compiled workflow with metadata."""
    name: str
    description: str
    config: WorkflowConfig
    workflow: Any  # CompiledStateGraph[WorkflowState]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.config.mode.value,
            "quality_level": self.config.quality_level.value,
            "refinement_mode": self.config.refinement_mode.value,
            "max_refinements": self.config.max_refinements,
            "quality_threshold": self.config.quality_threshold,
            "recursion_limit": self.config.recursion_limit,
            "features": {
                "caching": self.config.enable_caching,
                "artifact_management": self.config.enable_artifact_management,
                "interruption": self.config.enable_interruption,
                "skip_perfect_executions": self.config.skip_perfect_executions
            }
        }


# Global composer instance
workflow_composer = WorkflowComposer()

# Convenience functions
def create_development_workflow(**overrides) -> CompiledWorkflow:
    """Create a development workflow."""
    return workflow_composer.create_from_template("development", **overrides)

def create_production_workflow(**overrides) -> CompiledWorkflow:
    """Create a production workflow."""
    return workflow_composer.create_from_template("production", **overrides)

def create_research_workflow(**overrides) -> CompiledWorkflow:
    """Create a research workflow."""
    return workflow_composer.create_from_template("research", **overrides)

def create_prototype_workflow(**overrides) -> CompiledWorkflow:
    """Create a quick prototype workflow."""
    return workflow_composer.create_from_template("quick_prototype", **overrides)

def create_quality_workflow(**overrides) -> CompiledWorkflow:
    """Create a quality-focused workflow."""
    return workflow_composer.create_from_template("quality_focused", **overrides)

def list_workflow_templates() -> List[str]:
    """List all available workflow templates."""
    return workflow_composer.list_templates()

def get_template_info(template_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a workflow template."""
    template = workflow_composer.get_template(template_name)
    if not template:
        return None
        
    return {
        "name": template.name,
        "description": template.description,
        "config": {
            "mode": template.config.mode.value,
            "quality_level": template.config.quality_level.value,
            "refinement_mode": template.config.refinement_mode.value,
            "max_refinements": template.config.max_refinements,
            "quality_threshold": template.config.quality_threshold,
            "recursion_limit": template.config.recursion_limit
        }
    }