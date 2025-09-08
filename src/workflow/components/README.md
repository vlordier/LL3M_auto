# Modular Workflow Components

This directory contains the modular workflow architecture that replaces the monolithic workflow design. The modular approach provides better separation of concerns, configurability, and maintainability.

## Architecture Overview

The modular workflow system is built around these core principles:

1. **Separation of Concerns**: Each component handles a specific aspect of workflow execution
2. **Configuration-Driven**: Workflow behavior is controlled through declarative configuration
3. **Pluggable Components**: Quality assessment and refinement strategies can be swapped out
4. **Template System**: Pre-configured workflows for common use cases
5. **Composition Utilities**: Tools for creating hybrid and custom workflows

## Components

### Core Components (`core_components.py`)
- **ExecutionPipeline**: Handles the main execution flow (planner → retrieval → coding → execution)
- **CoreWorkflowBuilder**: Base builder for constructing workflow graphs

### Quality Components (`quality_components.py`)
- **QualityGate**: Individual quality assessment steps
- **QualityAssessmentPipeline**: Configurable quality evaluation pipeline
- **should_assess_quality**: Smart quality assessment gating

### Refinement Components (`refinement_components.py`)
- **RefinementStrategy**: Individual refinement approaches
- **RefinementEngine**: Configurable refinement pipeline with adaptive strategies
- **should_refine_enhanced**: Enhanced refinement decision logic

### Workflow Builder (`workflow_builder.py`)
- **WorkflowBuilder**: Main builder for creating configured workflows
- **WorkflowConfig**: Configuration dataclass with enums for modes and levels
- **Factory Functions**: Convenience functions for common workflow types

### Composition Utilities (`composition.py`)
- **WorkflowComposer**: Template-based workflow creation
- **WorkflowTemplate**: Predefined workflow configurations
- **CompiledWorkflow**: Workflow wrapper with metadata

## Workflow Modes

### Fast Mode
- **Purpose**: Development and rapid prototyping
- **Features**: Minimal quality checks, single refinement max
- **Use Case**: Quick iteration and testing

### Standard Mode
- **Purpose**: Balanced production workflows
- **Features**: Standard quality assessment, limited refinement (2 max)
- **Use Case**: Most production scenarios

### Thorough Mode
- **Purpose**: Research and high-quality outputs
- **Features**: Comprehensive quality pipeline, adaptive refinement (3+ max)
- **Use Case**: Research, critical applications

### Custom Mode
- **Purpose**: User-defined workflows
- **Features**: Fully configurable nodes, edges, and conditions
- **Use Case**: Specialized requirements

## Quality Levels

- **NONE**: No quality assessment
- **BASIC**: Quick validation only
- **STANDARD**: Critic analysis
- **COMPREHENSIVE**: Full critic + verification + assessment

## Refinement Modes

- **NONE**: No refinement
- **SINGLE**: One refinement attempt max
- **LIMITED**: Up to 2 refinements
- **ADAPTIVE**: Dynamic refinement based on quality analysis

## Templates

### Development
```python
from workflow.components import create_development_workflow
workflow = create_development_workflow()
```

### Production
```python
from workflow.components import create_production_workflow
workflow = create_production_workflow(
    quality_threshold=2.5,
    max_refinements=2
)
```

### Research
```python
from workflow.components import create_research_workflow
workflow = create_research_workflow(
    quality_threshold=3.0,
    max_refinements=4
)
```

### Quick Prototype
```python
from workflow.components import create_prototype_workflow
workflow = create_prototype_workflow()  # Minimal overhead
```

### Quality Focused
```python
from workflow.components import create_quality_workflow
workflow = create_quality_workflow(
    quality_threshold=4.0,
    critical_issues_threshold=1
)
```

## Usage Examples

### Basic Usage
```python
from workflow.components import WorkflowBuilder, WorkflowConfig, WorkflowMode

# Create a standard workflow
config = WorkflowConfig(mode=WorkflowMode.STANDARD)
builder = WorkflowBuilder(config)
workflow = builder.build()

# Execute
result = await workflow.invoke(initial_state)
```

### Template Usage
```python
from workflow.components import workflow_composer

# List available templates
templates = workflow_composer.list_templates()
print(templates)  # ['development', 'production', 'research', ...]

# Create from template
workflow = workflow_composer.create_from_template(
    "production", 
    max_refinements=3,
    quality_threshold=2.0
)

# Get template info
info = workflow_composer.get_template_info("production")
```

### Custom Workflow
```python
from workflow.components import workflow_composer

# Create custom workflow based on production template
custom_workflow = workflow_composer.create_custom_workflow(
    name="High Quality API Testing",
    description="Custom workflow for API endpoint testing",
    base_template="production",
    max_refinements=4,
    quality_threshold=3.5,
    critical_issues_threshold=2,
    recursion_limit=40
)

summary = custom_workflow.get_summary()
```

### Hybrid Workflow
```python
from workflow.components import workflow_composer

# Merge multiple templates
hybrid = workflow_composer.create_hybrid_workflow(
    templates=["development", "quality_focused"],
    merge_strategy="balanced"  # or "conservative" or "aggressive"
)
```

## Migration from Monolithic Workflow

The old monolithic workflow in `workflow/workflow.py` can be replaced with modular components:

```python
# Old way
from workflow.workflow import create_workflow
workflow = create_workflow()

# New way - equivalent functionality
from workflow.components import create_standard_workflow
workflow = create_standard_workflow()

# Or more specific
from workflow.components import WorkflowBuilder, WorkflowConfig
config = WorkflowConfig(
    quality_threshold=2.0,
    critical_issues_threshold=5,
    max_refinements=2
)
builder = WorkflowBuilder(config)
workflow = builder.build()
```

## Key Improvements

1. **Eliminated Infinite Loop Risk**: Refinement now goes to `coding` not `planner`
2. **Smart Quality Gating**: Skip expensive assessment for obviously successful results
3. **Better Stop Conditions**: Multiple hard stops prevent recursion issues
4. **Configurable Behavior**: Easy to adjust thresholds and limits per use case
5. **Template System**: Preconfigured workflows for common scenarios
6. **Composition Tools**: Mix and match components for custom needs

## Configuration Examples

### Conservative (High Quality)
```python
config = WorkflowConfig(
    mode=WorkflowMode.THOROUGH,
    quality_level=QualityLevel.COMPREHENSIVE,
    refinement_mode=RefinementMode.ADAPTIVE,
    quality_threshold=3.5,
    critical_issues_threshold=1,
    max_refinements=4
)
```

### Aggressive (Fast Execution)
```python
config = WorkflowConfig(
    mode=WorkflowMode.FAST,
    quality_level=QualityLevel.BASIC,
    refinement_mode=RefinementMode.SINGLE,
    quality_threshold=1.5,
    critical_issues_threshold=10,
    max_refinements=1,
    skip_perfect_executions=True
)
```

This modular system addresses the cycle issues identified in the analysis while providing flexible, maintainable workflow construction.