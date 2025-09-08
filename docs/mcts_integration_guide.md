# MCTS (LATS) Integration Guide for LL3M

## Overview

The MCTS (Monte Carlo Tree Search) agent implements a LATS (Language Agent Tree Search) approach for sophisticated task decomposition in the LL3M workflow system. This agent provides enhanced planning capabilities using tree search algorithms to explore multiple decomposition strategies and select optimal task breakdowns.

## Architecture

### Core Components

1. **MCTSTaskDecomposer**: Main agent implementing full MCTS algorithm
2. **MCTSNode**: Tree node structure for MCTS with task information and statistics
3. **MCTSState**: State management for tree search with best path tracking
4. **MCTSAgentConfig**: Type-safe configuration with exploration parameters

### MCTS Algorithm Implementation

The agent implements the complete MCTS cycle:

1. **Selection**: UCB1 (Upper Confidence Bound) strategy for node selection
2. **Expansion**: Generate multiple decomposition strategies using LLM
3. **Simulation**: Multi-criteria evaluation (feasibility, complexity, completeness)
4. **Backpropagation**: Update statistics throughout the tree path

## Integration Points

### 1. Planner Replacement

Replace the standard planner with MCTS for enhanced task decomposition:

```python
# In workflow configuration
factory = get_agent_factory()
mcts_planner = factory.create_mcts(
    max_iterations=20,
    exploration_weight=1.414,
    max_depth=3
)

# Use in workflow node
async def enhanced_planner_node(state: WorkflowState) -> WorkflowState:
    mcts_planner = _get_agent_factory().create_mcts()
    response = await mcts_planner.process(state)
    # Handle response...
```

### 2. Adaptive Planning

Dynamically choose between standard and MCTS planning based on task complexity:

```python
from src.workflow.enhanced_planning import adaptive_planner_node

# Uses heuristics to determine planning strategy:
# - Task complexity indicators
# - Quality requirements
# - Multi-object scenarios
# - Scene complexity
```

### 3. Hybrid Planning

Run both standard and MCTS planners concurrently and select the best result:

```python
from src.workflow.enhanced_planning import hybrid_planner_node

# Executes both planners and uses quality scoring to select optimal result
```

### 4. Refinement Enhancement

Use MCTS for sophisticated refinement task decomposition:

```python
from src.workflow.enhanced_planning import mcts_enhanced_refinement_node

# Applies MCTS reasoning to refinement requests
```

## Configuration

### Basic Configuration

```python
mcts_config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1500,
    # MCTS-specific settings
    "max_iterations": 20,
    "max_depth": 3,
    "exploration_weight": 1.414,  # sqrt(2) recommended
    "min_subtasks": 2,
    "max_subtasks": 8,
    # Evaluation thresholds
    "feasibility_threshold": 0.6,
    "complexity_penalty": 0.3,
    "completeness_bonus": 0.2,
}
```

### Advanced Configuration

For complex scenarios requiring extensive exploration:

```python
complex_mcts_config = {
    "max_iterations": 50,
    "max_depth": 4,
    "exploration_weight": 2.0,  # Higher exploration
    "max_subtasks": 12,
    # Lower thresholds for more thorough search
    "feasibility_threshold": 0.5,
}
```

## Usage Patterns

### When to Use MCTS

**Recommended for:**
- Complex multi-object scenes
- Photorealistic/professional quality requirements
- Tasks with "and" connectors (multiple requirements)
- Procedural or parametric modeling
- Animation and rigging tasks
- Tasks requiring sophisticated lighting setups

**Standard planner sufficient for:**
- Simple geometric objects
- Basic material application
- Straightforward single-object tasks
- Quick prototyping

### Performance Considerations

| Aspect | Standard Planner | MCTS Planner |
|--------|------------------|--------------|
| Speed | ~0.1s | ~2.5s |
| Subtasks Generated | 1-3 | 3-8 |
| Task Detail Level | Basic | Comprehensive |
| Dependency Analysis | Simple | Advanced |
| Quality Optimization | Limited | Multi-criteria |

## Quality Metrics

The MCTS agent provides rich metadata for analysis:

```python
response = await mcts_agent.process(state)
metadata = response.metadata
# Available metrics:
# - mcts_iterations: Number of MCTS iterations performed
# - total_nodes: Total nodes explored in tree
# - max_depth_reached: Maximum tree depth achieved
# - exploration_strategy: Algorithm used (UCB1)
```

## Multi-Criteria Evaluation

The MCTS agent evaluates task decompositions across three dimensions:

### 1. Feasibility (40% weight)
- Reasonable number of subtasks (optimal ~5)
- Balanced complexity distribution
- Clear task descriptions

### 2. Complexity (30% weight, inverted)
- Number of dependencies
- Task complexity ratings (based on priority)
- Task type diversity

### 3. Completeness (30% weight)
- Coverage of expected task types
- Priority distribution variety
- Logical task flow

## Best Practices

### 1. Configuration Tuning

```python
# For detailed architectural visualization:
architectural_config = {
    "max_iterations": 30,
    "max_depth": 4,
    "exploration_weight": 1.2,
    "max_subtasks": 10,
}

# For quick concept exploration:
concept_config = {
    "max_iterations": 15,
    "max_depth": 2,
    "exploration_weight": 1.8,
    "max_subtasks": 6,
}
```

### 2. Error Handling

```python
response = await mcts_agent.process(state)
if not response.success:
    logger.warning("MCTS planning failed, falling back to standard planner")
    fallback_planner = factory.create_planner()
    response = await fallback_planner.process(state)
```

### 3. Monitoring and Analysis

```python
# Analyze MCTS performance
if response.metadata:
    efficiency = response.metadata.get('mcts_iterations', 0) / config['max_iterations']
    coverage = response.metadata.get('total_nodes', 0)
    logger.info(f"MCTS efficiency: {efficiency:.2f}, coverage: {coverage} nodes")
```

## Integration Testing

The system includes comprehensive tests covering:

- **Unit Tests**: MCTS node operations, algorithm phases, evaluation methods
- **Integration Tests**: Workflow integration, error handling, performance comparison
- **Configuration Tests**: Parameter validation, default handling

Run tests with:
```bash
# Unit tests
python -m pytest tests/unit/test_mcts_agent.py -v

# Integration tests
python -m pytest tests/integration/test_mcts_workflow_integration.py -v
```

## Future Enhancements

### Planned Features
1. **Dynamic Exploration**: Adjust exploration parameters based on task success rates
2. **Learning Integration**: Cache successful decomposition patterns
3. **Multi-Agent MCTS**: Coordinate multiple agents using tree search
4. **Quality Prediction**: Predict final asset quality from decomposition structure

### Extension Points
- Custom evaluation criteria
- Alternative selection strategies (Thompson Sampling, PUCT)
- Parallel tree search for complex scenes
- Integration with reinforcement learning

## Troubleshooting

### Common Issues

**MCTS takes too long:**
- Reduce `max_iterations` or `max_depth`
- Increase `exploration_weight` for faster convergence

**Generated tasks are too simple:**
- Increase `max_iterations` and `max_depth`
- Lower `feasibility_threshold`

**Tree search gets stuck:**
- Verify LLM is generating diverse strategies
- Check `exploration_weight` isn't too low
- Ensure adequate `max_subtasks` limit

### Debug Information

Enable detailed logging:
```python
import structlog
logger = structlog.get_logger(__name__)
logger.setLevel("DEBUG")  # Shows MCTS iteration details
```

## Conclusion

The MCTS agent provides sophisticated task decomposition capabilities that significantly enhance the LL3M workflow's planning phase. By using tree search algorithms with multi-criteria evaluation, it generates higher-quality task breakdowns that lead to better final 3D assets, especially for complex scenarios requiring professional-level results.

The modular design allows for flexible integration as a planner replacement, enhancement, or hybrid approach, making it adaptable to various use cases and performance requirements.
