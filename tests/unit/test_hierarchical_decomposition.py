"""Test hierarchical decomposition functionality in MCTS agent."""

import pytest
from unittest.mock import patch

from src.agents.mcts import (
    MCTSTaskDecomposer, 
    get_task_complexity_level, 
    sort_tasks_by_complexity,
    TASK_COMPLEXITY_LEVELS
)
from src.utils.types import SubTask, TaskType
from tests.utils.builders import WorkflowStateBuilder


class TestHierarchicalDecomposition:
    """Test hierarchical decomposition features."""
    
    def test_task_complexity_levels(self):
        """Test that task types have correct complexity levels."""
        assert get_task_complexity_level(TaskType.GEOMETRY) == 1
        assert get_task_complexity_level(TaskType.SCENE_SETUP) == 2
        assert get_task_complexity_level(TaskType.MATERIAL) == 3
        assert get_task_complexity_level(TaskType.LIGHTING) == 4
        assert get_task_complexity_level(TaskType.ANIMATION) == 5
    
    def test_sort_tasks_by_complexity(self):
        """Test that tasks are sorted correctly by complexity."""
        tasks = [
            SubTask(id="t1", type=TaskType.LIGHTING, description="Light", priority=4, dependencies=[]),
            SubTask(id="t2", type=TaskType.GEOMETRY, description="Cube", priority=5, dependencies=[]),
            SubTask(id="t3", type=TaskType.MATERIAL, description="Texture", priority=3, dependencies=[]),
            SubTask(id="t4", type=TaskType.ANIMATION, description="Rotate", priority=2, dependencies=[]),
            SubTask(id="t5", type=TaskType.SCENE_SETUP, description="Camera", priority=4, dependencies=[]),
        ]
        
        sorted_tasks = sort_tasks_by_complexity(tasks)
        
        # Should be sorted by complexity level first
        complexity_sequence = [get_task_complexity_level(task.type) for task in sorted_tasks]
        assert complexity_sequence == [1, 2, 3, 4, 5]
    
    def test_sort_tasks_with_same_complexity(self):
        """Test sorting with tasks of same complexity level."""
        tasks = [
            SubTask(id="t1", type=TaskType.GEOMETRY, description="Cube", priority=3, dependencies=[]),
            SubTask(id="t2", type=TaskType.GEOMETRY, description="Sphere", priority=5, dependencies=[]),
            SubTask(id="t3", type=TaskType.GEOMETRY, description="Plane", priority=4, dependencies=[]),
        ]
        
        sorted_tasks = sort_tasks_by_complexity(tasks)
        
        # All have same complexity, should be sorted by priority (descending)
        priorities = [task.priority for task in sorted_tasks]
        assert priorities == [5, 4, 3]
        assert sorted_tasks[0].description == "Sphere"
        assert sorted_tasks[1].description == "Plane"
        assert sorted_tasks[2].description == "Cube"
    
    @pytest.mark.asyncio
    async def test_hierarchical_progression_evaluation(self, mcts_config):
        """Test evaluation of hierarchical progression in subtasks."""
        with patch("src.agents.base.get_settings"), \
             patch("src.agents.base.LMStudioClient"), \
             patch("src.agents.base.OpenAIClient"):
            
            agent = MCTSTaskDecomposer(mcts_config)
            
            # Good progression: geometry -> materials -> lighting
            good_tasks = [
                SubTask(id="t1", type=TaskType.GEOMETRY, description="Create cube", priority=5, dependencies=[]),
                SubTask(id="t2", type=TaskType.MATERIAL, description="Apply texture", priority=3, dependencies=["t1"]),
                SubTask(id="t3", type=TaskType.LIGHTING, description="Add lights", priority=4, dependencies=["t2"]),
            ]
            
            # Bad progression: lighting -> geometry -> materials (backwards)
            bad_tasks = [
                SubTask(id="t1", type=TaskType.LIGHTING, description="Add lights", priority=4, dependencies=[]),
                SubTask(id="t2", type=TaskType.GEOMETRY, description="Create cube", priority=5, dependencies=[]),
                SubTask(id="t3", type=TaskType.MATERIAL, description="Apply texture", priority=3, dependencies=[]),
            ]
            
            good_score = agent._evaluate_hierarchical_progression(good_tasks)
            bad_score = agent._evaluate_hierarchical_progression(bad_tasks)
            
            # Good progression should score higher than bad progression
            assert good_score > bad_score
            assert good_score > 0.5  # Should be reasonably high
            assert bad_score < 0.5   # Should be lower
    
    @pytest.mark.asyncio
    async def test_complexity_evaluation_with_hierarchy(self, mcts_config):
        """Test that complexity evaluation considers hierarchical progression."""
        with patch("src.agents.base.get_settings"), \
             patch("src.agents.base.LMStudioClient"), \
             patch("src.agents.base.OpenAIClient"):
            
            agent = MCTSTaskDecomposer(mcts_config)
            state = WorkflowStateBuilder().with_prompt("Create a textured cube").build()
            
            from src.agents.mcts import MCTSNode
            
            # Node with good hierarchical progression
            good_node = MCTSNode(
                task_description="hierarchical approach",
                subtasks=[
                    SubTask(id="t1", type=TaskType.GEOMETRY, description="Create cube", priority=5, dependencies=[]),
                    SubTask(id="t2", type=TaskType.MATERIAL, description="Apply texture", priority=3, dependencies=["t1"]),
                ]
            )
            
            # Node with poor hierarchical progression
            bad_node = MCTSNode(
                task_description="non-hierarchical approach", 
                subtasks=[
                    SubTask(id="t1", type=TaskType.MATERIAL, description="Apply texture", priority=3, dependencies=[]),
                    SubTask(id="t2", type=TaskType.GEOMETRY, description="Create cube", priority=5, dependencies=["t1"]),
                ]
            )
            
            good_complexity = await agent._evaluate_complexity(good_node, state)
            bad_complexity = await agent._evaluate_complexity(bad_node, state)
            
            # Good hierarchy should have lower complexity (better for MCTS)
            assert good_complexity < bad_complexity
    
    def test_depth_based_guidance_generation(self, mcts_config):
        """Test that prompts include depth-appropriate guidance."""
        with patch("src.agents.base.get_settings"), \
             patch("src.agents.base.LMStudioClient"), \
             patch("src.agents.base.OpenAIClient"):
            
            agent = MCTSTaskDecomposer(mcts_config)
            state = WorkflowStateBuilder().with_prompt("Create a wooden table").build()
            
            from src.agents.mcts import MCTSNode
            
            # Test different depths
            root_node = MCTSNode(task_description="root task")
            prompt_depth_0 = agent._create_decomposition_prompt(root_node, state)
            assert "GEOMETRY (level 1)" in prompt_depth_0
            assert "DEPTH 0" in prompt_depth_0
            
            # Simulate deeper node
            child_node = MCTSNode(task_description="child task", parent=root_node)
            grandchild_node = MCTSNode(task_description="grandchild task", parent=child_node)
            prompt_depth_2 = agent._create_decomposition_prompt(grandchild_node, state)
            assert "MATERIAL (level 3)" in prompt_depth_2
            assert "DEPTH 2" in prompt_depth_2


class TestTaskComplexityMapping:
    """Test the task complexity mapping system."""
    
    def test_all_task_types_have_complexity_levels(self):
        """Ensure all TaskType enum values have complexity levels."""
        for task_type in TaskType:
            complexity_level = get_task_complexity_level(task_type)
            assert 1 <= complexity_level <= 5, f"Task type {task_type} has invalid complexity level {complexity_level}"
    
    def test_complexity_levels_are_unique_and_ordered(self):
        """Test that complexity levels form a proper hierarchy."""
        levels = list(TASK_COMPLEXITY_LEVELS.values())
        unique_levels = set(levels)
        
        # All levels should be unique
        assert len(levels) == len(unique_levels), "Complexity levels should be unique"
        
        # Should form a complete sequence from 1 to max
        max_level = max(levels)
        expected_levels = set(range(1, max_level + 1))
        assert unique_levels == expected_levels, "Complexity levels should form complete sequence"
    
    def test_hierarchical_ordering_makes_sense(self):
        """Test that the hierarchical ordering follows logical 3D workflow."""
        # Geometry should be simplest (foundational)
        assert get_task_complexity_level(TaskType.GEOMETRY) == 1
        
        # Animation should be most complex (final polish)
        assert get_task_complexity_level(TaskType.ANIMATION) == 5
        
        # Materials should come before lighting (you texture before you light)
        material_level = get_task_complexity_level(TaskType.MATERIAL)
        lighting_level = get_task_complexity_level(TaskType.LIGHTING)
        assert material_level < lighting_level, "Materials should come before lighting in workflow"