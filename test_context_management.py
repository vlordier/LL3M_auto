"""Test script for context management system."""

from src.config.context_config import get_context_config
from src.utils.types import ExecutionResult, SubTask, TaskType, WorkflowState
from src.workflow.context_manager import ContextCompactor, ContextMonitor
from src.workflow.selective_state_manager import SelectiveStateManager


def create_large_test_state() -> WorkflowState:
    """Create a state with large context to test compression."""

    # Create large subtasks
    subtasks = []
    for i in range(5):
        subtask = SubTask(
            id=f"task_{i}",
            type=TaskType.GEOMETRY,
            description=f"Complex task {i} with detailed description " * 20,
            priority=i + 1,  # Priority must be >= 1
            dependencies=[],
            parameters={
                "complex_param": "This is a complex parameter with lots of text " * 10,
                "nested_data": {"level1": {"level2": {"data": "nested_content " * 15}}},
                "list_data": [f"item_{j}" for j in range(20)],
            },
        )
        subtasks.append(subtask)

    # Create large documentation
    large_docs = """
# Blender Python API Documentation

## Geometry Operations
""" + "\n".join([f"Function {i}: detailed explanation " * 30 for i in range(50)])

    # Create large generated code
    large_code = "import bpy\nimport mathutils\n\n"
    for i in range(100):
        large_code += f"""
def function_{i}():
    '''Complex function {i} with detailed implementation'''
    bpy.ops.mesh.primitive_cube_add(location=({i}, {i}, {i}))
    # Additional processing code here
    for j in range(10):
        print(f'Processing step {{j}} in function {i}')
"""

    # Create execution result with logs
    execution_result = ExecutionResult(
        success=True,
        asset_path="/path/to/asset.blend",
        screenshot_path="/path/to/screenshot.png",
        logs=[f"Log message {i}: " + "detailed log content " * 20 for i in range(20)],
        errors=[],
        execution_time=5.0,
    )

    # Create large analysis results
    critic_analysis = {
        "overall_score": 7,
        "visual_quality": {"score": 6, "notes": "Quality notes " * 50},
        "geometry": {"score": 8, "notes": "Geometry assessment " * 40},
        "critical_issues": [f"Issue {i}: " + "detailed issue description " * 15 for i in range(8)],
        "improvement_suggestions": [f"Suggestion {i}: " + "detailed suggestion " * 20 for i in range(10)],
        "detailed_analysis": "Comprehensive analysis " * 100,
    }

    verification_result = {
        "quality_score": 6.5,
        "polygon_count": 1234,
        "material_count": 5,
        "detailed_metrics": {f"metric_{i}": f"value_{i} " * 30 for i in range(20)},
        "recommendations": [f"Recommendation {i}: " + "detailed rec " * 25 for i in range(15)],
    }

    return WorkflowState(
        prompt=(
            "Create a complex 3D architectural scene with multiple buildings, detailed textures, "
            "advanced lighting, and realistic materials for a commercial visualization project "
        )
        * 10,
        original_prompt="Create an ovoid",
        subtasks=subtasks,
        documentation=large_docs,
        generated_code=large_code,
        execution_result=execution_result,
        critic_analysis=critic_analysis,
        verification_result=verification_result,
        refinement_request="Improve lighting and add more detail to materials " * 20,
        refinement_iterations=2,
    )


def test_context_monitoring():
    """Test context monitoring functionality."""
    print("🧪 Testing Context Monitoring")
    print("=" * 50)

    config = get_context_config()
    monitor = ContextMonitor(max_tokens=config.gpt_5_max_tokens)  # Use GPT-5 limits for testing

    # Create progressively larger states
    small_state = WorkflowState(prompt="Simple prompt")
    medium_state = WorkflowState(prompt="Medium complexity prompt " * 100, documentation="Some docs " * 200)
    large_state = create_large_test_state()

    print("Small state:")
    monitor.check_and_warn(small_state, "small_test")

    print("\nMedium state:")
    monitor.check_and_warn(medium_state, "medium_test")

    print("\nLarge state:")
    monitor.check_and_warn(large_state, "large_test")

    print("\n✅ Context monitoring test completed\n")


def test_context_compression():
    """Test context compression functionality."""
    print("🗜️  Testing Context Compression")
    print("=" * 50)

    compactor = ContextCompactor(max_tokens=200000, compression_threshold=0.7)  # Lower threshold for testing
    large_state = create_large_test_state()

    # Calculate initial size
    initial_tokens = compactor._calculate_state_tokens(large_state)
    print(f"📊 Initial state tokens: {initial_tokens}")

    # Test compression decision
    should_compress = compactor.should_compress(large_state)
    print(f"🤔 Should compress: {should_compress}")

    if should_compress:
        # Apply compression
        compressed_state = compactor.compress_state(large_state)
        compressed_tokens = compactor._calculate_state_tokens(compressed_state)

        compression_ratio = compressed_tokens / initial_tokens
        print(f"📉 Compressed state tokens: {compressed_tokens}")
        print(f"📈 Compression ratio: {compression_ratio:.2%}")
        print(f"💾 Space saved: {initial_tokens - compressed_tokens} tokens")

        # Show what was compressed
        print("\n🔍 Compression details:")
        print(f"  - Original docs length: {len(large_state.documentation or '')}")
        print(f"  - Compressed docs length: {len(compressed_state.documentation or '')}")
        print(f"  - Original code length: {len(large_state.generated_code or '')}")
        print(f"  - Compressed code length: {len(compressed_state.generated_code or '')}")
        print(f"  - Original subtasks: {len(large_state.subtasks or [])}")
        print(f"  - Compressed subtasks: {len(compressed_state.subtasks or [])}")

    print("\n✅ Context compression test completed\n")


def test_selective_state_management():
    """Test selective state management for different agent types."""
    print("🎯 Testing Selective State Management")
    print("=" * 50)

    manager = SelectiveStateManager()
    large_state = create_large_test_state()

    agent_types = [
        "planner",
        "retrieval",
        "coding",
        "critic",
        "verification",
        "user_proxy",
    ]

    for agent_type in agent_types:
        selective_context = manager.prepare_for_agent(large_state, agent_type)

        # Calculate size of selective context
        import json

        context_str = json.dumps(selective_context, default=str)
        context_tokens = len(context_str) // 4

        print(f"📋 {agent_type.ljust(12)}: {context_tokens:4d} tokens, {len(selective_context):2d} fields")

    print("\n✅ Selective state management test completed\n")


def test_error_recovery():
    """Test error recovery simulation."""
    print("🚨 Testing Error Recovery")
    print("=" * 50)

    from src.workflow.error_recovery import ErrorRecoveryMiddleware

    recovery = ErrorRecoveryMiddleware()

    # Test error detection
    test_errors = [
        Exception("This model's maximum context length is 4097 tokens"),
        Exception("context_length_exceeded"),
        Exception("requested too many tokens"),
        Exception("Some other error"),
    ]

    for error in test_errors:
        is_context_error = recovery.is_context_error(error)
        print(f"🔍 '{str(error)[:50]}...' -> Context error: {is_context_error}")

    print("\n✅ Error recovery test completed\n")


def main():
    """Run all context management tests."""
    print("🚀 LangGraph Context Management Test Suite")
    print("=" * 60)
    print()

    test_context_monitoring()
    test_context_compression()
    test_selective_state_management()
    test_error_recovery()

    print("🎉 All tests completed successfully!")
    print("\n💡 Integration tips:")
    print("  1. Use @with_error_recovery('agent_type') decorator on workflow nodes")
    print("  2. Add monitor.check_and_warn(state, 'stage') in workflow nodes")
    print("  3. Use compactor.monitor_and_compress(state) before heavy operations")
    print("  4. Configure thresholds in context_config.py for your use case")


if __name__ == "__main__":
    main()
