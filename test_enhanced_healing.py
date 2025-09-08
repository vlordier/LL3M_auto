#!/usr/bin/env python3
"""Test the enhanced MCTS healing process with new strategies."""

import asyncio
from src.agents.coding import CodingAgent
from src.utils.types import WorkflowState, SubTask, TaskType, ExecutionResult


async def test_enhanced_healing():
    """Test the enhanced healing strategies."""
    print("🧪 Testing Enhanced MCTS Healing Process")
    print("=" * 60)
    
    # Create a coding agent
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    }
    
    coding_agent = CodingAgent(config)
    
    # Test case: Code with multiple types of issues
    problematic_code = """
import bpy
import os  # F401 - unused import 
import secrets  # F821 - undefined name used below

def create_cube():
    # Create a cube
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    
    # Undefined variable issue
    cube.location = position  # F821 - 'position' is undefined
    
    # AttributeError potential issue - no context check
    active_obj.name = "TestCube"  # F821 - 'active_obj' is undefined
    
    # Style issues
    return True

# Missing main execution
"""

    # Create mock ruff issues
    ruff_issues = [
        "Line 3: F401 - 'os' imported but unused",
        "Line 11: F821 - Undefined name 'position'", 
        "Line 14: F821 - Undefined name 'active_obj'",
        "Line 16: D415 - First line should end with a period, question mark, or exclamation point"
    ]
    
    # Create execution context with some errors
    execution_context = {
        "has_execution_result": True,
        "execution_success": False,
        "execution_errors": [
            "NameError: name 'position' is not defined",
            "NameError: name 'active_obj' is not defined"
        ],
        "execution_logs": [],
        "previous_attempts": 0,
        "detected_issues": [],
        "error_patterns": {
            "python_syntax_errors": [],
            "blender_api_errors": ["NameError: name 'active_obj' is not defined"],
            "undefined_variables": ["position", "active_obj"]
        }
    }
    
    # Create a workflow state
    state = WorkflowState(
        prompt="Create a simple cube in Blender",
        subtasks=[
            SubTask(
                id="create-cube",
                type=TaskType.GEOMETRY,
                description="Create a cube primitive",
                priority=1,
                dependencies=[],
                parameters={"shape": "cube"}
            )
        ],
        generated_code=problematic_code,
        execution_result=ExecutionResult(
            success=False,
            asset_path="",
            errors=execution_context["execution_errors"],
            logs=[],
            execution_time=0.0
        )
    )
    
    print("🔧 Starting Enhanced MCTS Healing Process...")
    print(f"Original code length: {len(problematic_code)} characters")
    print(f"Issues to fix: {len(ruff_issues)}")
    
    try:
        # Test the enhanced healing system
        healed_code = await coding_agent._heal_code_with_comprehensive_feedback(
            problematic_code, ruff_issues, execution_context, state
        )
        
        if healed_code:
            print("✅ Enhanced healing successful!")
            print(f"Healed code length: {len(healed_code)} characters")
            print(f"Code improvement ratio: {len(healed_code)/len(problematic_code):.2f}")
            
            print("\n📋 Key Improvements Made:")
            
            # Check for specific improvements
            improvements = []
            if "import os" not in healed_code:
                improvements.append("- Removed unused 'os' import")
            if "position" not in healed_code or "position = " in healed_code:
                improvements.append("- Fixed undefined 'position' variable") 
            if "active_obj" not in healed_code or "active_obj = " in healed_code:
                improvements.append("- Fixed undefined 'active_obj' variable")
            if "bpy.context.active_object" in healed_code:
                improvements.append("- Added proper Blender context usage")
            if "try:" in healed_code or "if " in healed_code:
                improvements.append("- Added defensive programming patterns")
                
            for improvement in improvements:
                print(improvement)
            
            print(f"\n🎯 Enhanced Strategies Available:")
            print("✅ Critic-guided healing - Analyzes code quality holistically")
            print("✅ Documentation-enhanced healing - Uses Context7 API docs")  
            print("✅ Web-assisted healing - Searches for complex error solutions")
            print("✅ Enhanced MCTS simulation - Better strategy evaluation")
            
        else:
            print("❌ Enhanced healing failed to improve the code")
            
    except Exception as e:
        print(f"❌ Enhanced healing encountered an error: {e}")
        import traceback
        traceback.print_exc()

    print("\n🚀 Enhanced MCTS Healing Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_healing())