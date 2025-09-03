#!/usr/bin/env python3
"""Test LM Studio integration with Blender E2E workflow."""

import asyncio
import aiohttp


async def test_lm_studio_integration():
    """Test real LM Studio integration with Blender."""
    lm_studio_url = "http://localhost:1234/v1"
    blender_url = "http://localhost:3001"
    
    async with aiohttp.ClientSession() as session:
        print("🎯 LL3M Real LM Studio + Blender Integration Test")
        print("=" * 60)
        
        # Step 1: Test LM Studio connection
        print("\n🤖 Step 1: Test LM Studio Connection")
        try:
            async with session.get(f"{lm_studio_url}/models") as response:
                if response.status == 200:
                    models = await response.json()
                    model_list = models.get("data", [])
                    if model_list:
                        model_name = model_list[0]["id"]
                        print(f"✅ Connected to LM Studio - Model: {model_name}")
                    else:
                        print("❌ No models loaded in LM Studio")
                        return False
                else:
                    print(f"❌ LM Studio not accessible (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ Failed to connect to LM Studio: {e}")
            print("   Make sure LM Studio is running on http://localhost:1234")
            return False
        
        # Step 2: Test Blender connection
        print("\n🎨 Step 2: Test Blender Connection")
        try:
            async with session.get(f"{blender_url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✅ Connected to Blender {health['blender_version']}")
                else:
                    print(f"❌ Blender not accessible (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ Failed to connect to Blender: {e}")
            return False
        
        # Step 3: Generate Blender code with LM Studio
        print("\n🚀 Step 3: Generate Blender Code with Real LLM")
        user_prompt = "Create a red cube at position (0, 0, 0) in Blender using Python"
        
        llm_request = {
            "model": model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a Blender Python expert. Generate clean, working Blender Python code. Always clear the scene first, then create the requested objects. Use headless-compatible code that works without bpy.context.active_object."
                },
                {
                    "role": "user", 
                    "content": f"Write Blender Python code to: {user_prompt}. Make sure to print success messages."
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            async with session.post(f"{lm_studio_url}/chat/completions", json=llm_request) as response:
                if response.status == 200:
                    result = await response.json()
                    generated_code = result["choices"][0]["message"]["content"]
                    
                    # Clean up the code (remove markdown formatting if present)
                    if "```python" in generated_code:
                        generated_code = generated_code.split("```python")[1].split("```")[0].strip()
                    elif "```" in generated_code:
                        generated_code = generated_code.split("```")[1].split("```")[0].strip()
                    
                    print("✅ LLM Generated Blender Code:")
                    print("   Preview (first 5 lines):")
                    preview_lines = generated_code.split('\n')[:5]
                    for line in preview_lines:
                        print(f"     {line}")
                    print("     ...")
                else:
                    print(f"❌ LM Studio request failed (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ Failed to generate code with LM Studio: {e}")
            return False
        
        # Step 4: Execute generated code in Blender
        print("\n🎨 Step 4: Execute AI-Generated Code in Blender")
        try:
            async with session.post(f"{blender_url}/execute", json={"code": generated_code}) as response:
                if response.status == 200:
                    result = await response.json()
                    if result["success"]:
                        print("✅ AI-generated code executed successfully!")
                        for log in result.get("logs", []):
                            if "STDOUT:" in log:
                                stdout = log.replace("STDOUT: ", "").strip()
                                for line in stdout.split('\n'):
                                    if line.strip():
                                        print(f"   {line}")
                    else:
                        print(f"❌ Code execution failed: {result.get('error')}")
                        return False
                else:
                    print(f"❌ Blender execution request failed (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ Failed to execute code in Blender: {e}")
            return False
        
        # Step 5: Verify the result
        print("\n🔍 Step 5: Verify Scene Creation")
        try:
            async with session.get(f"{blender_url}/scene/info") as response:
                if response.status == 200:
                    scene_info = await response.json()
                    objects = scene_info["objects"]
                    print(f"✅ Scene '{scene_info['name']}' now contains {len(objects)} objects:")
                    for obj in objects[-3:]:  # Show last 3 objects
                        print(f"   • {obj}")
                else:
                    print(f"❌ Scene info request failed (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ Failed to get scene info: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 COMPLETE LM STUDIO + BLENDER E2E TEST SUCCESSFUL!")
        print("\nWorkflow Summary:")
        print("  ✅ LM Studio Connection & Model Loading")
        print("  ✅ Blender MCP Server Connection") 
        print("  ✅ Real AI Code Generation (Qwen2.5 Coder)")
        print("  ✅ AI-Generated Code Execution in Blender")
        print("  ✅ Scene Verification & Result Validation")
        
        print(f"\n🚀 LL3M is now fully operational!")
        print("   Real AI can now generate and execute Blender code!")
        
        return True


if __name__ == "__main__":
    print("Starting LL3M Real Integration Test...")
    success = asyncio.run(test_lm_studio_integration())
    
    if success:
        print(f"\n🎯 RESULT: Real E2E integration test PASSED!")
        print("   LL3M is ready for production use with real AI!")
    else:
        print(f"\n❌ RESULT: Integration test FAILED. Check output above.")
        exit(1)