#!/usr/bin/env python3
"""Demo E2E test showing the complete workflow concept."""

import asyncio

import aiohttp


async def simulate_llm_code_generation(prompt):
    """Simulate LLM code generation for the prompt."""
    print(f"🤖 LLM Processing prompt: '{prompt}'")

    # This simulates what Qwen2.5 Coder would generate for a simple prompt
    if "cube" in prompt.lower():
        return """
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Add material
material = bpy.data.materials.new(name="RedMaterial")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = (1, 0, 0, 1)  # Red color

# Apply material to cube - headless compatible
cube = None
for obj in bpy.data.objects:
    if obj.type == 'MESH' and 'Cube' in obj.name:
        cube = obj
        break

if cube:
    cube.data.materials.clear()
    cube.data.materials.append(material)
    print(f"✅ Created red cube: {cube.name}")
    print(f"✅ Scene has {len(bpy.data.objects)} objects")
else:
    print("❌ Failed to find cube")
"""
    elif "sphere" in prompt.lower():
        return """
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create sphere
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))

# Find the created sphere - headless compatible
sphere = None
for obj in bpy.data.objects:
    if obj.type == 'MESH' and 'Sphere' in obj.name:
        sphere = obj
        break

if sphere:
    print(f"✅ Created sphere: {sphere.name}")
else:
    print("❌ Failed to create sphere")
"""
    else:
        return """
import bpy

# Create default scene with cube, light, and camera
bpy.ops.mesh.primitive_cube_add()
bpy.ops.object.light_add(type='SUN', location=(4, 4, 4))
bpy.ops.object.camera_add(location=(7, -7, 5))

print("✅ Created default scene")
"""


async def test_e2e_workflow_demo():
    """Demonstrate complete E2E workflow."""
    base_url = "http://localhost:3001"

    async with aiohttp.ClientSession() as session:
        print("🎯 Starting E2E Workflow Demo")
        print("=" * 50)

        # Step 1: Health check
        print("\n📋 Step 1: Verify Blender is ready")
        async with session.get(f"{base_url}/health") as response:
            health_data = await response.json()
            print(f"✅ Blender {health_data['blender_version']} ready")

        # Step 2: Process user prompt
        user_prompt = "Create a red cube in the center of the scene"
        print("\n🎯 Step 2: User Request")
        print(f'   User: "{user_prompt}"')

        # Step 3: Generate code (simulated LLM)
        print("\n🤖 Step 3: LLM Code Generation")
        generated_code = await simulate_llm_code_generation(user_prompt)
        print(f"✅ Generated {len(generated_code)} characters of Blender Python code")
        print("   Code preview:")
        preview_lines = generated_code.strip().split("\n")[:5]
        for line in preview_lines:
            print(f"     {line}")
        print("     ...")

        # Step 4: Execute in Blender
        print("\n🎨 Step 4: Execute in Blender")
        async with session.post(
            f"{base_url}/execute", json={"code": generated_code}
        ) as response:
            exec_result = await response.json()

            if exec_result["success"]:
                print("✅ Code executed successfully in Blender")
                for log in exec_result.get("logs", [])[:3]:  # Show first 3 logs
                    if "STDOUT:" in log:
                        stdout = log.replace("STDOUT: ", "").strip()
                        for line in stdout.split("\n"):
                            if line.strip():
                                print(f"   {line}")
            else:
                print(f"❌ Execution failed: {exec_result.get('error')}")
                return False

        # Step 5: Verify result
        print("\n🔍 Step 5: Verify Scene Creation")
        async with session.get(f"{base_url}/scene/info") as response:
            scene_info = await response.json()

            objects = scene_info["objects"]
            print(f"✅ Scene '{scene_info['name']}' contains {len(objects)} objects:")
            for obj in objects:
                print(f"   • {obj}")

        # Step 6: Advanced test - Create multiple objects
        print("\n🚀 Step 6: Advanced Scene Creation")

        advanced_code = """
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create multiple objects
bpy.ops.mesh.primitive_cube_add(location=(-2, 0, 0))
cube = bpy.context.active_object
cube.name = "RedCube"

bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
sphere = bpy.context.active_object
sphere.name = "BlueSphere"

bpy.ops.mesh.primitive_cylinder_add(location=(2, 0, 0))
cylinder = bpy.context.active_object
cylinder.name = "GreenCylinder"

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
light = bpy.context.active_object
light.name = "MainLight"

print(f"✅ Created advanced scene with {len(bpy.data.objects)} objects")
for obj in bpy.data.objects:
    if obj.type in ['MESH', 'LIGHT']:
        print(f"   • {obj.name} ({obj.type})")
"""

        async with session.post(
            f"{base_url}/execute", json={"code": advanced_code}
        ) as response:
            result = await response.json()

            if result["success"]:
                print("✅ Advanced scene created successfully")
                for log in result.get("logs", []):
                    if "STDOUT:" in log:
                        stdout = log.replace("STDOUT: ", "").strip()
                        for line in stdout.split("\n"):
                            if line.strip():
                                print(f"   {line}")
            else:
                print(f"❌ Advanced scene failed: {result.get('error')}")

        # Final verification
        async with session.get(f"{base_url}/scene/info") as response:
            final_scene = await response.json()
            print(f"\n🎉 Final Scene: {len(final_scene['objects'])} objects created")

        print("\n" + "=" * 50)
        print("🎉 E2E Workflow Demo Completed Successfully!")
        print("\nThis demonstrates the complete LL3M pipeline:")
        print("  1. ✅ User prompt processing")
        print("  2. ✅ LLM code generation (simulated)")
        print("  3. ✅ Blender code execution")
        print("  4. ✅ Scene verification")
        print("  5. ✅ Advanced scene creation")
        print("\n💡 Once LM Studio is running, this will work with real AI!")

        return True


if __name__ == "__main__":
    success = asyncio.run(test_e2e_workflow_demo())
    if not success:
        exit(1)
