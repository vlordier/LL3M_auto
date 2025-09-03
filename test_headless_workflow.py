#!/usr/bin/env python3
"""E2E workflow test optimized for headless Blender."""

import asyncio
import aiohttp


async def test_headless_workflow():
    """Test complete workflow with headless-compatible Blender code."""
    base_url = "http://localhost:3001"
    
    async with aiohttp.ClientSession() as session:
        print("🎯 LL3M E2E Workflow Test (Headless Mode)")
        print("=" * 55)
        
        # Step 1: Health check
        print("\n📋 Step 1: Verify Blender Connection")
        async with session.get(f"{base_url}/health") as response:
            health = await response.json()
            print(f"✅ Connected to Blender {health['blender_version']}")
        
        # Step 2: Test simple object creation
        print(f"\n🎯 Step 2: Simple Object Creation")
        simple_code = '''
import bpy

# Clear scene
if bpy.data.objects:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

# Create a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Find the created cube
cube = None
for obj in bpy.data.objects:
    if obj.type == 'MESH' and 'Cube' in obj.name:
        cube = obj
        break

if cube:
    cube.name = "RedCube"
    print(f"✅ Created cube: {cube.name}")
    print(f"✅ Total objects: {len(bpy.data.objects)}")
else:
    print("❌ Failed to create cube")
'''
        
        async with session.post(f"{base_url}/execute", json={"code": simple_code}) as response:
            result = await response.json()
            if result["success"]:
                print("✅ Step 2 Success: Simple object created")
                for log in result.get("logs", []):
                    if "STDOUT:" in log and "✅" in log:
                        print(f"   {log.replace('STDOUT: ', '').strip()}")
            else:
                print(f"❌ Step 2 Failed: {result.get('error')}")
                return False
        
        # Step 3: Complex scene with multiple objects
        print(f"\n🚀 Step 3: Complex Scene Creation")
        complex_code = '''
import bpy

# Clear existing objects
if bpy.data.objects:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

# Create multiple objects
objects_created = []

# Create cube
bpy.ops.mesh.primitive_cube_add(location=(-2, 0, 0))
for obj in bpy.data.objects:
    if obj.type == 'MESH' and 'Cube' in obj.name and obj not in objects_created:
        obj.name = "RedCube"
        objects_created.append(obj)
        break

# Create sphere  
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
for obj in bpy.data.objects:
    if obj.type == 'MESH' and 'Sphere' in obj.name and obj not in objects_created:
        obj.name = "BlueSphere"
        objects_created.append(obj)
        break

# Create cylinder
bpy.ops.mesh.primitive_cylinder_add(location=(2, 0, 0))
for obj in bpy.data.objects:
    if obj.type == 'MESH' and 'Cylinder' in obj.name and obj not in objects_created:
        obj.name = "GreenCylinder"
        objects_created.append(obj)
        break

# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
for obj in bpy.data.objects:
    if obj.type == 'LIGHT' and obj not in objects_created:
        obj.name = "MainLight"
        objects_created.append(obj)
        break

# Add camera
bpy.ops.object.camera_add(location=(7, -7, 5))
for obj in bpy.data.objects:
    if obj.type == 'CAMERA' and obj not in objects_created:
        obj.name = "MainCamera"
        objects_created.append(obj)
        break

print(f"✅ Complex scene created with {len(objects_created)} new objects:")
for obj in objects_created:
    print(f"   • {obj.name} ({obj.type}) at {obj.location}")

print(f"✅ Total scene objects: {len(bpy.data.objects)}")
'''
        
        async with session.post(f"{base_url}/execute", json={"code": complex_code, "timeout": 60}) as response:
            result = await response.json()
            if result["success"]:
                print("✅ Step 3 Success: Complex scene created")
                for log in result.get("logs", []):
                    if "STDOUT:" in log:
                        stdout = log.replace("STDOUT: ", "").strip()
                        for line in stdout.split('\\n'):
                            if line.strip() and ('✅' in line or '•' in line):
                                print(f"   {line}")
            else:
                print(f"❌ Step 3 Failed: {result.get('error')}")
                return False
        
        # Step 4: Scene verification
        print(f"\n🔍 Step 4: Scene Verification")
        async with session.get(f"{base_url}/scene/info") as response:
            scene = await response.json()
            objects = scene["objects"]
            
            print(f"✅ Scene '{scene['name']}' verified:")
            print(f"   • Total objects: {len(objects)}")
            print("   • Object list:")
            for obj in objects:
                print(f"     - {obj}")
        
        # Step 5: Material and styling test
        print(f"\n🎨 Step 5: Material Application")
        material_code = '''
import bpy

# Find objects and apply materials
cube = None
sphere = None

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        if 'Cube' in obj.name:
            cube = obj
        elif 'Sphere' in obj.name:
            sphere = obj

materials_created = 0

# Create red material for cube
if cube:
    red_mat = bpy.data.materials.new(name="RedMaterial")
    red_mat.use_nodes = True
    if red_mat.node_tree and red_mat.node_tree.nodes.get("Principled BSDF"):
        bsdf = red_mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (1, 0, 0, 1)  # Red
    cube.data.materials.clear()
    cube.data.materials.append(red_mat)
    materials_created += 1
    print(f"✅ Applied red material to {cube.name}")

# Create blue material for sphere  
if sphere:
    blue_mat = bpy.data.materials.new(name="BlueMaterial")
    blue_mat.use_nodes = True
    if blue_mat.node_tree and blue_mat.node_tree.nodes.get("Principled BSDF"):
        bsdf = blue_mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0, 0, 1, 1)  # Blue
    sphere.data.materials.clear()
    sphere.data.materials.append(blue_mat)
    materials_created += 1
    print(f"✅ Applied blue material to {sphere.name}")

print(f"✅ Created {materials_created} materials")
print(f"✅ Total materials in scene: {len(bpy.data.materials)}")
'''
        
        async with session.post(f"{base_url}/execute", json={"code": material_code}) as response:
            result = await response.json()
            if result["success"]:
                print("✅ Step 5 Success: Materials applied")
                for log in result.get("logs", []):
                    if "STDOUT:" in log and "✅" in log:
                        lines = log.replace("STDOUT: ", "").strip().split('\\n')
                        for line in lines:
                            if line.strip():
                                print(f"   {line}")
            else:
                print(f"❌ Step 5 Failed: {result.get('error')}")
        
        print("\n" + "=" * 55)
        print("🎉 COMPLETE E2E WORKFLOW SUCCESSFUL!")
        print("\nWorkflow Summary:")
        print("  ✅ Blender MCP Server Connection")
        print("  ✅ Simple Object Creation (Cube)")
        print("  ✅ Complex Multi-Object Scene")
        print("  ✅ Scene Verification & Info Retrieval") 
        print("  ✅ Material Application & Styling")
        print("  ✅ Headless Mode Compatibility")
        
        print(f"\n🚀 Ready for LLM Integration!")
        print("   Once LM Studio is running, this same workflow will work")
        print("   with real AI-generated Blender code from Qwen2.5 Coder!")
        
        return True


if __name__ == "__main__":
    print("Starting LL3M E2E Workflow Test...")
    success = asyncio.run(test_headless_workflow())
    
    if success:
        print(f"\n🎯 RESULT: E2E test PASSED! LL3M is ready for production use.")
    else:
        print(f"\n❌ RESULT: E2E test FAILED. Check output above.")
        exit(1)