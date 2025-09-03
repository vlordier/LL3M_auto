#!/usr/bin/env python3
"""Direct test of Blender MCP server without pytest fixtures."""

import asyncio
import json

import aiohttp


async def test_blender_direct():
    """Test Blender MCP server directly."""
    base_url = "http://localhost:3001"
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Health check
        print("🔍 Testing health check...")
        try:
            async with session.get(f"{base_url}/health") as response:
                assert response.status == 200
                data = await response.json()
                print(f"✅ Health check passed: {data}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
        
        # Test 2: Simple code execution
        print("\n🔍 Testing code execution...")
        test_code = '''
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Add a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

print(f"Active object: {bpy.context.active_object.name}")
print(f"Scene has {len(bpy.context.scene.objects)} objects")
'''
        
        try:
            async with session.post(
                f"{base_url}/execute",
                json={"code": test_code}
            ) as response:
                assert response.status == 200
                data = await response.json()
                
                if data["success"]:
                    print("✅ Code execution successful")
                    print(f"  Result: {data.get('result', 'No output')}")
                    for log in data.get("logs", []):
                        print(f"  Log: {log}")
                else:
                    print(f"❌ Code execution failed: {data.get('error')}")
                    return False
        except Exception as e:
            print(f"❌ Code execution test failed: {e}")
            return False
        
        # Test 3: Scene info
        print("\n🔍 Testing scene info...")
        try:
            async with session.get(f"{base_url}/scene/info") as response:
                assert response.status == 200
                data = await response.json()
                print("✅ Scene info retrieved")
                print(f"  Scene: {data.get('name')}")
                print(f"  Objects: {data.get('objects')}")
        except Exception as e:
            print(f"❌ Scene info test failed: {e}")
            return False
        
        # Test 4: Complex scene creation
        print("\n🔍 Testing complex scene creation...")
        complex_code = '''
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create multiple objects
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "RedCube"

bpy.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))
sphere = bpy.context.active_object  
sphere.name = "BlueSphere"

bpy.ops.mesh.primitive_cylinder_add(location=(-2, 0, 0))
cylinder = bpy.context.active_object
cylinder.name = "GreenCylinder"

print(f"Created scene with {len(bpy.context.scene.objects)} objects:")
for obj in bpy.context.scene.objects:
    print(f"- {obj.name}")
'''
        
        try:
            async with session.post(
                f"{base_url}/execute", 
                json={"code": complex_code}
            ) as response:
                assert response.status == 200
                data = await response.json()
                
                if data["success"]:
                    print("✅ Complex scene created successfully")
                    for log in data.get("logs", []):
                        print(f"  {log}")
                else:
                    print(f"❌ Complex scene creation failed: {data.get('error')}")
                    return False
        except Exception as e:
            print(f"❌ Complex scene test failed: {e}")
            return False
            
    print("\n🎉 All Blender MCP tests passed!")
    return True


if __name__ == "__main__":
    print("🧪 Testing Blender MCP Integration\n")
    
    success = asyncio.run(test_blender_direct())
    
    if success:
        print("\n✅ Blender MCP integration is working correctly!")
    else:
        print("\n❌ Some tests failed.")
        exit(1)