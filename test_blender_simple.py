#!/usr/bin/env python3
"""Simple Blender MCP test with headless-compatible code."""

import asyncio
import aiohttp


async def test_simple_blender():
    """Test with headless-compatible Blender code."""
    base_url = "http://localhost:3001"
    
    async with aiohttp.ClientSession() as session:
        print("🔍 Testing health check...")
        async with session.get(f"{base_url}/health") as response:
            data = await response.json()
            print(f"✅ Blender version: {data['blender_version']}")
        
        print("\n🔍 Testing simple code execution...")
        simple_code = '''
import bpy

# Clear existing objects
if bpy.context.scene.objects:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

# Add a cube
bpy.ops.mesh.primitive_cube_add()

# Get the cube (it should be the active object)
cube = None
for obj in bpy.data.objects:
    if obj.type == 'MESH' and 'Cube' in obj.name:
        cube = obj
        break

if cube:
    print(f"✅ Created cube: {cube.name}")
    print(f"✅ Scene now has {len(bpy.data.objects)} objects")
else:
    print("❌ Failed to create cube")
'''
        
        async with session.post(f"{base_url}/execute", json={"code": simple_code}) as response:
            data = await response.json()
            if data["success"]:
                print("✅ Simple execution successful:")
                for log in data.get("logs", []):
                    print(f"  {log}")
            else:
                print(f"❌ Failed: {data.get('error')}")
                return False
        
        print("\n🔍 Testing scene info...")
        async with session.get(f"{base_url}/scene/info") as response:
            scene_info = await response.json()
            print(f"✅ Scene '{scene_info['name']}' has objects: {scene_info['objects']}")
        
        print("\n🔍 Testing error handling...")
        bad_code = '''
import bpy
# This should fail gracefully
bpy.ops.nonexistent.operation()
'''
        
        async with session.post(f"{base_url}/execute", json={"code": bad_code}) as response:
            data = await response.json()
            if not data["success"] and "error" in data:
                print(f"✅ Error handling works: {data['error'][:50]}...")
            else:
                print("❌ Error handling failed")
                return False
                
    print("\n🎉 All basic Blender tests passed!")
    return True


if __name__ == "__main__":
    print("🧪 Simple Blender MCP Test\n")
    success = asyncio.run(test_simple_blender())
    if not success:
        exit(1)