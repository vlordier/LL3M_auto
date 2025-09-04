"""E2E tests for Blender MCP integration."""

import pytest


@pytest.mark.asyncio
async def test_blender_mcp_health_check(blender_mcp_server, http_session):
    """Test Blender MCP server health check."""
    async with http_session.get(f"{blender_mcp_server}/health") as response:
        assert response.status == 200
        data = await response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "blender_version" in data

        print(f"✓ Blender MCP server healthy, version: {data['blender_version']}")


@pytest.mark.asyncio
async def test_blender_code_execution(blender_mcp_server, http_session):
    """Test basic code execution in Blender."""
    code = """
# Clear scene
import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Add a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Get the created object from the scene
cube = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and 'Cube' in obj.name:
        cube = obj
        break

if cube:
    print(f"Created object: {cube.name}")
else:
    print("No cube found")

print(f"Scene has {len(bpy.context.scene.objects)} objects")
"""

    payload = {"code": code, "timeout": 30}

    async with http_session.post(
        f"{blender_mcp_server}/execute", json=payload
    ) as response:
        assert response.status == 200
        data = await response.json()

        assert data["success"] is True
        assert "result" in data
        assert "logs" in data

        # Check that output contains expected information
        logs = " ".join(data["logs"])
        assert "Created object:" in logs
        assert "Scene has" in logs
        assert "objects" in logs

        print("✓ Code executed successfully")
        print(f"  Result: {data['result']}")
        for log in data["logs"]:
            print(f"  Log: {log}")


@pytest.mark.asyncio
async def test_blender_scene_info(blender_mcp_server, http_session):
    """Test getting scene information from Blender."""
    # First, set up a simple scene
    setup_code = """
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Add objects
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
# Find and rename the cube
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and 'Cube' in obj.name:
        obj.name = "TestCube"
        break

bpy.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))
# Find and rename the sphere
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and 'Sphere' in obj.name:
        obj.name = "TestSphere"
        break
"""

    # Execute setup code
    async with http_session.post(
        f"{blender_mcp_server}/execute", json={"code": setup_code}
    ) as response:
        assert response.status == 200
        setup_result = await response.json()
        assert setup_result["success"] is True

    # Get scene info
    async with http_session.get(f"{blender_mcp_server}/scene/info") as response:
        assert response.status == 200
        scene_info = await response.json()

        assert "name" in scene_info
        assert "objects" in scene_info
        assert "frame_start" in scene_info
        assert "frame_end" in scene_info

        # Check that our test objects are present
        objects = scene_info["objects"]
        assert "TestCube" in objects
        assert "TestSphere" in objects

        print("✓ Scene info retrieved successfully")
        print(f"  Scene name: {scene_info['name']}")
        print(f"  Objects: {objects}")


@pytest.mark.asyncio
async def test_blender_error_handling(blender_mcp_server, http_session):
    """Test error handling in Blender code execution."""
    # Code with intentional error
    bad_code = """
import bpy

# This should cause an error
bpy.ops.nonexistent.operation()
"""

    payload = {"code": bad_code}

    async with http_session.post(
        f"{blender_mcp_server}/execute", json=payload
    ) as response:
        assert response.status == 200
        data = await response.json()

        assert data["success"] is False
        assert "error" in data
        assert data["error"] is not None

        # Error should mention the invalid operation
        error_msg = data["error"].lower()
        assert any(
            keyword in error_msg for keyword in ["attribute", "nonexistent", "error"]
        )

        print(f"✓ Error handled correctly: {data['error']}")


@pytest.mark.asyncio
async def test_blender_complex_scene_creation(blender_mcp_server, http_session):
    """Test creating a complex scene with multiple objects."""
    complex_code = """
import bpy
import bmesh

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
# Find the cube
cube = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and 'Cube' in obj.name:
        cube = obj
        cube.name = "RedCube"
        break

# Add material to cube
material = bpy.data.materials.new(name="RedMaterial")
material.use_nodes = True
# Red color
material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 0, 0, 1)
cube.data.materials.append(material)

# Create sphere
bpy.ops.mesh.primitive_uv_sphere_add(location=(3, 0, 0))
# Find the sphere
sphere = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and 'Sphere' in obj.name:
        sphere = obj
        sphere.name = "BlueSphere"
        break

# Add blue material to sphere
blue_material = bpy.data.materials.new(name="BlueMaterial")
blue_material.use_nodes = True
# Blue color
blue_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 0, 1, 1)
sphere.data.materials.append(blue_material)

# Add cylinder
bpy.ops.mesh.primitive_cylinder_add(location=(-3, 0, 0))
# Find the cylinder
cylinder = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and 'Cylinder' in obj.name:
        cylinder = obj
        cylinder.name = "GreenCylinder"
        break

# Add green material to cylinder
green_material = bpy.data.materials.new(name="GreenMaterial")
green_material.use_nodes = True
# Green color
green_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 1, 0, 1)
cylinder.data.materials.append(green_material)

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
# Find the light
sun = None
for obj in bpy.context.scene.objects:
    if obj.type == 'LIGHT':
        sun = obj
        sun.name = "SunLight"
        sun.data.energy = 3
        break

# Add camera
bpy.ops.object.camera_add(location=(7, -7, 5))
# Find the camera
camera = None
for obj in bpy.context.scene.objects:
    if obj.type == 'CAMERA':
        camera = obj
        camera.name = "MainCamera"
        break

# Point camera at origin
if camera:
    import mathutils
    direction = mathutils.Vector((0, 0, 0)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

print(f"Scene created with {len(bpy.context.scene.objects)} objects")
for obj in bpy.context.scene.objects:
    print(f"- {obj.name} at {obj.location}")
"""

    payload = {"code": complex_code, "timeout": 60}

    async with http_session.post(
        f"{blender_mcp_server}/execute", json=payload
    ) as response:
        assert response.status == 200
        data = await response.json()

        assert data["success"] is True

        # Check that all expected objects were created
        logs = " ".join(data["logs"])
        expected_objects = [
            "RedCube",
            "BlueSphere",
            "GreenCylinder",
            "SunLight",
            "MainCamera",
        ]
        for obj_name in expected_objects:
            assert obj_name in logs

        print("✓ Complex scene created successfully")
        print(f"  Result: {data['result']}")


@pytest.mark.asyncio
async def test_blender_save_functionality(
    blender_mcp_server, http_session, temp_output_dir
):
    """Test saving Blender files."""
    # Create a simple scene first
    setup_code = """
import bpy

# Clear and create simple scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)
bpy.ops.mesh.primitive_cube_add()
print("Scene prepared for saving")
"""

    # Execute setup
    async with http_session.post(
        f"{blender_mcp_server}/execute", json={"code": setup_code}
    ) as response:
        assert response.status == 200

    # Save the file
    save_path = temp_output_dir / "test_scene.blend"

    async with http_session.post(
        f"{blender_mcp_server}/scene/save", json={"filepath": str(save_path)}
    ) as response:
        assert response.status == 200
        data = await response.json()

        assert data["success"] is True
        assert data["filepath"] == str(save_path)

        # Verify file was actually created (file size > 0)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        print(f"✓ Scene saved successfully to {save_path}")
        print(f"  File size: {save_path.stat().st_size} bytes")
