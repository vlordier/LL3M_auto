"""Code templates for common Blender operations."""

# Base templates for common 3D operations
GEOMETRY_TEMPLATES: dict[str, str] = {
    "cube": """
# Add a cube
bpy.ops.mesh.primitive_cube_add(location=({x}, {y}, {z}))
cube = bpy.context.object
cube.name = "{name}"
""",
    "sphere": """
# Add a UV sphere
bpy.ops.mesh.primitive_uv_sphere_add(location=({x}, {y}, {z}))
sphere = bpy.context.object
sphere.name = "{name}"
""",
    "cylinder": """
# Add a cylinder
bpy.ops.mesh.primitive_cylinder_add(location=({x}, {y}, {z}))
cylinder = bpy.context.object
cylinder.name = "{name}"
""",
    "plane": """
# Add a plane
bpy.ops.mesh.primitive_plane_add(location=({x}, {y}, {z}))
plane = bpy.context.object
plane.name = "{name}"
""",
}

MATERIAL_TEMPLATES: dict[str, str] = {
    "basic": """
# Create basic material
material = bpy.data.materials.new(name="{name}")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = ({r}, {g}, {b}, 1.0)

# Assign material to object
{object_name}.data.materials.append(material)
""",
    "metallic": """
# Create metallic material
material = bpy.data.materials.new(name="{name}")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = ({r}, {g}, {b}, 1.0)
bsdf.inputs['Metallic'].default_value = {metallic}
bsdf.inputs['Roughness'].default_value = {roughness}

# Assign material to object
{object_name}.data.materials.append(material)
""",
    "emission": """
# Create emission material
material = bpy.data.materials.new(name="{name}")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Emission'].default_value = ({r}, {g}, {b}, 1.0)
bsdf.inputs['Emission Strength'].default_value = {strength}

# Assign material to object
{object_name}.data.materials.append(material)
""",
}

LIGHTING_TEMPLATES: dict[str, str] = {
    "sun": """
# Add sun light
bpy.ops.object.light_add(type='SUN', location=({x}, {y}, {z}))
light = bpy.context.object
light.data.energy = {energy}
light.rotation_euler = ({rx}, {ry}, {rz})
""",
    "point": """
# Add point light
bpy.ops.object.light_add(type='POINT', location=({x}, {y}, {z}))
light = bpy.context.object
light.data.energy = {energy}
""",
    "area": """
# Add area light
bpy.ops.object.light_add(type='AREA', location=({x}, {y}, {z}))
light = bpy.context.object
light.data.energy = {energy}
light.data.size = {size}
""",
}

SCENE_TEMPLATES: dict[str, str] = {
    "clear_scene": """
# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)
""",
    "camera_setup": """
# Set up camera
bpy.ops.object.camera_add(location=({x}, {y}, {z}))
camera = bpy.context.object
camera.rotation_euler = ({rx}, {ry}, {rz})
bpy.context.scene.camera = camera
""",
    "render_settings": """
# Configure render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = {samples}
bpy.context.scene.render.resolution_x = {width}
bpy.context.scene.render.resolution_y = {height}
""",
}

MODIFIER_TEMPLATES: dict[str, str] = {
    "subdivision": """
# Add subdivision surface modifier
modifier = {object_name}.modifiers.new(name="Subsurf", type='SUBSURF')
modifier.levels = {levels}
""",
    "bevel": """
# Add bevel modifier
modifier = {object_name}.modifiers.new(name="Bevel", type='BEVEL')
modifier.width = {width}
modifier.segments = {segments}
""",
    "array": """
# Add array modifier
modifier = {object_name}.modifiers.new(name="Array", type='ARRAY')
modifier.count = {count}
modifier.relative_offset_displace[0] = {offset_x}
modifier.relative_offset_displace[1] = {offset_y}
modifier.relative_offset_displace[2] = {offset_z}
""",
}
