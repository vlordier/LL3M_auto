import bpy

# Clear the default scene completely
def clear_scene():
    print("Clearing the scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    print("Scene cleared.")

# Create a detailed bunny shape
def create_bunny():
    print("Creating bunny...")
    # Add an ico-sphere to represent the body of the bunny
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=2)
    bunny_body = bpy.context.object
    bunny_body.name = "Bunny_Body"

    # Create ears
    bpy.ops.mesh.primitive_cone_add(radius1=0.5, radius2=0.5, depth=1, location=(3, 0, 4))
    ear = bpy.context.object
    ear.name = "Bunny_Ear"
    ear.scale[2] = 0.5

    bpy.ops.mesh.primitive_cone_add(radius1=0.5, radius2=0.5, depth=1, location=(-3, 0, 4))
    ear_right = bpy.context.object
    ear_right.name = "Bunny_Ear_Right"
    ear_right.scale[2] = 0.5
    ear_right.rotation_euler[1] = -1.57

    # Create tail
    bpy.ops.mesh.primitive_cone_add(radius1=0.5, radius2=0, depth=1, location=(0, 3, -2))
    tail = bpy.context.object
    tail.name = "Bunny_Tail"
    tail.rotation_euler[0] = 1.57

    # Join all parts into one object
    bpy.context.view_layer.objects.active = bunny_body
    bpy.ops.object.select_all(action='DESELECT')
    bunny_body.select_set(True)
    ear.select_set(True)
    ear_right.select_set(True)
    tail.select_set(True)

    bpy.ops.object.join()
    print("Bunny created.")

# Apply a glittery pink material with proper nodes and textures
def apply_glittery_material(obj):
    print("Applying glittery pink material...")
    # Create a new material
    mat = bpy.data.materials.new(name="Glittery_Pink")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create Principled BSDF Node
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    # Create Emission Node for glitter effect
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[1].default_value = 5.0  # Strength
    emission.location = (-200, 0)

    # Create Texture Coordinate and Noise Textures
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    noise1 = nodes.new(type='ShaderNodeTexNoise')
    noise2 = nodes.new(type='ShaderNodeTexNoise')

    tex_coord.location = (-800, 200)
    noise1.location = (-600, 200)
    noise2.location = (-400, 200)

    # Connect nodes
    mat.node_tree.links.new(tex_coord.outputs['Object'], noise1.inputs['Vector'])
    mat.node_tree.links.new(noise1.outputs['Fac'], emission.inputs['Color'])
    mat.node_tree.links.new(emission.outputs['Emission'], bsdf.inputs['Base Color'])

    # Create Material Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Set material color to pink
    bsdf.inputs[0].default_value = (1.0, 0.75, 0.8, 1)  # Pink color

    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print("Glittery pink material applied.")

# Set up professional lighting
def setup_lighting():
    print("Setting up lighting...")
    # Add a main key light
    bpy.ops.object.light_add(type='SUN', location=(5, 0, 10))
    key_light = bpy.context.object
    key_light.name = "Key_Light"
    key_light.data.energy = 1500

    # Add a fill light
    bpy.ops.object.light_add(type='SUN', location=(-5, 0, -10))
    fill_light = bpy.context.object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 800
    fill_light.rotation_euler[2] = 3.14

    # Add a rim light
    bpy.ops.object.light_add(type='SUN', location=(0, 5, -10))
    rim_light = bpy.context.object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 500
    print("Lighting set up.")

# Position the camera for the best view of the bunny
def position_camera():
    print("Positioning camera...")
    bpy.ops.object.camera_add(location=(0, 8, 6))
    camera = bpy.context.object
    camera.name = "Bunny_Camera"
    camera.rotation_euler[0] = 1.2
    bpy.context.scene.camera = camera
    print("Camera positioned.")

# Main function to run the script
def main():
    clear_scene()
    create_bunny()
    bunny = bpy.data.objects['Bunny_Body']
    apply_glittery_material(bunny)
    setup_lighting()
    position_camera()
    print("Bunny creation and setup complete.")

if __name__ == "__main__":
    main()