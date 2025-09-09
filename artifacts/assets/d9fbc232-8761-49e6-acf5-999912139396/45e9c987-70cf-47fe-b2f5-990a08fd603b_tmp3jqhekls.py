import sys

try:
    import bpy
    import bmesh
    import mathutils
    from mathutils import Vector, Matrix, Euler
    import math
except ModuleNotFoundError:
    print("This script requires Blender's Python environment (bpy not found).")
    sys.exit(0)


def clear_scene():
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception:
            pass
    except Exception as e:
        print(f"Error clearing scene: {e}")


def create_geometry():
    try:
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), rotation=(0, 0, 0))
        obj = bpy.context.active_object
        if obj:
            obj.name = "cube_geometry"
            obj.scale = (1, 1, 1)
            obj.dimensions = (2, 2, 2)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
    except Exception as e:
        print(f"Error creating geometry: {e}")


def create_materials():
    try:
        mat = bpy.data.materials.new(name="cube_material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (300, 0)
        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.location = (0, 0)
        links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])
        bsdf_node.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        bsdf_node.inputs["Roughness"].default_value = 0.5
        bsdf_node.inputs["Metallic"].default_value = 0.0
        obj = bpy.data.objects.get("cube_geometry")
        if obj is not None:
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
    except Exception as e:
        print(f"Error creating materials: {e}")


def setup_lighting():
    try:
        bpy.ops.object.light_add(type="POINT", location=(4, 4, 4))
        light = bpy.context.active_object
        if light:
            light.name = "scene_light"
            light.data.energy = 1000
            light.data.color = (1, 1, 1)
            bpy.context.view_layer.objects.active = light
            light.select_set(True)
    except Exception as e:
        print(f"Error setting up lighting: {e}")


def setup_scene():
    try:
        bpy.ops.object.camera_add(location=(5, -5, 5))
        cam = bpy.context.active_object
        if cam:
            cam.name = "scene_camera"
            cam.rotation_euler = Euler((math.radians(60), math.radians(0), math.radians(45)), "XYZ")
            cam.data.lens = 50
            bpy.context.scene.camera = cam
            scene = bpy.context.scene
            scene.render.engine = "CYCLES"
            scene.render.resolution_x = 1920
            scene.render.resolution_y = 1080
            if hasattr(scene, "cycles"):
                scene.cycles.samples = 64
    except Exception as e:
        print(f"Error setting up scene: {e}")


def main():
    clear_scene()
    create_geometry()
    create_materials()
    setup_lighting()
    setup_scene()


if __name__ == "__main__":
    main()
    x, y, z = 0.0, 0.0, 0.0
    name = "custom_cube"
    try:
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z))
        cube = bpy.context.object
        if cube:
            cube.name = name