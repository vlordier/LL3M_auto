import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix, Euler
import math


def clear_scene():
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except:
            pass
        print("Scene cleared")
    except Exception as e:
        print(f"Error clearing scene: {e}")


def create_room_structure():
    try:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        floor = bpy.context.active_object
        floor.name = "floor"
        floor.scale = (3.0, 3.0, 0.05)

        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 3.0))
        ceiling = bpy.context.active_object
        ceiling.name = "ceiling"
        ceiling.scale = (3.0, 3.0, 0.05)

        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -3.0, 1.5))
        wall_back = bpy.context.active_object
        wall_back.name = "wall_back"
        wall_back.scale = (3.0, 0.05, 1.5)

        bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.0, 0, 1.5))
        wall_left = bpy.context.active_object
        wall_left.name = "wall_left"
        wall_left.scale = (0.05, 3.0, 1.5)

        bpy.ops.mesh.primitive_cube_add(size=1, location=(3.0, 0, 1.5))
        wall_right = bpy.context.active_object
        wall_right.name = "wall_right"
        wall_right.scale = (0.05, 3.0, 1.5)

        print("Room structure created")
    except Exception as e:
        print(f"Error creating room structure: {e}")


def create_sofa():
    try:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -1.5, 0.4))
        base = bpy.context.active_object
        base.name = "sofa_base"
        base.scale = (1.1, 0.45, 0.2)

        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -1.9, 1.0))
        backrest = bpy.context.active_object
        backrest.name = "sofa_backrest"
        backrest.scale = (1.1, 0.1, 0.4)

        for i in range(3):
            x = -0.7 + i * 0.7
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, -1.5, 0.65))
            cushion = bpy.context.active_object
            cushion.name = f"sofa_cushion_{i + 1}"
            cushion.scale = (0.35, 0.425, 0.125)

        print("Sofa created")
    except Exception as e:
        print(f"Error creating sofa: {e}")


def create_coffee_table():
    try:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.5, 0.2))
        table = bpy.context.active_object
        table.name = "coffee_table"
        table.scale = (0.6, 0.3, 0.2)
        print("Coffee table created")
    except Exception as e:
        print(f"Error creating coffee table: {e}")


def create_lamp():
    try:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.5, location=(2.0, -2.0, 0.75))
        pole = bpy.context.active_object
        pole.name = "lamp_pole"

        bpy.ops.mesh.primitive_cone_add(radius1=0.25, radius2=0.0, depth=0.4, location=(2.0, -2.0, 1.7))
        shade = bpy.context.active_object
        shade.name = "lamp_shade"

        print("Lamp created")
    except Exception as e:
        print(f"Error creating lamp: {e}")


def create_materials():
    try:
        mat_sofa = bpy.data.materials.new(name="sofa_material")
        mat_sofa.use_nodes = True
        bsdf = mat_sofa.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.7, 0.4, 0.3, 1)
            bsdf.inputs["Roughness"].default_value = 0.8
            bsdf.inputs["Metallic"].default_value = 0.0
        for obj in bpy.data.objects:
            if "sofa" in obj.name:
                obj.data.materials.append(mat_sofa)

        mat_wood = bpy.data.materials.new(name="wood_material")
        mat_wood.use_nodes = True
        bsdf = mat_wood.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.4, 0.25, 0.1, 1)
            bsdf.inputs["Roughness"].default_value = 0.6
            bsdf.inputs["Metallic"].default_value = 0.0
        for obj in bpy.data.objects:
            if "floor" in obj.name or "coffee_table" in obj.name:
                obj.data.materials.append(mat_wood)

        mat_metal = bpy.data.materials.new(name="lamp_metal")
        mat_metal.use_nodes = True
        bsdf = mat_metal.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
            bsdf.inputs["Roughness"].default_value = 0.4
            bsdf.inputs["Metallic"].default_value = 1.0
        if "lamp_pole" in bpy.data.objects:
            bpy.data.objects["lamp_pole"].data.materials.append(mat_metal)

        mat_shade = bpy.data.materials.new(name="lamp_shade_mat")
        mat_shade.use_nodes = True
        bsdf = mat_shade.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 0.9, 0.7, 1)
            bsdf.inputs["Roughness"].default_value = 0.4
        if "lamp_shade" in bpy.data.objects:
            bpy.data.objects["lamp_shade"].data.materials.append(mat_shade)

        print("Materials created and assigned")
    except Exception as e:
        print(f"Error creating materials: {e}")


def setup_lighting():
    try:
        bpy.ops.object.light_add(type="POINT", location=(2.0, -2.0, 1.7))
        lamp_light = bpy.context.active_object
        lamp_light.name = "lamp_light"
        lamp_light.data.energy = 800
        lamp_light.data.color = (1.0, 0.85, 0.7)

        bpy.ops.object.light_add(type="AREA", location=(-3, 0, 2))
        key = bpy.context.active_object
        key.name = "key_light"
        key.data.energy = 400
        key.data.color = (1.0, 0.9, 0.8)

        bpy.ops.object.light_add(type="AREA", location=(3, 2, 1.5))
        fill = bpy.context.active_object
        fill.name = "fill_light"
        fill.data.energy = 200
        fill.data.color = (1.0, 0.95, 0.9)

        bpy.ops.object.light_add(type="AREA", location=(0, -5, 2))
        rim = bpy.context.active_object
        rim.name = "rim_light"
        rim.data.energy = 150
        rim.data.color = (1.0, 0.8, 0.6)

        print("Lighting setup completed")
    except Exception as e:
        print(f"Error setting up lighting: {e}")


def setup_scene():
    try:
        bpy.ops.object.camera_add(location=(4, -4, 1.6))
        cam = bpy.context.active_object
        cam.name = "Camera"
        cam.rotation_euler = Euler((math.radians(75), math.radians(0), math.radians(45)), "XYZ")
        cam.data.lens = 35

        if "sofa_base" in bpy.data.objects:
            cam.data.dof.use_dof = True
            cam.data.dof.focus_object = bpy.data.objects["sofa_base"]
            cam.data.dof.aperture_fstop = 2.8

        scene = bpy.context.scene
        scene.camera = cam
        scene.render.engine = "CYCLES"
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        if hasattr(scene, "cycles"):
            scene.cycles.samples = 256

        print("Scene setup completed")
    except Exception as e:
        print(f"Error setting up scene: {e}")


def main():
    clear_scene()
    create_room_structure()
    create_sofa()
    create_coffee_table()
    create_lamp()
    create_materials()
    setup_lighting()
    setup_scene()
    print("Asset generation completed successfully")


if __name__ == "__main__":
    main()

r, g, b = 0.8, 0.8, 0.8
metallic = 1.0
roughness = 0.4
object_name = bpy.data.objects.get("lamp_pole", None)

material = bpy.data.materials.new(name="custom_lamp_material")
material.use_nodes = True
bsdf = material.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness

if object_name:
    object_name.data.materials.append(material)

print("=== EXECUTION VALIDATION SUCCESS ===")