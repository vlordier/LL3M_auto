import math
import sys

try:
    import bmesh
    import bpy
    import mathutils
    from mathutils import Euler, Vector
except ModuleNotFoundError:
    class Dummy:
        def __getattr__(self, name):
            return self
        def __call__(self, *args, **kwargs):
            return None
    bpy = Dummy()
    bmesh = Dummy()
    mathutils = Dummy()
    Vector = tuple
    Euler = lambda *a, **k: (0,0,0)


def clear_scene():
    try:
        if hasattr(bpy, "ops"):
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete(use_global=False)
            if hasattr(bpy.ops.outliner, "orphans_purge"):
                bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        print("Scene cleared successfully")
    except Exception as e:
        print(f"Error clearing scene: {e}")


def create_room_structure():
    try:
        if hasattr(bpy, "ops"):
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
            floor = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
            if floor:
                floor.name = "floor"
                floor.scale = (3, 3, 0.05)
                floor.location.z = 0
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 3))
            ceiling = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
            if ceiling:
                ceiling.name = "ceiling"
                ceiling.scale = (3, 3, 0.05)
            wall_positions = [
                ((0, -3, 1.5), (3, 0.05, 1.5)),
                ((0, 3, 1.5), (3, 0.05, 1.5)),
                ((-3, 0, 1.5), (0.05, 3, 1.5)),
                ((3, 0, 1.5), (0.05, 3, 1.5)),
            ]
            for i, (loc, scale) in enumerate(wall_positions):
                bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
                wall = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
                if wall:
                    wall.name = f"wall_{i + 1}"
                    wall.scale = scale
        print("Room structure created")
    except Exception as e:
        print(f"Error creating room structure: {e}")


def create_sofa():
    try:
        if hasattr(bpy, "ops"):
            bpy.ops.mesh.primitive_cube_add(location=(0, -1.5, 0.2))
            base = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
            if base:
                base.name = "sofa_base"
                base.scale = (1.1, 0.45, 0.2)
            bpy.ops.mesh.primitive_cube_add(location=(0, -1.95, 0.8))
            backrest = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
            if backrest:
                backrest.name = "sofa_backrest"
                backrest.scale = (1.1, 0.1, 0.4)
            for i in range(3):
                x_offset = -0.7 + i * 0.7
                bpy.ops.mesh.primitive_cube_add(location=(x_offset, -1.5, 0.45))
                cushion = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
                if cushion:
                    cushion.name = f"sofa_cushion_{i + 1}"
                    cushion.scale = (0.35, 0.45, 0.25)
        print("Sofa created")
    except Exception as e:
        print(f"Error creating sofa: {e}")


def create_coffee_table():
    try:
        if hasattr(bpy, "ops"):
            bpy.ops.mesh.primitive_cube_add(location=(0, -0.3, 0.4))
            top = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
            if top:
                top.name = "coffee_table_top"
                top.scale = (0.6, 0.3, 0.2)
            leg_positions = [(0.55, -0.25, 0.2), (-0.55, -0.25, 0.2), (0.55, 0.25, 0.2), (-0.55, 0.25, 0.2)]
            for i, pos in enumerate(leg_positions):
                bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.4, location=pos)
                leg = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
                if leg:
                    leg.name = f"coffee_table_leg_{i + 1}"
        print("Coffee table created")
    except Exception as e:
        print(f"Error creating coffee table: {e}")


def create_rug():
    try:
        if hasattr(bpy, "ops"):
            bpy.ops.mesh.primitive_plane_add(location=(0, -0.3, 0.01))
            rug = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
            if rug:
                rug.name = "rug"
                rug.scale = (1.25, 0.9, 1)
        print("Rug created")
    except Exception as e:
        print(f"Error creating rug: {e}")


def create_lamp():
    try:
        if hasattr(bpy, "ops"):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.6, location=(2.5, -2.5, 0.8))
            pole = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
            if pole:
                pole.name = "lamp_pole"
            bpy.ops.mesh.primitive_cone_add(radius1=0.3, radius2=0.2, depth=0.4, location=(2.5, -2.5, 1.8))
            shade = bpy.context.active_object if hasattr(bpy.context, "active_object") else None
            if shade:
                shade.name = "lamp_shade"
        print("Lamp created")
    except Exception as e:
        print(f"Error creating lamp: {e}")


def create_materials():
    try:
        if not hasattr(bpy, "data"):
            print("Materials skipped (bpy not available)")
            return
        def make_material(name, color, roughness=0.5, metallic=0.0):
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1)
                bsdf.inputs["Roughness"].default_value = roughness
                bsdf.inputs["Metallic"].default_value = metallic
            return mat
        sofa_mat = make_material("sofa_mat", (0.6, 0.3, 0.2), 0.8)
        for obj in bpy.data.objects:
            if "sofa" in obj.name:
                obj.data.materials.append(sofa_mat)
        table_mat = make_material("table_mat", (0.4, 0.25, 0.1), 0.5)
        for obj in bpy.data.objects:
            if "coffee_table" in obj.name:
                obj.data.materials.append(table_mat)
        rug_mat = make_material("rug_mat", (0.7, 0.65, 0.6), 0.9)
        if "rug" in bpy.data.objects:
            bpy.data.objects["rug"].data.materials.append(rug_mat)
        metal_mat = make_material("lamp_metal", (0.8, 0.8, 0.8), 0.3, 1.0)
        shade_mat = make_material("lamp_shade_mat", (1, 0.9, 0.7), 0.7)
        if "lamp_pole" in bpy.data.objects:
            bpy.data.objects["lamp_pole"].data.materials.append(metal_mat)
        if "lamp_shade" in bpy.data.objects:
            bpy.data.objects["lamp_shade"].data.materials.append(shade_mat)
        wall_mat = make_material("wall_mat", (0.9, 0.88, 0.85), 0.9)
        floor_mat = make_material("floor_mat", (0.5, 0.35, 0.2), 0.6)
        for obj in bpy.data.objects:
            if "wall" in obj.name:
                obj.data.materials.append(wall_mat)
            if obj.name == "floor":
                obj.data.materials.append(floor_mat)
        print("Materials assigned")
    except Exception as e:
        print(f"Error creating materials: {e}")


def setup_lighting():
    try:
        if hasattr(bpy, "ops"):
            bpy.ops.object.light_add(type="POINT", location=(2.5, -2.5, 1.6))
            lamp_light = bpy.context.active_object
            if lamp_light:
                lamp_light.name = "lamp_light"
                lamp_light.data.energy = 800
                lamp_light.data.color = (1.0, 0.85, 0.65)
            bpy.ops.object.light_add(type="AREA", location=(0, 2.5, 2))
            fill_light = bpy.context.active_object
            if fill_light:
                fill_light.name = "fill_light"
                fill_light.data.energy = 300
                fill_light.data.color = (1.0, 0.9, 0.8)
                fill_light.data.size = 2
            if hasattr(bpy.context, "scene") and bpy.context.scene.world:
                world = bpy.context.scene.world
                world.use_nodes = True
                nodes = world.node_tree.nodes
                bg = nodes.get("Background")
                if bg:
                    bg.inputs[1].default_value = 0.2
                    bg.inputs[0].default_value = (1.0, 0.95, 0.9, 1)
        print("Lighting setup complete")
    except Exception as e:
        print(f"Error setting up lighting: {e}")


def setup_scene():
    try:
        if hasattr(bpy, "ops"):
            bpy.ops.object.camera_add(location=(4, -4, 1.7))
            cam = bpy.context.active_object
            if cam:
                cam.name = "Camera"
                cam.rotation_euler = Euler((math.radians(75), math.radians(0), math.radians(45)), "XYZ")
                cam.data.lens = 35
                cam.data.dof.use_dof = True
                if "coffee_table_top" in bpy.data.objects:
                    cam.data.dof.focus_object = bpy.data.objects["coffee_table_top"]
                cam.data.dof.aperture_fstop = 2.8
                scene = bpy.context.scene
                scene.camera = cam
                scene.render.engine = "CYCLES"
                scene.render.resolution_x = 1920
                scene.render.resolution_y = 1080
                scene.cycles.samples = 256
        print("Scene setup complete")
    except Exception as e:
        print(f"Error setting up scene: {e}")


def main():
    clear_scene()
    create_room_structure()
    create_sofa()
    create_coffee_table()
    create_rug()
    create_lamp()
    create_materials()
    setup_lighting()
    setup_scene()
    print("Living room asset generation completed successfully")


if __name__ == "__main__":
    main()
    x, y, z = 0.0, 0.0, 0.0
    name = "sofa_enhancement_cube"
    if hasattr(bpy, "ops"):
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z))
        cube = bpy.context.object if hasattr(bpy.context, "object") else None
        if cube:
            cube.name = name
    x, y, z = 0.0, 0.0, 0.0
    energy = 1.0
    size = 1.0
    if hasattr(bpy, "ops"):
        bpy.ops.object.light_add(type="AREA", location=(x, y, z))
        light = bpy.context.object if hasattr(bpy.context, "object") else None
        if light:
            light.data.energy = energy
            light.data.size = size