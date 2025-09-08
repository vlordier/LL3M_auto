import sys
import math
import random

try:
    import bpy
    import bmesh
    import mathutils
    from mathutils import Vector, Matrix, Euler
except ImportError:
    class Dummy:
        def __getattr__(self, item):
            return self
        def __call__(self, *args, **kwargs):
            return self
    bpy = Dummy()
    bmesh = Dummy()
    mathutils = Dummy()
    Vector = Matrix = Euler = Dummy()

def clear_scene():
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        if hasattr(bpy.ops.outliner, "orphans_purge"):
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass

def create_room_shell():
    try:
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1.5))
        obj = bpy.context.active_object
        obj.name = "room-shell"
        obj.scale = (3, 3, 1.5)
        mod = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
        mod.thickness = 0.1
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception:
        pass

def create_sofa_base():
    try:
        bpy.ops.mesh.primitive_cube_add(location=(0, -1.5, 0.45))
        obj = bpy.context.active_object
        obj.name = "sofa-base"
        obj.scale = (1.1, 0.45, 0.225)
        mod1 = obj.modifiers.new(name="Subdivision", type="SUBSURF")
        mod1.levels = 2
        mod2 = obj.modifiers.new(name="Bevel", type="BEVEL")
        mod2.width = 0.05
        bpy.ops.object.modifier_apply(modifier=mod1.name)
        bpy.ops.object.modifier_apply(modifier=mod2.name)
    except Exception:
        pass

def create_sofa_back():
    try:
        bpy.ops.mesh.primitive_cube_add(location=(0, -1.15, 0.9))
        obj = bpy.context.active_object
        obj.name = "sofa-back"
        obj.scale = (1.1, 0.1, 0.45)
        mod1 = obj.modifiers.new(name="Subdivision", type="SUBSURF")
        mod1.levels = 2
        bpy.ops.object.modifier_apply(modifier=mod1.name)
    except Exception:
        pass

def create_coffee_table():
    try:
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.2))
        obj = bpy.context.active_object
        obj.name = "coffee-table"
        obj.scale = (0.5, 0.3, 0.2)
        mod = obj.modifiers.new(name="Bevel", type="BEVEL")
        mod.width = 0.05
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception:
        pass

def create_area_rug():
    try:
        bpy.ops.mesh.primitive_plane_add(location=(0, -0.5, 0.01))
        obj = bpy.context.active_object
        obj.name = "area-rug"
        obj.scale = (1.5, 1.0, 0.005)
    except Exception:
        pass

def create_lamp():
    try:
        bpy.ops.mesh.primitive_cylinder_add(location=(2.0, -1.0, 0.75))
        stand = bpy.context.active_object
        stand.name = "lamp-stand"
        stand.scale = (0.05, 0.05, 0.75)
        bpy.ops.mesh.primitive_cone_add(location=(2.0, -1.0, 1.7))
        shade = bpy.context.active_object
        shade.name = "lamp-shade"
        shade.scale = (0.25, 0.25, 0.2)
    except Exception:
        pass

def create_materials():
    try:
        def make_material(name, color, roughness=0.5, metallic=0.0):
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
                bsdf.inputs["Roughness"].default_value = roughness
                bsdf.inputs["Metallic"].default_value = metallic
            return mat
        sofa_mat = make_material("SofaFabric", (0.8, 0.7, 0.6), 0.7)
        table_mat = make_material("WoodTable", (0.5, 0.35, 0.2), 0.5)
        rug_mat = make_material("RugFabric", (0.9, 0.8, 0.7), 0.8)
        wall_mat = make_material("WallPaint", (0.95, 0.9, 0.85), 0.6)
        lamp_mat = make_material("LampMetal", (0.6, 0.6, 0.6), 0.3, 0.5)
        objs = bpy.data.objects
        if "sofa-base" in objs:
            objs["sofa-base"].data.materials.append(sofa_mat)
        if "sofa-back" in objs:
            objs["sofa-back"].data.materials.append(sofa_mat)
        if "coffee-table" in objs:
            objs["coffee-table"].data.materials.append(table_mat)
        if "area-rug" in objs:
            objs["area-rug"].data.materials.append(rug_mat)
        if "room-shell" in objs:
            objs["room-shell"].data.materials.append(wall_mat)
        if "lamp-stand" in objs:
            objs["lamp-stand"].data.materials.append(lamp_mat)
        if "lamp-shade" in objs:
            objs["lamp-shade"].data.materials.append(lamp_mat)
    except Exception:
        pass

def setup_lighting():
    try:
        bpy.ops.object.light_add(type="POINT", location=(2.0, -1.0, 1.5))
        lamp_light = bpy.context.active_object
        lamp_light.name = "LampLight"
        lamp_light.data.energy = 800
        lamp_light.data.color = (1.0, 0.85, 0.7)
        bpy.ops.object.light_add(type="AREA", location=(-2.0, 2.0, 2.5))
        fill_light = bpy.context.active_object
        fill_light.name = "FillLight"
        fill_light.data.energy = 200
        fill_light.data.color = (1.0, 0.9, 0.8)
        fill_light.data.size = 2.0
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        bg = nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (1.0, 0.9, 0.8, 1.0)
            bg.inputs[1].default_value = 0.3
    except Exception:
        pass

def setup_scene():
    try:
        bpy.ops.object.camera_add(location=(4, -4, 2))
        cam = bpy.context.active_object
        cam.name = "Camera"
        cam.rotation_euler = Euler((math.radians(75), math.radians(0), math.radians(45)), "XYZ")
        cam.data.lens = 35
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = 3.0
        cam.data.dof.aperture_fstop = 2.8
        scene = bpy.context.scene
        scene.camera = cam
        scene.render.engine = "CYCLES"
    except Exception:
        pass

def main():
    clear_scene()
    create_room_shell()
    create_sofa_base()
    create_sofa_back()
    create_coffee_table()
    create_area_rug()
    create_lamp()
    create_materials()
    setup_lighting()
    setup_scene()

if __name__ == "__main__":