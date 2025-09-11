import sys

try:
    import bpy
except ModuleNotFoundError:
    print("bpy module not found. This script must be run inside Blender's Python environment.")
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
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = "Cube"
        obj.scale = (1.0, 1.0, 1.0)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        obj.dimensions = (2.0, 2.0, 2.0)
        return obj
    except Exception as e:
        print(f"Error creating geometry: {e}")
        return None


def create_materials():
    try:
        mat = bpy.data.materials.new(name="BlueMetallic")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for n in nodes:
            nodes.remove(n)
        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 1.0, 1.0)
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.3
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        return mat
    except Exception as e:
        print(f"Error creating material: {e}")
        return None


def assign_material(obj, mat):
    try:
        if obj is not None and mat is not None:
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
    except Exception as e:
        print(f"Error assigning material: {e}")


def main():
    clear_scene()
    cube = create_geometry()
    mat = create_materials()
    assign_material(cube, mat)
    print("Blue metallic cube created successfully")


if __name__ == "__main__":
