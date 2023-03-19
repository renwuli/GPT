import bpy
import numpy as np
import math
import mathutils
import json


def load_camera(camera_json):
    with open(camera_json, "r") as f:
        camera_position = np.array(json.load(f)['position']) * 1.5  # / 2
        return camera_position


def setup_camera_light(camera, position, light_color):
    bpy.ops.object.select_by_type(type='MESH')
    meshes = [object for object in bpy.data.objects if object.type == 'MESH']
    obj = meshes[0]
    cns = camera.constraints.new('TRACK_TO')
    cns.target = obj
    cns.track_axis = 'TRACK_NEGATIVE_Z'
    cns.up_axis = 'UP_Y'
    camera.location = mathutils.Vector(position)

    light = bpy.data.lights['Light']
    light.color = light_color
    light.type = 'POINT'
    light.use_shadow = True
    light.specular_factor = 1.0
    light.energy = 2750
    light = bpy.data.objects['Light']
    light.location = camera.location


def setup_material():
    bpy.ops.object.select_by_type(type='MESH')
    meshes = [object for object in bpy.data.objects if object.type == 'MESH']
    obj = meshes[0]

    mat = bpy.data.materials.new(name="VertexMat")
    mat.use_nodes = True  # Make so it has a node tree

    # Add the vertex color node
    vc = mat.node_tree.nodes.new('ShaderNodeVertexColor')
    # Assign its layer
    vc.layer_name = "Col"

    # Get the shader
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    # Link the vertex color to the shader
    mat.node_tree.links.new(vc.outputs[0], bsdf.inputs[0])

    obj.data.materials.append(mat)


def setup_blender(render_path, width=1920, height=1080):
    # render layer
    scene = bpy.context.scene
    scene.cycles.device = 'GPU'
    scene.render.film_transparent = True
    # output
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    # compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)
    image_node = tree.nodes.new('CompositorNodeImage')
    scale_node = tree.nodes.new('CompositorNodeScale')
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')
    scale_node.space = 'RENDER_SIZE'
    file_output_node.base_path = render_path
    links.new(image_node.outputs[0], scale_node.inputs[0])
    links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])
    links.new(alpha_over_node.outputs[0], file_output_node.inputs[0])


def render_mesh(mesh_path, camera_position, fov, light_color):
    print(mesh_path)
    # load mesh
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete(use_global=False)
    if mesh_path.endswith("ply"):
        bpy.ops.import_mesh.ply(filepath=mesh_path)
        setup_material()
    elif mesh_path.endswith("obj"):
        bpy.ops.import_scene.obj(filepath=mesh_path)
    # camera
    camera = bpy.data.objects['Camera']
    camera.data.lens_unit = 'FOV'
    camera.data.angle = math.radians(fov / 2)
    setup_camera_light(camera, camera_position, light_color)

    image_node = bpy.context.scene.node_tree.nodes[0]
    file_output_node = bpy.context.scene.node_tree.nodes[4]
    file_output_node.file_slots[0].path = 'figure-##.png'
    bpy.ops.render.render(write_still=True)
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(current_frame + 1)


def render_meshes(mesh_pathes, camera_position, fov, light_colors):
    for i, (mesh_path, light_color) in enumerate(zip(mesh_pathes, light_colors)):
        render_mesh(mesh_path, camera_position, fov, light_color)


def main(Args):
    camera_position = load_camera(Args.camera_json)
    setup_blender(Args.render_path, Args.width, Args.height)
    if Args.mesh_path is not None and Args.mesh_pathes is None:
        render_mesh(Args.mesh_path, camera_position, Args.fov, Args.light_color)
    elif Args.mesh_path is None and Args.mesh_pathes is not None:
        render_meshes(Args.mesh_pathes, camera_position, Args.fov,
                      [Args.light_color for i in range(len(Args.mesh_pathes))])


if __name__ == '__main__':

    class Args:
        render_path = "./"

        width = 1920
        height = 1080
        camera_json = "camera_all.json"

        fov = 30
        light_color = [1., 1., 1.]
        mesh_path = None

        mesh_pathes = ["./mesh.ply"]

    main(Args)
