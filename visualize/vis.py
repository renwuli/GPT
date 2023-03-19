import copy
import random

import cv2
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import measure

__all__ = ["vis_vol", "vis_pc", "vis_dmap", "vis_mesh", "vis_sdf", "vis_pcs", "vis_label", "vis_box"]


def vis_vol(volume):
    """
    :volume numpy array
    :return None
    """
    volume = np.squeeze(volume)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(volume, edgecolor='k')
    plt.show()


def vis_pc(pointcloud):
    """
    :pointcloud 1. str, path of point cloud file
                2. np.ndarray, [N, 3]
                3. torch.tensor, [N, 3]
                4. open3d.geometry.PointCloud
    :return None
    """
    if isinstance(pointcloud, str):
        pcd = o3d.io.read_point_cloud(pointcloud)
    elif isinstance(pointcloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
    elif isinstance(pointcloud, torch.tensor):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud.detach().cpu().numpy())
    elif isinstance(pointcloud, o3d.geometry.PointCloud):
        pcd = pointcloud
    else:
        AssertionError("pointcloud is invalid")
    points = np.asarray(pcd.points)
    print(type(points))
    max, min, mean = points.max(), points.min(), points.mean()
    print("max: {}, min: {}, mean: {}".format(max, min, mean))
    o3d.visualization.draw_geometries([pcd])


def vis_mesh(mesh, color=None):
    """
    visualize mesh
    :mesh 1. an instance of open3d.geometry.TriangleMesh
          2. a list or tuple of [vertices, faces], type=np.ndarray
          3. a string: file path
    :color color to paint on mesh surfaces
    :return None
    """
    if isinstance(mesh, str):
        mesh = o3d.io.read_triangle_mesh(mesh)
    if isinstance(mesh, np.ndarray):
        vertices, faces = mesh[0].T.copy().astype(np.float32), mesh[1].T.copy().astype(np.int32)
        print("verts: {}, faces: {}".format(vertices.shape, faces.shape))
        print("==before normalized==")
        print("min: {0}, max: {1}".format(vertices.min(), vertices.max()))
        vertices = 2. * (vertices - vertices.min()) / \
            (vertices.max() - vertices.min()) - 1.
        print("==after normalized==")
        print("min: {0}, max: {1}".format(vertices.min(), vertices.max()))
        mesh = o3d.geometry.TriangleMesh()
        vertices = o3d.utility.Vector3dVector(vertices)
        faces = o3d.utility.Vector3iVector(faces)
        mesh.vertices = vertices
        mesh.triangles = faces
    mesh.compute_vertex_normals()
    if color is None:
        color = [random.random() for i in range(3)]
        mesh.paint_uniform_color(color)
    o3d.visualization.draw_geometries([mesh])


def vis_dmap(dmap, name=None):
    """
    :dmap single depth map
    :name name of dmap
    :return None
    """
    if isinstance(dmap, str):
        dmap = cv2.imread(dmap)
    dmax, dmin, dmean = dmap.max(), dmap.min(), dmap.mean()
    print("max: {}, min: {}, mean: {}".format(dmax, dmin, dmean))
    dmap = 255 * (dmap - dmin) / (dmax - dmin)
    dmap = dmap.astype(np.uint8)
    if name is None:
        cv2.imshow("Depthmaps", dmap)
    else:
        cv2.imshow(name, dmap)
    cv2.waitKey()


def vis_sdf(sdf, bbox, resolution):
    """
    :sdf tsdf with color
    :bbox bounding box
    :resolution
    :return (v, f, norm, color)
    """
    tsdf_vol = sdf[:, :, :, -1]
    color_vol = sdf[:, :, :, :-1]

    verts, faces, norms, _ = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts * resolution + bbox[:, 0]  # voxel grid coordinates to world coordinates

    colors = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors = colors.astype(np.uint8)

    vis_mesh((verts, faces), color)
    return verts, faces, norms, colors


def tensor2numpy(tensor):
    if len(tensor.shape) == 3:
        tensor = tensor[0]
    assert len(tensor.shape) == 2

    if tensor.size(0) == 3:
        return tensor.detach().cpu().numpy().transpose()
    else:
        return tensor.detach().cpu().numpy()


def numpy2pointcloud(points: np.ndarray):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud


def vis_pcs(points: Union[List[np.ndarray], Tuple[np.ndarray]], show_axis: bool = True) -> None:

    geometries = []
    colors = [
        np.array([1., 0., 0.]).astype(np.float64),
        np.array([0., 1., 0.]).astype(np.float64),
        np.array([0., 0., 1.]).astype(np.float64)
    ]
    colors.extend([np.random.rand(3, 1).astype(np.float64) for i in range(len(points) - 2)])

    for i in range(len(points)):
        point = points[i]
        if isinstance(point, torch.Tensor):
            point = tensor2numpy(point)

        pointcloud = numpy2pointcloud(point)
        pointcloud.paint_uniform_color(colors[i])
        geometries.append(pointcloud)

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
        geometries.append(axis)

    o3d.visualization.draw_geometries(geometries, width=640, height=480)


def vis_label(points: np.ndarray, label: np.ndarray) -> None:
    n = points.shape[0]
    dim = points.shape[1]
    label = np.repeat(label[:, np.newaxis], dim, 1)
    red_color = np.repeat(np.array([[1., 0., 0.]]), n, 0)
    green_color = np.repeat(np.array([[0., 1., 0.]]), n, 0)
    colors = np.where(label == 0., red_color, green_color)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pointcloud], width=640, height=480)


def vis_box(points: np.ndarray, center: np.ndarray, size: np.ndarray, rotation: np.ndarray, window_name="") -> None:
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices) * size)
    mesh.rotate(rotation)
    mesh.translate(center - size / 2)
    bbox = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
    axis.rotate(rotation)
    axis.translate(center)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pointcloud, bbox], width=640, height=480, window_name=window_name)
