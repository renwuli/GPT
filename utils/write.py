import sys

import numpy as np
import open3d as o3d
import torch

from .binvox import *

__all__ = ["write_pc", "write_mesh"]


def write_pc(pointcloud, path):
    """
    :pointcloud 1. numpy array
                2. torch tensor
                3. open3d PointCloud object
    :path str, path to write
    :return None
    """
    assert isinstance(path, str)
    if isinstance(pointcloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
    elif isinstance(pointcloud, torch.tensor):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud.detach().cpu().numpy())
    elif isinstance(pointcloud, o3d.geometry.PointCloud):
        pcd = pointcloud
    else:
        AssertionError("point cloud is invalid")

    o3d.io.write_point_cloud(path, pcd)


def write_mesh(mesh, path):
    """
    :mesh 1. tuple of numpy arrays, (V, F)
          2. open3d TriangleMesh object
    :path str, path to write
    :return None
    """
    assert isinstance(path, str)
    if isinstance(mesh, tuple) or isinstance(mesh, list):
        vertices, faces = mesh[0].T.copy().astype(np.float32), \
            mesh[1].T.copy().astype(np.int32)
        mesh = o3d.geometry.TriangleMesh()
        vertices = o3d.utility.Vector3dVector(vertices)
        faces = o3d.utility.Vector3dVector(faces)
        mesh.vertices = vertices
        mesh.triangles = faces

    o3d.io.write_triangle_mesh(path, mesh)