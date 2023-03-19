import sys

import numpy as np
import open3d as o3d

from .binvox import read_as_3d_array

__all__ = ["read_pc", "read_mesh", "read_vol"]


def read_pc(pointcloud):
    """
    :pointcloud str, point cloud file path
    :return numpy array
    """
    return np.asarray(o3d.io.read_point_cloud(pointcloud).points)


def load_off(mesh_path):
    f = open(mesh_path, 'r')
    lines = f.readlines()
    f.close()

    if lines[0].strip().lower() != 'off' and lines[0].strip()[:3].lower(
    ) != 'off':
        print("[ERROR] path: %s" % mesh_path)
        print(lines[0])
        raise Exception('Header error')
    if lines[0].strip().lower() == 'off':
        splits = lines[1].strip().split(' ')
        n_verts = int(splits[0])
        n_faces = int(splits[1])
        line_nb = 2
    else:
        splits = lines[0].strip()[3:].split(' ')
        n_verts = int(splits[0])
        n_faces = int(splits[1])
        line_nb = 1

    verts = []
    for idx in range(line_nb, line_nb + n_verts):
        verts.append([float(v) for v in lines[idx].strip().split(' ')])
        line_nb += 1
    faces = []
    for idx in range(line_nb, line_nb + n_faces):
        faces.append([int(v) for v in lines[idx].strip().split(' ')])

    faces = faces_to_triangles(faces)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    return verts, faces


def load_obj(mesh_path):
    with open(mesh_path, 'r') as fobj:
        verts = []
        faces = []
        for line in fobj:
            if len(line) > 2:
                lead, nums = line.split(" ", 1)
                if lead == 'v':
                    v = [float(x) for x in nums.split()]
                    verts.append(v)
                elif lead == 'vt':
                    v = [float(x) for x in nums.split()]
                elif lead == 'f':
                    f = [int(x.split('/')[0]) - 1 for x in nums.split()]
                    faces.append(f)
                else:
                    pass

        verts = np.asarray(verts, dtype=np.float64)
        faces = np.array(faces, dtype=np.int32)

        return verts, faces


def load_ply(mesh_path):
    mesh = plyfile.PlyData.read(mesh_path)
    ply_verts = mesh['vertex']
    verts = np.vstack([ply_verts['x'], ply_verts['y'],
                       ply_verts['z']]).astype(np.float64)
    faces = np.vstack(mesh['face']['vertex_indices']).astype(np.int32)

    return verts, faces


def faces_to_triangles(faces):
    new_faces = []
    for f in faces:
        if f[0] == 3:
            new_faces.append([f[1], f[2], f[3]])
        elif f[0] == 4:
            new_faces.append([f[1], f[2], f[3]])
            new_faces.append([f[3], f[4], f[1]])
        else:
            raise Exception('unknown face count %d' % f[0])
    return new_faces


def read_mesh(mesh_path):
    """
    :mesh str: mesh file path
    :return numpy array (V, F)
    """
    if mesh_path.endswith('.ply'):
        return load_ply(mesh_path)
    elif mesh_path.endswith('.obj'):
        return load_obj(mesh_path)
    elif mesh_path.endswith('.off'):
        return load_off(mesh_path)
    else:
        raise Exception('unknown file format')


def read_vol(volume):
    """
    :volume str: voxel file path
    :return data_structure.binvox.Voxel object
    """
    return read_as_3d_array(volume)
