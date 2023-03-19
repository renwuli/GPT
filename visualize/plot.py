import numpy as np
import pyvista as pv
from pyvista import themes
from sklearn.decomposition import PCA
import torch
try:
    __IPYTHON__
except NameError:
    ipy = False
else:
    ipy = True
pv.start_xvfb()

__all__ = [
    "plot_features", "plot_points", "plot_plane", "plot_planes", "plot_correspondence", "plot_rotations", "Visualizer"
]


def feature2rgb(features):
    """
    # features: [n, c]
    """
    pca = PCA(n_components=3)
    new_features = pca.fit_transform(features)
    new_features = (new_features - new_features.min()) / (new_features.max() - new_features.min())
    new_features = new_features
    return new_features


def plot_features(points, features, feat=True, size=20, title=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    if points.shape[0] == 3:
        points = points.transpose()
        features = features.transpose()

    plotter = pv.Plotter(theme=themes.DocumentTheme(), title=title)

    pointcloud = pv.PolyData(points)
    if feat:
        pointcloud['feat'] = feature2rgb(features)
        plotter.add_mesh(
            pointcloud,
            scalars='feat',
            rgb=True,
            render_points_as_spheres=True,
            point_size=size,
        )
    else:
        pointcloud['feat'] = features
        if len(features.shape) == 1 or features.shape[1] == 1:
            plotter.add_mesh(
                pointcloud,
                scalars='feat',
                cmap='Blues',
                render_points_as_spheres=True,
                point_size=size,
            )
        else:
            plotter.add_mesh(
                pointcloud,
                scalars='feat',
                rgb=True,
                render_points_as_spheres=True,
                point_size=size,
            )

    plotter.add_axes(x_color='red', y_color='green', z_color='blue', labels_off=True)
    plotter.show(jupyter_backend='panel' if ipy else None)
    plotter.close()


def plot_points(points, size=20, color='silver', title=None):
    points = prepare_points(points)

    plotter = pv.Plotter(theme=themes.DocumentTheme(), title=title)
    pointcloud = pv.PolyData(points)
    plotter.add_mesh(
        pointcloud,
        color=color,
        render_points_as_spheres=True,
        point_size=size,
    )
    plotter.add_axes(x_color='red', y_color='green', z_color='blue', labels_off=True)
    plotter.show(jupyter_backend='panel' if ipy else None)
    plotter.close()


def plot_plane(points, plane, size=20, title=None):
    # points: [n, 3]
    # plane: [4]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(plane, torch.Tensor):
        plane = plane.detach().cpu().numpy()
    if points.shape[0] == 3:
        points = points.transpose()

    raidus = sorted(np.amax(points, 0).tolist())

    plane = pv.Plane(center=(-plane[:3] * plane[-1]).tolist(),
                     direction=[plane[0] + 1e-6, plane[1] + 1e-6, plane[2] + 1e-6],
                     i_size=raidus[-1] * 1.5,
                     j_size=raidus[-2] * 1.5,
                     i_resolution=2,
                     j_resolution=2)

    plotter = pv.Plotter(theme=themes.DocumentTheme(), title=title)
    plotter.add_points(points, render_points_as_spheres=True, point_size=size, color='silver')
    plotter.add_mesh(plane, show_edges=False, color='green', opacity=0.3)
    plotter.add_axes(x_color='red', y_color='green', z_color='blue', labels_off=True)
    plotter.show(jupyter_backend='panel' if ipy else None)
    plotter.close()


def plot_planes(points, planes, size=20, title=None):
    # points: [n, 3]
    # planes: [k, 4]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if planes is None:
        plot_points(points, size=size, title=title)
        return

    if isinstance(planes, torch.Tensor):
        planes = planes.detach().cpu().numpy()

    if points.shape[0] == 3:
        points = points.transpose()

    raidus = sorted(np.amax(points, 0).tolist())

    plotter = pv.Plotter(theme=themes.DocumentTheme(), title=title)
    plotter.add_points(points, render_points_as_spheres=True, point_size=size, color='silver')

    for i in range(len(planes)):
        plane = pv.Plane(center=(-planes[i, :3] * planes[i, 3]).tolist(),
                         direction=[planes[i, 0], planes[i, 1], planes[i, 2]],
                         i_size=0.9,
                         j_size=0.9,
                         i_resolution=2,
                         j_resolution=2)
        plotter.add_mesh(plane, show_edges=False, color='green', opacity=0.3)

    plotter.add_axes(x_color='red', y_color='green', z_color='blue', labels_off=True)
    plotter.show(jupyter_backend='panel' if ipy else None)
    plotter.close()


def plot_correspondence(points1, points2, size=20, title=None):
    points1 = prepare_points(points1)
    points2 = prepare_points(points2)

    n = points1.shape[0]
    assert (points2.shape[0] == n)

    plotter = pv.Plotter(theme=themes.DocumentTheme(), title=title)
    plotter.add_points(points1, render_points_as_spheres=True, point_size=size, color='silver')
    for i in range(n):
        line = pv.Line(points1[i], points2[i])
        plotter.add_mesh(line, color='red')
    plotter.add_axes(x_color='red', y_color='green', z_color='blue', labels_off=True)
    plotter.show(jupyter_backend='panel' if ipy else None)
    plotter.close()


def plot_rotations(points, axes, size=5, title=None, plot=True):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if axes is None:
        plot_points(points, size=size, title=title)
        return

    if isinstance(axes, torch.Tensor):
        axes = axes.detach().cpu().numpy()

    if points.shape[0] == 3:
        points = points.transpose()

    raidus = sorted(np.amax(points, 0).tolist())

    plotter = pv.Plotter(theme=themes.DocumentTheme(), title=title)
    plotter.add_points(points, render_points_as_spheres=True, point_size=size, color='silver')

    mesh = pv.PolyData(points)
    a, b, c, d, e, f = mesh.bounds
    if (b - a) < 0.2:
        a = (a + b) / 2 - 0.1
        b = (a + b) / 2 + 0.1

    if (d - c) < 0.2:
        c = (c + d) / 2 - 0.1
        d = (c + d) / 2 + 0.1

    if (f - e) < 0.2:
        e = (e + f) / 2 - 0.1
        f = (e + f) / 2 + 0.1

    bound = [a, b, c, d, e, f]

    pv_axes = []
    for i in range(len(axes)):
        axis = axes[i]
        direction = axis[:3]
        origin = axis[3:6]
        line_1 = pv.Cylinder(center=origin, direction=direction, radius=0.01, height=1.2, capping=True)
        line_2 = pv.Cylinder(center=origin, direction=-direction, radius=0.01, height=1.2, capping=True)
        line = line_1.merge(line_2)

        # line = line.clip_box(bound, invert=False)
        pv_axes.append(line.extract_geometry())
        plotter.add_mesh(line, color='red')

    if plot:
        plotter.add_axes(x_color='red', y_color='green', z_color='blue', labels_off=True)
        plotter.show(jupyter_backend='panel' if ipy else None)
        plotter.close()
    return pv_axes


class Visualizer():

    def __init__(self, meshes, rows, cols, path, window_size, **kwargs):
        self.meshes = meshes
        assert len(meshes) == rows * cols
        self.path = path

        self.plotter = pv.Plotter(theme=pv.themes.DocumentTheme(),
                                  shape=(rows, cols),
                                  border=False,
                                  window_size=window_size)

        for i, mesh in enumerate(self.meshes):
            self.plotter.subplot(i // cols, i % cols)
            self.plotter.add_mesh(mesh, **kwargs)
        self.plotter.link_views()

        self.plotter.add_key_event('s', self.save)

    def show(self):
        self.plotter.show()

    def close(self):
        self.plotter.close()

    def save(self):
        self.plotter.save_graphic(self.path)
        self.close()


if __name__ == '__main__':
    points = np.random.randn(1000, 3)
    features = np.random.randn(1000, 64)
    plot_points(points)
    visualize_feat(points, features)

    points1 = np.random.randn(100, 3)
    points2 = np.random.randn(100, 3)

    meshes = [pv.PolyData(points1), pv.PolyData(points2)]
    rows = 1
    cols = 2
    path = 'vis.pdf'
    window_size = (1000, 1000)
    visualizer = Visualizer(meshes,
                            rows,
                            cols,
                            path,
                            window_size,
                            render_points_as_spheres=True,
                            point_size=30,
                            pbr=True)
    visualizer.show()
