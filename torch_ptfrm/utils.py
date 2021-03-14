import open3d


def visualize(x):
    open3d.visualization.draw_geometries([x])