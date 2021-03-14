import torch
import open3d
import numpy as np
import torchvision.transforms as T


class LoadMesh(object):

    def __call__(self, x: str) -> open3d.geometry.TriangleMesh:
        return open3d.io.read_triangle_mesh(x)


class ArrayToPCL(object):

    def __call__(self, x: np.ndarray) -> open3d.geometry.PointCloud:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(x)
        return pcd

class TensorToPCL(object):

    def __call__(self, x: torch.Tensor) -> open3d.geometry.PointCloud:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(x.detach().cpu().numpy())
        return pcd


class PCLToArray(object):

    def __call__(self, x: open3d.geometry.PointCloud) -> np.ndarray:
        return np.asarray(x.points)


class PointSampler(object):

    def __init__(self, num_points: int):
        self.num_points = num_points

    def __call__(self, x: open3d.geometry.TriangleMesh) -> open3d.geometry.PointCloud:
        return x.sample_points_uniformly(number_of_points=self.num_points)


class PointJitter(object):

    def __init__(self, mu: float = 0.0, sigma: float = 0.02):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        noise = np.random.normal(loc=self.mu, scale=self.sigma, size=x.shape)
        return x + noise


class PointRotateZ(object):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        theta = np.random.random() * 2. * np.pi
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        return rotation_matrix.dot(x.T).T


class PointShuffle(object):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        np.random.shuffle(x)
        return x


class NormalizePoints(object):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x -= np.mean(x, axis=0)
        return x / np.max(np.linalg.norm(x, axis=1))


class PointsToTensor(object):

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


def train_transforms(num_points: int, mu: float = 0.0, sigma: float = 0.02) -> T.Compose:
    return T.Compose([
        LoadMesh(),
        PointSampler(num_points),
        PCLToArray(),
        PointJitter(mu, sigma),
        PointRotateZ(),
        PointShuffle(),
        NormalizePoints(),
        PointsToTensor()
    ])


def test_transforms(num_points: int) -> T.Compose:
    return T.Compose([
        LoadMesh(),
        PointSampler(num_points),
        PCLToArray(),
        NormalizePoints(),
        PointsToTensor()
    ])
