# COPY from Fantasia3D
import numpy as np
import torch


def rotate_x_homo(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[1, 0, 0], [0, c, s], [0, -s, c]], dtype=torch.float32, device=device
    )


def rotate_z(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float32, device=device
    )


def rotate_z_homo(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def rotate_y_homo(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32, device=device
    )


def rotate_y_numpy(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotate_x_numpy(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def scale(s, device=None):
    return torch.tensor(
        [[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
