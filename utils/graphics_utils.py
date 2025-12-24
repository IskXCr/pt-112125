#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
import open3d as o3d
from typing import NamedTuple, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BasicPointCloud:
    points : torch.Tensor
    colors : torch.Tensor
    normals : torch.Tensor

    def __post_init__(self):
        assert isinstance(self.points, torch.Tensor) and self.points.is_cuda
        assert isinstance(self.colors, torch.Tensor) and self.colors.is_cuda
        assert isinstance(self.normals, torch.Tensor) and self.normals.is_cuda
        N = self.points.shape[0]
        assert self.points.shape == (N, 3)
        assert self.colors.shape == (N, 3)
        assert self.normals.shape == (N, 3)

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def floater_mask_dbscan_open3d(
    positions_cuda: torch.Tensor,
    eps: float = 5e-2,
    min_points: int = 10,
    keep_top_k: int = 1,
    min_cluster_size: Optional[int] = None
):
    """
    Cluster points with Open3D DBSCAN and keep the top-K largest clusters
    (optionally also enforcing a minimum cluster size). Returns a (N,1) mask.

    Args:
        positions_cuda: torch.Tensor (N,3) on CUDA.
        eps: DBSCAN radius (your EPS = 5e-2).
        min_points: DBSCAN min_points (Open3D requirement).
        keep_top_k: keep K largest clusters (K=1 keeps main object).
        min_cluster_size: if set, clusters smaller than this are dropped even if in top-K.

    Returns:
        mask: torch.bool tensor of shape (N,1) on same device as input.
        (optional) labels: torch.int64 tensor of shape (N,) on same device as input.
    """
    if positions_cuda.ndim != 2 or positions_cuda.shape[1] != 3:
        raise ValueError(f"positions_cuda must have shape (N,3), got {tuple(positions_cuda.shape)}")
    if not positions_cuda.is_cuda:
        raise ValueError("positions_cuda must be on CUDA device")
    if positions_cuda.numel() == 0:
        mask = torch.empty((0, 1), dtype=torch.bool, device=positions_cuda.device)
        return mask

    # Open3D legacy DBSCAN runs on CPU in most installs -> move to CPU numpy
    pts = positions_cuda.detach().float().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))

    # Open3D returns labels as a list/array of ints, -1 = noise
    labels_np = np.asarray(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False),
        dtype=np.int32,
    )

    # Find cluster sizes (exclude noise = -1)
    valid = labels_np >= 0
    if not np.any(valid):
        # Degenerate case: everything marked as noise -> keep everything to avoid nuking geometry
        mask = torch.ones((pts.shape[0], 1), dtype=torch.bool, device=positions_cuda.device)
        return mask

    cluster_ids, counts = np.unique(labels_np[valid], return_counts=True)
    order = np.argsort(counts)[::-1]  # descending by size
    cluster_ids = cluster_ids[order]
    counts = counts[order]

    # Choose clusters to keep
    k = int(max(1, keep_top_k))
    keep_ids = cluster_ids[:k]

    if min_cluster_size is not None:
        keep_ids = keep_ids[counts[:k] >= int(min_cluster_size)]

    keep_mask_np = np.isin(labels_np, keep_ids)

    mask = torch.from_numpy(keep_mask_np).to(device=positions_cuda.device).bool().view(-1, 1)
    return mask.to(device=positions_cuda.device)

