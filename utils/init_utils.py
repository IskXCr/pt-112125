import torch
import torchhull
import kaolin
import numpy as np

from scene.cameras import Camera
from .graphics_utils import getProjectionMatrix
from .render_utils import focus_point_fn

import torch

# Transform z coords from [0, 1] to [-1, 1]
__builtin_NDC_transform_M = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 2., -1.],
    [0., 0., 0., 1.],
], device="cuda", dtype=torch.float32)

def extract_vh_args_from_cameras(cameras: list[Camera]):
    lst_0 = [(cam.gt_alpha_mask.permute(1, 2, 0).cuda(), cam.full_proj_transform.T) for cam in cameras]
    mask_lst, transform_lst = zip(*lst_0)
    masks, transforms = torch.stack(mask_lst, dim=0), torch.stack(transform_lst, dim=0)
    transforms = __builtin_NDC_transform_M[None] @ transforms

    return masks, transforms

def apply_gaussian_blur(
    masks: torch.Tensor,
    kernel_size: int=3,
    sigma: float=0.1
):
    return torchhull.gaussian_blur(
        masks, # [B, H, W, 1]
        kernel_size,
        sigma,
        sparse=True,
    )

def estimate_bounding_sphere(cameras: list[Camera]):
    """
    Estimate the bounding sphere given camera pose
    """
    torch.cuda.empty_cache()
    c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in cameras])
    poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    center = (focus_point_fn(poses))
    radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
    return center.tolist(), radius

def compute_visual_hull(
    masks: torch.Tensor,
    transforms: torch.Tensor,
    camera_center: torch.Tensor,
    radius: torch.Tensor,
    level: int=11,
    masks_partial: bool=False
) -> tuple[torch.Tensor, torch.Tensor]:
    verts, faces = torchhull.visual_hull(
        masks,  # [B, H, W, 1]
        transforms,  # [B, 4, 4]
        level,
        [camera_center[0] - radius, camera_center[1] - radius, camera_center[2] - radius],
        radius * 2,
        masks_partial=masks_partial,
        transforms_convention="opengl",
        unique_verts=True,
    )
    return verts, faces