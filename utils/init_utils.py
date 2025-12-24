import torch
import torch.nn.functional as F
import torchhull
import kaolin
import numpy as np
from plyfile import PlyData, PlyElement
from functools import partial

from scene.cameras import Camera
from .graphics_utils import getProjectionMatrix
from .render_utils import focus_point_fn

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    positions_cuda, colors_cuda, normals_cuda = map(lambda x: torch.tensor(x).to(dtype=torch.float32, device="cuda"), (positions, colors, normals))
    return positions_cuda, colors_cuda, normals_cuda

def storePly(path, xyz, normals, rgb):
    xyz, normals, rgb = map(lambda x: x.contiguous().cpu().numpy(), (xyz, normals, rgb))
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

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
    # assert not (faces >= 2 ** 31 - 1).any()
    return verts, faces

def sample_mesh_kaolin(verts: torch.Tensor,
                       faces: torch.Tensor,
                       num_samples: int = 100_000,
                       eps: float = 1e-12):
    """
    Sample points uniformly by triangle area using Kaolin and return:
      - positions (N,3)
      - smooth normals (N,3) (interpolated from vertex normals)

    verts: (V,3) float tensor on CUDA
    faces: (F,3) int tensor on CUDA (int64 preferred)
    """

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must be (V,3), got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {faces.shape}")

    if verts.device.type != "cuda" or faces.device.type != "cuda":
        raise ValueError("Put verts and faces on CUDA for fast sampling.")

    if faces.dtype not in (torch.int64, torch.int32):
        faces = faces.to(torch.int64)
    if verts.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        verts = verts.float()

    B = 1
    V = verts.shape[0]
    F = faces.shape[0]

    verts_b = verts.unsqueeze(0)  # (1, V, 3)

    # (1, F, 3, 3) = per-face triangle vertex positions
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(verts_b, faces)

    # (1, F, 3) per-face unit normals
    face_normals = kaolin.ops.mesh.face_normals(face_verts, unit=True)  # (1, F, 3)

    # ---- FIX: expand per-face normals -> per-corner normals (1, F, 3, 3) ----
    face_corner_normals = face_normals.unsqueeze(2).expand(B, F, 3, 3)  # (1, F, 3, 3)

    # (1, V, 3) vertex normals by averaging incident face-corner normals
    vnorm_b = kaolin.ops.mesh.compute_vertex_normals(
        faces=faces,
        face_normals=face_corner_normals,
        num_vertices=V
    )  # (1, V, 3)
    vnorm_b = vnorm_b / vnorm_b.norm(dim=-1, keepdim=True).clamp_min(eps)

    # (1, F, 3, 3) gather vertex normals per face-corner for interpolation
    vnorm_f = kaolin.ops.mesh.index_vertices_by_faces(vnorm_b, faces)  # (1, F, 3, 3)

    # sample_points returns: points, face_ids, interpolated face_features
    pts_b, face_ids_b, nrm_b = kaolin.ops.mesh.sample_points(
        vertices=verts_b, faces=faces, num_samples=num_samples, face_features=vnorm_f
    )

    pts = pts_b.squeeze(0).contiguous()   # (N, 3)
    nrm = nrm_b.squeeze(0).contiguous()   # (N, 3)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(eps)

    return pts, nrm

def z_axis_to_quat(z_dirs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Args
    ----
    z_dirs : (N,3) torch.Tensor
        Each row is the desired (possibly non-unit) rotated z-axis.
    Returns
    -------
    quat  : (N,4) torch.Tensor
        Unit quaternions in [w, x, y, z] order.
    """
    # 1. normalise the incoming axes
    z = F.normalize(z_dirs, dim=-1, eps=eps)                       # (N,3)

    # 2. choose a stable 'up' for each vector
    world_up = torch.tensor([0., 1., 0.], device=z.device)         # (3,)
    alt_up   = torch.tensor([1., 0., 0.], device=z.device)
    world_up = world_up.expand_as(z)                               # (N,3)
    alt_up   = alt_up.expand_as(z)
    use_alt  = (torch.abs((world_up * z).sum(-1)) > 0.99).unsqueeze(-1)
    up       = torch.where(use_alt, alt_up, world_up)              # (N,3)

    # 3. construct orthonormal basis
    x = F.normalize(torch.cross(up, z, dim=-1), dim=-1, eps=eps)   # (N,3)
    y = torch.cross(z, x, dim=-1)                                  # (N,3)
    R = torch.stack((x, y, z), dim=-1)                             # (N,3,3)

    # 4. matrix → quaternion (w first)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + trace, min=0.0)) * 0.5

    # avoid division by very small w
    denom = 4.0 * w + eps
    xq = (R[:, 2, 1] - R[:, 1, 2]) / denom
    yq = (R[:, 0, 2] - R[:, 2, 0]) / denom
    zq = (R[:, 1, 0] - R[:, 0, 1]) / denom

    quat = torch.stack((w, xq, yq, zq), dim=-1)                    # (N,4)
    quat = F.normalize(quat, dim=-1, eps=eps)
    return quat

def random_color(n_pts: int):
    return torch.rand((n_pts, 3), dtype=torch.float32, device="cuda") / 255.0

# class BoundedMeshExtractor:
#     def __init__(self, gaussians, render_kwargs):
