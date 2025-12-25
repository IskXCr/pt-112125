import torch
import torch.nn.functional as F
import torchhull
import kaolin
import numpy as np
import open3d as o3d
import open3d.core as o3c
import os
import gc
from plyfile import PlyData, PlyElement
from functools import partial
from tqdm import tqdm
from typing import Callable

from scene.cameras import Camera
from .graphics_utils import getProjectionMatrix, BasicPointCloud
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
    sigma: float=1.0
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

def random_color(n_pts: int):
    return torch.rand((n_pts, 3), dtype=torch.float32, device="cuda") / 255.0

def camera_to_o3d(cam: Camera) -> o3d.camera.PinholeCameraParameters:
    W = cam.image_width
    H = cam.image_height
    
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W-1) / 2],
        [0, H / 2, 0, (H-1) / 2],
        [0, 0, 0, 1]]).float().cuda().T

    intrins =  (cam.projection_matrix @ ndc2pix)[:3,:3].T
    intrinsic=o3d.camera.PinholeCameraIntrinsic(
        width=cam.image_width,
        height=cam.image_height,
        cx = intrins[0,2].item(),
        cy = intrins[1,2].item(), 
        fx = intrins[0,0].item(), 
        fy = intrins[1,1].item()
    )

    extrinsic=np.asarray((cam.world_view_transform.T).cpu().numpy())
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic
    camera.intrinsic = intrinsic

    return camera

class BoundedVisullHullExtractor:
    @classmethod
    def sample_once(
        cls,
        init_n_points: int,
        masks: torch.Tensor,
        transforms: torch.Tensor,
        center: list[float],
        radius: float,
        level: int,
        isolevel: float = 0.5
    ):
        print(f"[BoundedVisullHullExtractor] =========================================")
        print(f"[BoundedVisullHullExtractor] Computing visual hull at level {level}...")
        print(f"[BoundedVisullHullExtractor] Isolevel: {isolevel}")
        verts, faces = torchhull.visual_hull(
            masks,  # [B, H, W, 1]
            transforms,  # [B, 4, 4]
            level,
            [center[0] - radius, center[1] - radius, center[2] - radius],
            radius * 2,
            masks_partial=False,
            transforms_convention="opengl",
            unique_verts=True,
        )
        # assert not (faces >= 2 ** 31 - 1).any()
        
        print(f"[BoundedVisullHullExtractor] Extracted mesh: n_vertices: {verts.shape[0]}, n_triangles: {faces.shape[0]}")
        assert verts.shape[0] != 0 and faces.shape[0] != 0, "invalid construct"

        print(f"[BoundedVisullHullExtractor] Sampling {init_n_points} pts")
        pts, nrm = sample_mesh_kaolin(verts, faces, init_n_points)
        shs = random_color(init_n_points)
        print(f"[BoundedVisullHullExtractor] =========================================")
        return pts, nrm, shs

    @classmethod
    def reconstruct(
        cls,
        cameras: list[Camera],
        init_n_points: int,
        save_sample_path: str,
        levels: list[int] = [11],
        isolevel: float = 0.5
    ):
        print("[BoundedVisullHullExtractor] Running extraction...")
        masks, transforms = extract_vh_args_from_cameras(cameras)

        print("[BoundedVisullHullExtractor] Smoothing masks")
        masks = (apply_gaussian_blur(masks) > 0.5).to(dtype=torch.float32)
        print(f"[BoundedVisullHullExtractor] #Non-Binary-Elements: {((masks > 0) & (masks < 1.0)).sum()}")

        print("[BoundedVisullHullExtractor] Estimating bounding sphere")
        center, radius = estimate_bounding_sphere(cameras)
        print(f"[BoundedVisullHullExtractor] Center={center}, radius={radius}")

        pts, nrm, shs = BoundedVisullHullExtractor.sample_once(
            init_n_points,
            masks,
            transforms,
            center,
            radius,
            levels[0],
            isolevel
        )

        point_cloud = BasicPointCloud(points=pts, colors=shs, normals=nrm)

        print(f"[BoundedVisullHullExtractor] Saving sampled ply from visull hull mesh to {save_sample_path}")
        storePly(save_sample_path, pts, nrm, shs)
        
        return point_cloud

class BoundedMeshExtractor:
    @classmethod
    @torch.no_grad()
    @torch.cuda.nvtx.range("BoundedMeshExtractor.reconstruction")
    def reconstruct(
        cls,
        render_f: Callable,
        cameras: list[Camera],
        save_mesh_path: str,
        mesh_resolution: int=1024,
        n_clusters_to_keep: int=1000
    ) -> o3d.t.geometry.TriangleMesh:
        """
        Docstring for reconstruction
        
        :param render_f: Will be invoked in the form of `render_f(cam, gaussians)`
        :type render_f: Callable
        :param gaussians: Description
        :type gaussians: GaussianModel
        :param cameras: Description
        :type cameras: list[Camera]
        :param mesh_resolution: Description
        :type mesh_resolution: int
        :param n_clusters_to_keep: Description
        :type n_clusters_to_keep: int
        """
        print(f"[BoundedMeshExtractor] Estimating bounding sphere")
        scene_center, scene_radius = estimate_bounding_sphere(cameras)
        print(f"[BoundedMeshExtractor] center={list(map(lambda x: f'{x:.4e}', scene_center))}, radius={scene_radius:.4e}")

        depth_trunc: float = scene_radius * 2.0
        voxel_size = depth_trunc / mesh_resolution
        sdf_trunc = 5.0 * voxel_size

        print(f"[BoundedMeshExtractor] Creating TSDF volume, depth_trunc={depth_trunc:.4e}, voxel_size={voxel_size:.4e}, sdf_trunc={sdf_trunc:.4e}")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        for cam in tqdm(cameras, desc="[BoundedMeshExtractor] Bounded Mesh Extraction..."):
            render_pkg = render_f(cam)
            pred_rgb = render_pkg['render']
            pred_depth = render_pkg['surf_depth']

            # Erase depth outside mask
            pred_depth[cam.gt_alpha_mask < 0.5] = 0

            o3d_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(pred_rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(pred_depth.permute(1, 2, 0).cpu().numpy(), order="C")),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0
            )

            o3d_cam = camera_to_o3d(cam)
            volume.integrate(o3d_rgbd_image, intrinsic=o3d_cam.intrinsic, extrinsic=o3d_cam.extrinsic)
        
        mesh: o3d.geometry.TriangleMesh = volume.extract_triangle_mesh()
        print(f"\n[BoundedMeshExtractor] #vertices before postprocessing {len(mesh.vertices)}")

        print(f"[BoundedMeshExtractor] Running postprocessing clustering: Target #clusters: {n_clusters_to_keep}")

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

        triangle_clusters = np.asanyarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        n_clusters_to_keep = min(n_clusters_to_keep, cluster_n_triangles.shape[0])

        print(f"[BoundedMeshExtractor] #clusters present: {n_clusters_to_keep}")

        n_cluster = np.sort(cluster_n_triangles.copy())[-n_clusters_to_keep]
        n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh.remove_triangles_by_mask(triangles_to_remove)

        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()

        print(f"[BoundedMeshExtractor] #vertices after postprocessing {len(mesh.vertices)}")

        print(f"[BoundedMeshExtractor] Converting to o3d.t.geometry.TriangleMesh for further processing.")
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        o3d_device = o3d.core.Device("CUDA:0")
        print(f"[BoundedMeshExtractor] Using o3d_device: {o3d_device}")
        tmesh = tmesh.to(device=o3d_device)

        o3d.t.io.write_triangle_mesh(save_mesh_path, tmesh)
        print(f"[BoundedMeshExtractor] Saved mesh to {save_mesh_path}")

        return tmesh

def filter_points_near_mesh_from_torch(
    pts_xyz_cuda: torch.Tensor,
    mesh: o3d.t.geometry.TriangleMesh,
    eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        kept_mask:      (N,)  torch.bool on the same CUDA device
        signed_dists:   (N,)  torch.float32 signed SDF (inside<0, outside>0)
    """
    assert pts_xyz_cuda.ndim == 2 and pts_xyz_cuda.shape[1] == 3, "pts must be (N,3)"
    if pts_xyz_cuda.dtype != torch.float32:
        pts_xyz_cuda = pts_xyz_cuda.float()
    pts_xyz_cuda = pts_xyz_cuda.contiguous()

    device_str = f"CPU:0"
    o3d_dev = o3d.core.Device(device_str)
    mesh = mesh.to(device=o3d_dev)

    # Zero-copy via DLPack
    pts_dlpack = torch.utils.dlpack.to_dlpack(pts_xyz_cuda.cpu())
    o3d_pts = o3d.core.Tensor.from_dlpack(pts_dlpack)

    # Build GPU BVH and compute signed distances
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # For watertight meshes we can use signed distance directly.
    # Negative: inside; Positive: outside.
    sdf = scene.compute_signed_distance(o3d_pts)  # (N,) Float32 CUDA

    # Make the mask (|SDF| <= thresh)
    thr = o3d.core.Tensor([eps], dtype=sdf.dtype)
    keep_mask_o3d = (sdf.abs() <= thr).to(o3d.core.Dtype.Float32)

    # Also export the full signed distance array + boolean mask to torch
    signed_dists = torch.utils.dlpack.from_dlpack(sdf.to_dlpack()).to(device=pts_xyz_cuda.device)
    kept_mask = torch.utils.dlpack.from_dlpack(keep_mask_o3d.to_dlpack()).to(device=pts_xyz_cuda.device)
    kept_mask = (kept_mask > 0)

    return kept_mask, signed_dists