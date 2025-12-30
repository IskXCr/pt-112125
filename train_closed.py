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

import os
import torch
import math
from random import randint
from utils.loss_utils import (
    l1_loss,
    ssim,
    compute_gradient_smoothness,
    compute_laplacian_smoothness
)
from trace_utils import WarpMeshTracer
from gaussian_renderer import render, network_gui
import sys
import open3d as o3d
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from utils.init_utils import BoundedMeshExtractor, BoundedVisullHullExtractor
from functools import partial
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from typing import Optional
import random
import mathutils
from utils.init_utils import estimate_bounding_sphere
from pathlib import Path
import numpy as np

from torch.utils.tensorboard import SummaryWriter
TENSORBOARD_FOUND = True

class Class_0:
    def __init__(
        self,
        seed: int = 114514,
        frame_current: int = 0,
        upper_views: bool = False,
        center: np.ndarray = np.array([0.0, 0.0, 0.0]),
        radius: float = 1.0,
        n_frames: int = 120
    ):
        self.seed = seed
        self.frame_current = frame_current
        self.upper_views = upper_views
        self.center = center
        self.radius = radius
        self.n_frames = n_frames

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))


def _look_at_opencv_w2c(
    cam_centers: torch.Tensor,   # (N, 3) in world
    target: torch.Tensor,        # (3,) or (N, 3) in world
    up_world: torch.Tensor = None,  # (3,)
) -> torch.Tensor:
    """
    Build OpenCV-style W2C extrinsics:
      X_cam = R * X_world + t
    where camera axes are:
      +x: right, +y: down, +z: forward (towards target)
    """
    device = cam_centers.device
    dtype = cam_centers.dtype
    N = cam_centers.shape[0]

    if target.ndim == 1:
        target = target.unsqueeze(0).expand(N, 3)
    if up_world is None:
        up_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    up = up_world.unsqueeze(0).expand(N, 3)

    # Forward (+z): from camera to target
    z = _normalize(target - cam_centers)

    # If z is too parallel to up, switch to an alternative up to avoid degeneracy
    # (e.g., camera on the "north pole")
    parallel = (torch.abs((z * up).sum(dim=-1)) > 0.999).unsqueeze(-1)  # (N,1)
    up_alt = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).unsqueeze(0).expand(N, 3)
    up = torch.where(parallel, up_alt, up)

    # Right (+x) = z × up  (this choice yields correct "right" for OpenCV y-down convention)
    x = _normalize(torch.cross(z, up, dim=-1))

    # Down (+y) = z × x
    y = torch.cross(z, x, dim=-1)  # already normalized if x,z are orthonormal

    # W2C rotation: rows are camera axes expressed in world coords
    R = torch.stack([x, y, z], dim=1)  # (N, 3, 3)

    # t = -R * C
    t = -(R @ cam_centers.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    # Assemble 4x4
    extr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    extr[:, :3, :3] = R
    extr[:, :3, 3] = t
    return extr

# https://github.com/maximeraafat/BlenderNeRF/blob/main/helper.py
def sample_from_sphere(scene: Class_0):
    np.random.seed(scene.seed)

    N = scene.n_frames
    theta = np.random.random(N) * 2.0 * math.pi

    if scene.upper_views:
        # Uniform hemisphere: z ~ U[0,1], theta ~ U[0,2pi]
        z = np.random.random(N)  # [0,1]
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        unit_x = np.cos(theta) * r
        unit_y = np.sin(theta) * r
        unit_z = z
    else:
        # Uniform sphere
        u = np.random.random(N)
        phi = np.arccos(1.0 - 2.0 * u)
        unit_x = np.cos(theta) * np.sin(phi)
        unit_y = np.sin(theta) * np.sin(phi)
        unit_z = np.cos(phi)

    unit = np.stack([unit_x, unit_y, unit_z], axis=1).astype(np.float32)  # (N,3)

    center = torch.tensor(scene.center, device="cuda", dtype=torch.float32)
    cam_centers = center.unsqueeze(0) + float(scene.radius) * torch.from_numpy(unit).to("cuda")

    extr_w2c = _look_at_opencv_w2c(cam_centers, center)
    return extr_w2c

from kornia.utils._compat import torch_meshgrid


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])

    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    # TODO: torchscript doesn't like `torch_version_ge`
    # if torch_version_ge(1, 13, 0):
    #     x, y = torch_meshgrid([xs, ys], indexing="xy")
    #     return stack([x, y], -1).unsqueeze(0)  # 1xHxWx2
    # TODO: remove after we drop support of old versions
    base_grid: torch.Tensor = torch.stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2



@torch.amp.autocast(device_type="cuda", dtype=torch.float32)
def get_ray_directions(
    H: torch.Tensor, W: torch.Tensor, K: torch.Tensor, device="cpu", ray_jitter=None, return_uv=False, flatten=False
):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
            ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        ray_jitter: Optional RayJitter component, for whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if ray_jitter is None:  # pass by the center
        directions = torch.stack(
            [(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1
        )
    else:
        jitter = ray_jitter(u.shape)
        directions = torch.stack(
            [
                ((u + jitter[:, :, 0]) - cx) / fx,
                ((v + jitter[:, :, 1]) - cy) / fy,
                torch.ones_like(u),
            ],
            -1,
        )
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid

    return torch.nn.functional.normalize(directions, dim=-1)

@torch.amp.autocast(device_type="cuda", dtype=torch.float32)
def get_rays(directions: torch.Tensor, c2w: torch.Tensor):
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
            ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (..., 3) ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (..., 3), the origin of the rays in world coordinate
        rays_d: (..., 3), the direction of the rays in world coordinate
    """
    assert c2w.ndim == 2

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., 3].expand_as(rays_d)

    return rays_o, rays_d

class DummyCamera(torch.nn.Module):
    def __init__(self):
        super(DummyCamera, self).__init__()
        self.image_width = None
        self.image_height = None
        self.znear = None
        self.zfar = None
        self.camera_center = None
        self.world_view_transform = None
        self.projection_matrix = None
        self.full_proj_transform = None
        self.FoVx = None
        self.FoVy = None
        self.gt_alpha_mask = None
        self.original_image = None

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, skip_visual_hull=True, skip_gaussian_init=True)
    # TODOs:
    # [x] 1. Read a sample camera in. Get the intrinsics. We'll compute the extrinsics later. 
    # [x] 2. Render alpha images of the mesh. Distribute them over a sphere (Copy BlenderNeRF's CoS code).
    #        We'll need 120 views and corresponding alpha masks.
    # [x] 3. Run visual hull once to get an initial point distribution.
    # [x] 4. Train starting from 15000 to 30000 iterations without densification/pruning.
    # [x] 5. Run TSDF extraction once at 27000 iteration, prune floaters
    # [x] 6. Run an additional 5000 iterations to stabilize.
    # [x] 7. Run TSDF extraction again to obtain a mesh.
    # [x] 8. Save the representation.

    # ===========================================
    print("===========================================")
    # 0. Parameters
    N = 120 # number of cameras
    P = 300000 # number of sample points

    print(f"Number of desired cameras: {N}, sample points: {P}")
    
    # ===========================================
    # 1. Read a sample camera
    print("===========================================")

    print(f"Obtaining a sample camera")
    sample_camera = scene.getTrainCameras()[0]
    W, H = sample_camera.image_width, sample_camera.image_height

    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W-1) / 2],
        [0, H / 2, 0, (H-1) / 2],
        [0, 0, 0, 1]]).float().cuda()
    
    proj_mat = sample_camera.projection_matrix.detach().clone().transpose(0, 1) # This projection matrix will be shared amongst all cameras later.

    intrins = (ndc2pix @ proj_mat)[:3, :3]

    # ===========================================
    # 2. Now we compute the extrinsics by sampling on the sphere

    print("===========================================")
    print("Estimating bounding sphere")

    center, radius = estimate_bounding_sphere(scene.getTrainCameras())
    center = np.array(center)
    print(f"Bounding sphere with center {center} and radius {radius}")

    class_0 = Class_0(center=center, radius=radius, n_frames=N)
    w2c_transforms = sample_from_sphere(class_0)

    # For each transform and view we get the corresponding rays and render them through WarpMeshTracer
    base_mesh_path = os.path.join(scene.model_path, "reconstruct_closed.ply")
    print(f"Reading mesh from {base_mesh_path}")
    mesh = o3d.io.read_triangle_mesh(base_mesh_path)
    assert not mesh.is_empty()
    tracer = WarpMeshTracer.from_open3d(mesh)

    rays_d_view_cached = get_ray_directions(H=H, W=W, K=intrins, device="cuda")

    silhouettes = []
    
    # Render silhouettes
    for i in tqdm(range(N), desc="Obtaining silhouttes"):
        rays_ori, rays_dir = get_rays(rays_d_view_cached, w2c_transforms[i].inverse()[:3, :])
        _, depth = tracer.trace(rays_ori[None], rays_dir[None])
        silhouettes.append((depth > 0).float().reshape(-1, H, W).contiguous().cpu())
    
    print(f"Silhouette shape: {silhouettes[0].shape}")

    # ===========================================
    # 3. Now we run visual hull once to get an initial point distribution

    print("===========================================")

    # First by generating a bunch of cameras the BoundedVisualHullExtractor expects
    # This will also later be used during rendering
    dummy_image = torch.zeros((3, H, W), dtype=torch.float32)
    cameras = [DummyCamera() for _ in range(N)]

    for i, cam in tqdm(enumerate(cameras), desc="Creating cameras"):
        # Properties that will be used:
        # image_width, image_height, znear, zfar, world_view_transform, camera_center, projection_matrix, full_proj_transform, FoVx, FoVy, gt_alpha_mask
        cam.image_width = W
        cam.image_height = H
        cam.znear = sample_camera.znear
        cam.zfar = sample_camera.zfar
        cam.world_view_transform = w2c_transforms[i].transpose(0, 1).cuda()
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        cam.projection_matrix = proj_mat.transpose(0, 1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.FoVx = sample_camera.FoVx
        cam.FoVy = sample_camera.FoVy
        cam.gt_alpha_mask = silhouettes[i]
        cam.original_image = dummy_image
    
    # Run visual hull once
    print("Running visual hull")
    pcd = BoundedVisullHullExtractor.reconstruct(
        cameras,
        P,
        os.path.join(scene.model_path, "closed", "visual_hull_mesh.ply"),
        os.path.join(scene.model_path, "closed", "visual_hull_samples.ply")
    )

    # Last, initialize Gaussians
    print("Initializing Gaussians")
    gaussians.create_from_pcd(pcd, scene.cameras_extent)

    # ===========================================
    # 4. Train

    # Setting desired parameters
    first_iter = 15000
    final_iter = 32000
    lambda_depth_distortion_ex = 1e5
    lambda_avg_scale_ex = 0.7

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_total_for_log = 0.0
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    mesh: Optional[o3d.t.geometry.TriangleMesh] = None

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, final_iter + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, rend_alpha = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["rend_alpha"]
        )

        gt_image = viewpoint_cam.original_image.cuda()
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        rend_dist = render_pkg["rend_dist"]

        raw_mask_loss = 0
        
        if viewpoint_cam.gt_alpha_mask is not None:
            alpha_mask = viewpoint_cam.gt_alpha_mask.cuda().expand(3, -1, -1)
            raw_mask_loss = (rend_alpha - alpha_mask).abs().mean()
            image = image * alpha_mask
            gt_image = gt_image * alpha_mask
            rend_normal = torch.full_like(rend_normal, fill_value=math.sqrt(1/3)) * (1.0 - alpha_mask) + rend_normal * alpha_mask
            surf_normal = torch.full_like(surf_normal, fill_value=math.sqrt(1/3)) * (1.0 - alpha_mask) + surf_normal * alpha_mask
            rend_dist = rend_dist * alpha_mask[0]

        Ll1 = torch.zeros([1], device="cuda")
        loss = Ll1
    
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = lambda_depth_distortion_ex if iteration > 3000 else 0.0
        lambda_normal_grad = opt.lambda_normal_grad if iteration > 15000 else 0.0
        lambda_avg_scale = lambda_avg_scale_ex if iteration > 3000 else 0.0
        lambda_max_scale = opt.lambda_max_scale if iteration > 3000 else 0.0

        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        smooth_loss = lambda_normal_grad * compute_gradient_smoothness(rend_normal[None, ...])
        scales = gaussians.get_scaling.abs()
        avg_scale_loss = lambda_avg_scale * scales.mean(dim=-1).std()
        max_scale_loss = lambda_max_scale * scales.max(dim=-1)[0].mean()
        
        total_loss = loss * opt.lambda_rgb + dist_loss + normal_loss + smooth_loss + avg_scale_loss + max_scale_loss + opt.lambda_mask * raw_mask_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_total_for_log = 0.4 * total_loss.item() + 0.6 * ema_total_for_log
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/total_loss', ema_total_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print("\n[ITER {}] Saving fused mesh now.".format(iteration))
                parent_dir = os.path.join(scene.model_path, "mesh", f"iteration_{iteration}")
                os.makedirs(parent_dir, exist_ok=True)
                mesh = BoundedMeshExtractor.reconstruct(
                    render_f=partial(render, pc=gaussians, pipe=pipe, bg_color=background),
                    cameras=cameras.copy(),
                    save_mesh_path=os.path.join(parent_dir, f"fused_post.ply")
                )
            
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < final_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Cluster-based Pruning
            if iteration >= opt.densify_until_iter and iteration > opt.prune_floaters_from_iter\
                and iteration < opt.prune_floaters_until_iter and iteration % opt.prune_floaters_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.opacity_prune(opt.opacity_cull, scene.cameras_extent, size_threshold, True)
                print(f"\n[ITER {iteration}] Running intermediate mesh extraction")
                parent_dir = os.path.join(scene.model_path, "mesh", f"iteration_{iteration}")
                os.makedirs(parent_dir, exist_ok=True)
                mesh = BoundedMeshExtractor.reconstruct(
                    render_f=partial(render, pc=gaussians, pipe=pipe, bg_color=background),
                    cameras=cameras.copy(),
                    save_mesh_path=os.path.join(parent_dir, f"fused_post.ply")
                )
                print(f"[ITER {iteration}] Pruning Gaussians...")
                gaussians.mesh_prune(mesh, opt.prune_floaters_eps)
                torch.cuda.empty_cache()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(final_iter)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[32_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[32_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    print(f"invert_mask=\"{args.invert_mask}\", dilate_mask=\"{args.dilate_mask}\", dilation_radius=\"{args.dilation_radius}\"")

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
