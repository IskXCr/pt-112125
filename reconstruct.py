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
import open3d as o3d
import numpy as np
import pymeshfix
from random import randint
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.init_utils import BoundedMeshExtractor
from functools import partial
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args

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

def main(dataset, pipe, iteration):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_visual_hull=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    parent_dir = os.path.join(scene.model_path)
    os.makedirs(parent_dir, exist_ok=True)
    tmesh_cuda = BoundedMeshExtractor.reconstruct(
        render_f=partial(render, pc=gaussians, pipe=pipe, bg_color=background),
        cameras=scene.getTrainCameras().copy(),
        save_mesh_path=os.path.join(parent_dir, f"reconstruct.ply")
    )
    tmesh = tmesh_cuda.cpu()
    print(f"[reconstruct.py] pymeshfix")

    tmesh_filled = tmesh.fill_holes(1e8)
    o3d.t.io.write_triangle_mesh(os.path.join(parent_dir, "reconstruct_filled.ply"), tmesh_filled)
    verts, faces = tmesh.vertex["positions"].numpy(), tmesh.triangle["indices"].numpy()

    meshfix = pymeshfix.MeshFix(verts, faces)
    meshfix.repair()
    mesh_fixed = meshfix.mesh
    verts_fixed, faces_fixed = o3d.core.Tensor.from_numpy(meshfix.v), o3d.core.Tensor.from_numpy(meshfix.f)

    tmhes_fixed = o3d.t.geometry.TriangleMesh(verts_fixed, faces_fixed)

    tmesh_fixed_path = os.path.join(parent_dir, f"reconstruct_fixed.ply")
    print(f"[reconstruct.py] Saving processed mesh to {tmesh_fixed_path}")
    o3d.t.io.write_triangle_mesh(tmesh_fixed_path, tmhes_fixed)

    print("[reconstruct.py] All files have been written")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)

    print(f"Rendering {args.model_path}, iteration={args.iteration}")
    
    # Initialize system state (RNG)
    safe_state(None)
    os.makedirs(args.model_path, exist_ok = True)

    # Start GUI server, configure and run training
    dataset, pipe, iteration = model.extract(args), pipeline.extract(args), args.iteration
    prepare_output_and_logger(dataset)
    main(dataset, pipe, iteration)
