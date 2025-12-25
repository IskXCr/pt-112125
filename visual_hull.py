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
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

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

def main(dataset, opt, pipe):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    # cams = scene.getTrainCameras().copy()
    # print("Running extraction...")
    # masks, transforms = extract_vh_args_from_cameras(cams)
    # print("Smoothing masks")
    # masks = apply_gaussian_blur(masks)
    # print(masks.shape)
    # print(transforms.shape)
    # print("Estimating bounding sphere")
    # center, radius = estimate_bounding_sphere(cams)
    # print(f"Center={center}, radius={radius}")
    # print("Computing visual hull...")
    # verts, faces = compute_visual_hull(masks, transforms, center, radius)
    # print(verts.shape)
    # print(verts.dtype)
    # print(faces.shape)
    # print(faces.dtype)
    
    # print("Saving USD...")
    # kaolin.io.usd.export_mesh(
    #     "./new_stage.usd",
    #     vertices=verts,
    #     faces=faces,
    #     overwrite=True,          # set True if you want to overwrite
    # )

    print("Complete")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    
    # Initialize system state (RNG)
    safe_state(None)
    os.makedirs(args.model_path, exist_ok = True)

    # Start GUI server, configure and run training
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    prepare_output_and_logger(dataset)
    main(dataset, opt, pipe)
