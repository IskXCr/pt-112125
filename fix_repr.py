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
from collections import defaultdict

from fix_utils import *

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
    parent_dir = os.path.join(dataset.model_path)
    os.makedirs(parent_dir, exist_ok=True)
    print("[fix_repr] Closing the hole")
    seal_bottom_with_cap(
        os.path.join(parent_dir, f"reconstruct.ply"),
        os.path.join(parent_dir, f"estimated_plane.ply"),
        os.path.join(parent_dir, f"reconstruct_closed.ply")
    )


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
