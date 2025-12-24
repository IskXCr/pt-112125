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
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import BasicPointCloud, GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.init_utils import (
    storePly,
    extract_vh_args_from_cameras,
    apply_gaussian_blur,
    estimate_bounding_sphere,
    compute_visual_hull,
    sample_mesh_kaolin,
    random_color
)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            print("[ INFO ] Loading trained CHECKPOINT.")
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply"
                )
            )
            return
        
        # if not scene_info.point_cloud:
        # print("[ INFO ] Initial ply is not present. Computing initialization via visual hull...")
        print("[ DEBUG ] Force computing initialization via visual hull...")
        cams = self.getTrainCameras().copy()
        print("Running extraction...")
        masks, transforms = extract_vh_args_from_cameras(cams)
        print("Smoothing masks")
        masks = apply_gaussian_blur(masks)
        # print(masks.shape)
        # print(transforms.shape)
        print("Estimating bounding sphere")
        center, radius = estimate_bounding_sphere(cams)
        print(f"Center={center}, radius={radius}")
        print("Computing visual hull...")
        verts, faces = compute_visual_hull(masks, transforms, center, radius, level=12)
        print(f"Extracted mesh: n_vertices: {verts.shape[0]}, n_triangles: {faces.shape[0]}")
        assert verts.shape[0] != 0 and faces.shape[0] != 0, "invalid construct"
        print(f"Sampling {args.init_n_points} pts")
        pts, nrm = sample_mesh_kaolin(verts, faces, args.init_n_points)
        shs = random_color(args.init_n_points)
        scene_info.point_cloud = BasicPointCloud(points=pts, colors=shs, normals=nrm)
        print(f"Saving sampled ply from visull hull mesh to {scene_info.ply_path}")
        storePly(scene_info.ply_path, pts, nrm, shs)
        
        print(f"Creating from sampled point cloud...")
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0) -> list[Camera]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> list[Camera]:
        return self.test_cameras[scale]