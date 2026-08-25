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

from scene.cameras import Camera
import numpy as np
import torch
import torchvision
import cv2
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False
MASK_READ = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if cam_info.mask_path != "":
        global MASK_READ
        if not MASK_READ:
            print("[ INFO ] Dataset has provided at least one non-empty mask. You would see this info only once.")
            print(f"[ INFO ] Path of the target mask: {cam_info.mask_path}")
            MASK_READ = True
        mask_np = cv2.imread(cam_info.mask_path, 0)
        if mask_np is None:
            raise FileNotFoundError(f"Failed to read mask image at: {cam_info.mask_path}")

        # Exact-mask mode uses the input silhouette after binary thresholding,
        # with nearest-neighbor resizing and no boundary expansion.
        mask_np = ((mask_np > 127).astype(np.uint8)) * 255

        # Keep mask aligned with the (potentially) resized RGB image.
        # Note: cv2.resize expects (width, height).
        if mask_np.shape[1] != resolution[0] or mask_np.shape[0] != resolution[1]:
            mask_np = cv2.resize(mask_np, dsize=resolution, interpolation=cv2.INTER_NEAREST)

        # CPU dilation with OpenCV (fast, avoids CUDA OOM)
        if not args.exact_masks and args.dilate_mask and int(args.dilation_radius) > 0:
            r = int(args.dilation_radius)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)

        mask_bool = (mask_np > 0)
        if args.invert_mask:
            mask_bool = ~mask_bool

        alpha_mask = torch.from_numpy(mask_bool)[None]
        assert alpha_mask.ndim == 3
        alpha_mask = alpha_mask.expand(3, -1, -1).contiguous()
    else:
        alpha_mask = None
    gt_image = resized_image_rgb

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, alpha_mask=alpha_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
