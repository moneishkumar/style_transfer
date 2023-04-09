"""
Utilities taken from : 
https://github.com/ztex08010518/Stylizing-3D-Scene/blob/73c95e77b89d5cfa674dc40ec4bcb1c58ccaeead/dataset/data_util.py#L134

"""
import glob
import math
import os
import random
from collections import OrderedDict

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

data_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(256, scale=(256 / 480, 1), ratio=(1, 1)),
        transforms.ToTensor(),
    ]
)


def find_files(dir, exts):
    # types should be ['*.png', '*.jpg']
    files_grabbed = []
    for ext in exts:
        files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
    if len(files_grabbed) > 0:
        files_grabbed = sorted(files_grabbed)
    return files_grabbed


def parse_txt(filename):
    assert os.path.isfile(filename)
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)


def get_rays_single_image(H, W, intrinsics, c2w):
    """
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    """
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5  # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth


class RaySamplerSingleImage(object):
    def __init__(
        self,
        H,
        W,
        intrinsics,
        c2w,
        img_path=None,
        resolution_level=1,
        mask_path=None,
        min_depth_path=None,
        max_depth=None,
        style_imgs=None,
    ):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w
        self.img_path = img_path

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)
        self.style_imgs = style_imgs

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = int(self.W_orig // resolution_level)
            self.H = int(self.H_orig // resolution_level)
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level

            # only load image at this time
            self.img = imageio.imread(self.img_path)[..., :3].astype(np.float32) / 255.0
            self.img = cv2.resize(
                self.img, (self.W, self.H), interpolation=cv2.INTER_AREA
            )
            self.img = self.img.reshape((-1, 3))

            self.rays_o, self.rays_d, self.depth = get_rays_single_image(
                self.H, self.W, self.intrinsics, self.c2w_mat
            )

    def get_img(self):
        return self.img.reshape((self.H, self.W, 3))

    def get_style_img(self):
        return imageio.imread(self.style_img_path)[..., :3].astype(np.float32) / 255.0

    def get_style_input(self, mode=None, test_seed=None, style_ID=None):
        if mode == "test":
            # Fixed seed to choose the same style image for each GPU
            random.seed(test_seed)
            if style_ID != None:
                self.style_img_path = self.style_imgs[style_ID]
            else:
                self.style_img_path = random.sample(self.style_imgs, 1)[0]
        else:
            self.style_img_path = np.random.choice(self.style_imgs, 1)[0]

        ori_style_img = Image.open(self.style_img_path).convert("RGB")
        style_img = data_transform(ori_style_img)
        style_idx = torch.from_numpy(
            np.array([self.style_imgs.index(self.style_img_path)])
        )

        return style_img, style_idx

    def get_all(self):
        min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])
        ret = OrderedDict(
            [
                ("ray_o", self.rays_o),
                ("ray_d", self.rays_d),
                ("depth", self.depth),
                ("rgb", self.img),
                ("min_depth", min_depth),
            ]
        )
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])
        return ret
