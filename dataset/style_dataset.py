import logging
import pdb

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

logger = logging.getLogger(__package__)

import os

from utils import *


class StyleTransferDataset(Dataset):

    """
    Dataset for StyleTransfer
    -> (scene images) and (styles)
    """

    def __init__(self, config_path):
        conf = OmegaConf.load(config_path)

        self.basedir = conf.dataset.scene.basedir
        self.split = conf.dataset.scene.split
        self.name = conf.dataset.scene.name

        self.style_basedir = conf.dataset.style.basedir
        self.num_styles = conf.dataset.style.num_styles
        self.load_files()
        self.read_files()
        # pdb.set_trace()

    def load_files(self):
        path = os.path.join(self.basedir, self.name)
        self.intrinsics_files = find_files(path, exts=["*.txt"])

        path = os.path.join(self.basedir, self.name, "pose/")
        self.pose_files = find_files(path, exts=["*.txt"])
        self.camera_count = len(self.pose_files)

        path = os.path.join(self.basedir, self.name, "rgb/")
        self.img_files = find_files(path, exts=["*.png", "*.jpg"])

        self.style_img_files = find_files(self.style_basedir, exts=["*.png", "*.jpg"])[
            : self.num_styles
        ]

        logger.info("raw intrinsics_files: {}".format(len(self.intrinsics_files)))
        logger.info("raw pose_files: {}".format(len(self.pose_files)))
        logger.info("raw img_files: {}".format(len(self.img_files)))
        logger.info("camera count: {}".format(self.camera_count))
        logger.info("Styles count: {}".format(len(self.style_img_files)))

    def read_files(self):
        self.ray_samplers = []

        # TODO: write a dict file
        H, W = [1084, 1957]
        intrinsics = parse_txt(self.intrinsics_files[0])
        for i in range(self.camera_count):
            pose = parse_txt(self.pose_files[i])

            # TODO: this should not take style images
            self.ray_samplers.append(
                RaySamplerSingleImage(
                    H=H,
                    W=W,
                    intrinsics=intrinsics,
                    c2w=pose,
                    img_path=self.img_files[i],
                    mask_path=None,
                    min_depth_path=None,
                    max_depth=None,
                    style_imgs=self.style_img_files,
                )
            )

        logger.info("Images Read: {}".format(len(self.ray_samplers)))
        pass

    def __len__(self):
        return len(self.ray_samplers)

    def __getitem__(self, idx):
        return self.ray_samplers[idx]


def unit_test():
    """ """
    config_path = "/home/moneish/style_transfer/configs/data_config.yaml"
    dataset = StyleTransferDataset(config_path)
    print(dataset[0].get_img().shape)


if __name__ == "__main__":
    logging.basicConfig(
        filename="test.log", encoding="utf-8", filemode="w", level=logging.DEBUG
    )
    unit_test()
