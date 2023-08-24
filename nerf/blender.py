import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np
from PIL import Image

import glob

from kornia import create_meshgrid

# This code is borrowed from https://github.com/kwea123/nerf_pl/blob/master/datasets/blender.py
# I modified the return batch for whole resolution image


def get_ray_directions(H, W, focal):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack(
        [(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1)
    return directions  # (H, W, 3)


def get_rays(directions, c2w):
    rays_d = directions @ c2w[:, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:, 3].expand(rays_d.shape)  # H, W, 3
    rays_d = rays_d.view(-1, 3)  # H*W, 3
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


class BlenderDataset(Dataset):
    def __init__(self, device, root_dir, split='train', img_wh=(80, 80)):
        self.root_dir = root_dir
        self.split = split
        # assert img_wh[0] == img_wh[1]
        self.img_wh = img_wh
        self.define_transform()

        self.read_meta()
        self.white_back = True
        self.device = device

    def define_transform(self):
        self.transform = T.ToTensor()

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5*self.meta['camera_angle_x'])

        self.focal *= self.img_wh[0]/800

        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        self.directions = get_ray_directions(h, w, self.focal)

    def __len__(self):
        if self.split == "train":
            return len(self.meta['frames'])
        if self.split == "val":
            return 1  # only validate  1 images (to support <=8 gpus)
        return len(self.meta["frames"])  # for test

    def __getitem__(self, idx):
        frame = self.meta["frames"][idx]

        c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

        img = Image.open(os.path.join(
            self.root_dir, f"{frame['file_path']}.png")).convert("RGB")

        file_name = frame['file_path'].split('/')[-1]
        # depth = Image.open(os.path.join(self.root_dir, f"train_depth/{file_name}-dpt_beit_large_512.png"))

        mask_img = Image.open(os.path.join(
            self.root_dir, f'train_mask/{file_name}.png')).convert("RGB")
        img = img.resize(self.img_wh, Image.LANCZOS)
        mask_img = mask_img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)

        mask_img = self.transform(mask_img)

        valid_mask = (img[-1] > 0).flatten()
        img = img.view(3, -1).permute(1, 0)
        mask_img = mask_img.view(3, -1).permute(1, 0)[..., :1]

        rays_o, rays_d = get_rays(self.directions, c2w)

        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)

        sample = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "rgbs": img,
            "mask": mask_img,
            "H": self.img_wh[1],
            "W": self.img_wh[0],
            "c2w": c2w,
            # TODO: return dirs(方向)
            "valid_mask": valid_mask}
        return sample
