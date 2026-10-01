import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

from src.utils.train_util import instantiate_from_config

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def create_blender_camera(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create Blender camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # Blender camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = normalize_vecs(z_axis)
    x_axis = torch.cross(up_world, z_axis, dim=-1)
    x_axis = normalize_vecs(x_axis)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = normalize_vecs(y_axis)

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics


def normalize_cameras(extrinsics, camera_position: torch.Tensor = None, camera_system: str = 'opencv'):
    """
    Normalize the first camera to the canonical camera position, and transform other cameras accordingly.

    extrinsics: (N, 4, 4)
    """
    if camera_position is None:
        camera_position = torch.tensor([[0, -2, 0]]).float()
    assert camera_system in ['opencv', 'opengl']

    canonical_distance = camera_position.norm()

    # compute conditional camera distances
    cond_extrinsic = extrinsics[0]
    cond_camera_distance = cond_extrinsic[:3, 3].norm(dim=-1, keepdim=False)

    # scale camera distances
    scale = canonical_distance / cond_camera_distance
    extrinsics[:, :3, 3] = extrinsics[:, :3, 3] * scale

    # rotate all cameras
    canonical_extrinsic = create_camera_to_world(camera_position, camera_system=camera_system).to(extrinsics)
    transform_matrix = torch.matmul(canonical_extrinsic, torch.linalg.inv(extrinsics[0:1]))
    normalized_extrinsics = torch.matmul(transform_matrix, extrinsics)

    return normalized_extrinsics, scale


def create_camera_to_world(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None, camera_system: str = 'opencv'):
    """
    Create OpenCV or OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    assert camera_system in ['opencv', 'opengl']
    if camera_system == 'opencv':
        # OpenCV camera: z-forward, x-right, y-down
        z_axis = look_at - camera_position
        z_axis = normalize_vecs(z_axis).float()
        x_axis = torch.cross(z_axis, up_world, dim=-1)
        x_axis = normalize_vecs(x_axis).float()
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = normalize_vecs(y_axis).float()
    else:
        # OpenGL camera: z-backward, x-right, y-up
        z_axis = camera_position - look_at
        z_axis = normalize_vecs(z_axis).float()
        x_axis = torch.cross(up_world, z_axis, dim=-1)
        x_axis = normalize_vecs(x_axis).float()
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = normalize_vecs(y_axis).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=8,
        num_workers=4,
        train=None,
        validation=None,
        test=None,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test

    def setup(self, stage):
        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        sampler = DistributedSampler(self.datasets['train'])
        return DataLoader(self.datasets['train'],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler)

    def val_dataloader(self):
        sampler = DistributedSampler(self.datasets['validation'], shuffle=False)
        return DataLoader(self.datasets['validation'],
                          batch_size=1,
                          num_workers=self.num_workers,
                          sampler=sampler)

    def test_dataloader(self):
        sampler = DistributedSampler(self.datasets['test'], shuffle=False)
        return DataLoader(self.datasets['test'],
                          batch_size=1,
                          num_workers=self.num_workers,
                          sampler=sampler)



class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='data/objaverse',
        file_name='lvis.json',
        return_paths=False,
        total_view_n=32,
        sel_view_n=8,
        validation=False,
        image_size=512,
        fov=50,
        training=True,
        overfit=False,
        validation_num=16,
        prob_grid_distortion=0.5,
        prob_cam_jitter=0.5,
        random_rotate=True,
        random_scaling=True,
        normalize_camera=False,
        supervision_reso=256,
    ):
        self.training = training
        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
        self.total_view_n = total_view_n
        self.sel_view_n = sel_view_n
        self.image_size = image_size
        self.prob_cam_jitter = prob_cam_jitter
        self.prob_grid_distortion = prob_grid_distortion
        self.random_rotate = random_rotate
        self.random_scaling = random_scaling
        self.normalize_camera = normalize_camera
        self.supervision_reso = supervision_reso

        with open(os.path.join(self.root_dir, file_name), 'r') as f:
            split = json.load(f)
        if isinstance(split, list):
            paths = split
        elif isinstance(split, dict) and 'good_objs' in split:
            paths = split['good_objs']
        elif isinstance(split, dict):
            paths = list(split.keys())
        else:
            raise ValueError(f"Unsupported split format in {file_name}")

        self.paths = list(paths)
        if validation_num > 0:
            if validation:
                self.paths = self.paths[-validation_num:]
            else:
                self.paths = self.paths[:-validation_num]

        # default camera intrinsics
        self.fov = fov
        # self.znear = 0.01
        # self.zfar = 100
        # gs-lrm
        self.znear = 0.1
        self.zfar = 4.5
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.fov))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.zfar + self.znear) / (self.zfar - self.znear)
        self.proj_matrix[3, 2] = - (self.zfar * self.znear) / (self.zfar - self.znear)
        self.proj_matrix[2, 3] = 1

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha

    def calculate_angle_between_matrices(self, R1, R2):
        cos_theta = np.dot(np.transpose(R1), R2)
        theta_rad = np.arccos(np.trace(cos_theta) / 2)
        theta_degree = np.rad2deg(theta_rad)

        return theta_degree

    def __getitem__(self, index):
        while True:
            sel_idx_list = np.random.choice(range(self.total_view_n), self.sel_view_n, replace=False)
            obj_path = self.root_dir / 'rendering_random_32views' / self.paths[index]
            # Render against the white background used by the released model.
            bg_color = np.ones((1, 3))

            img_list = []
            pose_list = []
            mask_list = []
            img_in_list = []
            bg_color_list = []

            try:
                poses = np.load(obj_path / 'cameras.npz')['cam_poses']
                for idx in sel_idx_list:
                    img_in, _ = self.load_im(obj_path / f'{idx:03d}.png', [1., 1., 1.])
                    img, mask = self.load_im(obj_path / f'{idx:03d}.png', bg_color)
                    img_in_list.append(img_in)
                    img_list.append(img)
                    mask_list.append(mask)
                    pose = poses[idx]
                    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
                    pose_list.append(pose)
                    bg_color_list.append(bg_color)

            except Exception as error:
                warnings.warn(
                    f"Failed to load {obj_path}: {error}; sampling another object.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                index = np.random.randint(0, len(self.paths))
                continue

            break

        imgs = torch.stack(img_list, 0).float()
        masks = torch.stack(mask_list, 0).float()
        imgs_in = torch.stack(img_in_list, 0).float()
        bg_colors = torch.from_numpy(np.stack(bg_color_list, 0)).float()

        if self.supervision_reso == 256:
            masks = F.interpolate(masks, size=(self.supervision_reso, self.supervision_reso), mode='nearest')

        if self.supervision_reso == 256:
            imgs = F.interpolate(imgs, size=(self.supervision_reso, self.supervision_reso), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        imgs_out = (imgs - 0.5)*2

        imgs_in = F.interpolate(imgs_in, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        imgs_in = (imgs_in - 0.5)*2

        # extrinsics (opengl camera)
        w2cs = torch.from_numpy(np.stack(pose_list, axis=0)).float()
        c2ws = torch.linalg.inv(w2cs)

        # camera normalization
        if self.normalize_camera:
            cam_radius= 2.0
            c2ws, scale = normalize_cameras(c2ws, torch.tensor([0, -cam_radius, 0]), 'opengl')
        # depths = depths * scale

        # opengl to opencv
        c2ws[:, :3, 1:3] *= -1

        # random_rotate
        if self.random_rotate:
            degree = np.random.uniform(0, math.pi * 2)
            rot = torch.tensor([
                [np.cos(degree), -np.sin(degree), 0, 0],
                [np.sin(degree), np.cos(degree), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]).unsqueeze(0).float()
            c2ws = torch.matmul(rot, c2ws)

        # normalized camera feats (transform the first pose to a fixed position)
        # if self.normalize_camera:
        #     c2ws, _ = normalize_cameras(c2ws, cond_camera_indices=[0])

        # random scaling
        if self.random_scaling:
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.7, 1.1)
                c2ws[:, :3, 3] *= scale

        # fov=50, normalize pixel coordinates to [0, 1]
        fx = fy = 0.5 / np.tan(np.deg2rad(self.fov) / 2)
        cx = cy = 0.5
        K = torch.tensor([fx, fy, cx, cy]).float()
        Ks = K.repeat(self.sel_view_n, 1)

        data = {}
        data["imgs_in"] = imgs_in
        data["imgs_out"] = imgs_out
        data["Ks"] = Ks
        data["masks"] = masks
        data["c2ws"] = c2ws
        data["bg"] = bg_colors.squeeze(1)
        if self.return_paths:
            data["path"] = str(obj_path)

        return data