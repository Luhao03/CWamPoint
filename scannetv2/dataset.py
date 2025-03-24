import __init__

import math
import random
from pathlib import Path

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset

from backbone.camera import CameraOptions, CameraHelper, make_cam_points, merge_cam_points
from utils.cutils import grid_subsampling, grid_subsampling_test

train_file = Path(__file__).parent / "scannetv2_train.txt"
val_file = Path(__file__).parent / "scannetv2_val.txt"
with open(train_file, 'r') as file:
    scan_train_list = [line.strip() for line in file.readlines()]
with open(val_file, 'r') as file:
    scan_val_list = [line.strip() for line in file.readlines()]


class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, coord):
        coord = coord.numpy()
        if random.random() < 0.95:
            for granularity, magnitude in self.distortion_params:
                coord = self.elastic_distortion(coord, granularity, magnitude)
        return torch.from_numpy(coord)


class ScanNetV2(Dataset):
    classes = [
        'wall', 'floor', 'cabinet', 'bed',
        'chair', 'sofa', 'table', 'door',
        'window', 'bookshelf', 'picture', 'counter',
        'desk', 'curtain', 'refrigerator', 'shower curtain',
        'toilet', 'sink', 'bathtub', 'otherfurniture']
    num_classes = len(classes)
    label2color = {
        0: (0., 0., 0.),
        1: (174., 199., 232.),
        2: (152., 223., 138.),
        3: (31., 119., 180.),
        4: (255., 187., 120.),
        5: (188., 189., 34.),
        6: (140., 86., 75.),
        7: (255., 152., 150.),
        8: (214., 39., 40.),
        9: (197., 176., 213.),
        10: (148., 103., 189.),
        11: (196., 156., 148.),
        12: (23., 190., 207.),
        14: (247., 182., 210.),
        15: (66., 188., 102.),
        16: (219., 219., 141.),
        17: (140., 57., 197.),
        18: (202., 185., 52.),
        19: (51., 176., 203.),
        20: (200., 54., 131.),
        21: (92., 193., 61.),
        22: (78., 71., 183.),
        23: (172., 114., 82.),
        24: (255., 127., 14.),
        25: (91., 163., 138.),
        26: (153., 98., 156.),
        27: (140., 153., 101.),
        28: (158., 218., 229.),
        29: (100., 125., 154.),
        30: (178., 127., 135.),
        32: (146., 111., 194.),
        33: (44., 160., 44.),
        34: (112., 128., 144.),
        35: (96., 207., 209.),
        36: (227., 119., 194.),
        37: (213., 92., 176.),
        38: (94., 106., 211.),
        39: (82., 84., 163.),
        40: (100., 85., 144.),
    }

    def __init__(self,
                 dataset_dir: Path,
                 loop=6,
                 train=True,
                 warmup=False,
                 voxel_max=64000,
                 k=[24, 24, 24, 24, 24],
                 grid_size=[0.04, 0.08, 0.16, 0.32],
                 alpha=0.,
                 batch_size=8,
                 cam_opts: CameraOptions = CameraOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)
        data_list = scan_train_list if train else scan_val_list
        self.data_paths = [f"{dataset_dir}/{p}.pt" for p in data_list]
        self.loop = loop
        self.train = train
        self.warmup = warmup
        self.voxel_max = voxel_max
        self.k = k
        self.grid_size = grid_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.cam_opts = cam_opts

        assert len(self.data_paths) > 0

        if train and warmup:
            max_n = 0
            selected_data = self.data_paths[0]
            for data in self.data_paths:
                n = torch.load(data)[0].shape[0]
                if n > max_n:
                    max_n = n
                    selected_data = data
            # use selected data with max n to warmup model
            self.data_paths = [selected_data]

        self.datas = [torch.load(path) for path in self.data_paths]
        self.els = ElasticDistortion()

    def __len__(self):
        return len(self.data_paths) * self.loop

    @classmethod
    def get_classes(cls):
        return cls.classes

    def __getitem__(self, idx):
        if not self.train:
            return self.get_test_item(idx)

        idx //= self.loop
        xyz, col, norm, lbl = self.datas[idx]

        angle = random.random() * 2 * math.pi
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        norm = norm @ rotmat
        rotmat *= random.uniform(0.8, 1.2)
        xyz = xyz @ rotmat
        xyz = self.els(xyz)
        xyz -= xyz.min(dim=0)[0]

        # here grid size is assumed 0.02, so estimated downsampling ratio is ~1.5
        indices = grid_subsampling(xyz, 0.02, 2.5 / 1.5)
        xyz = xyz[indices]
        if xyz.shape[0] > self.voxel_max:
            pt = random.choice(xyz)
            condition = (xyz - pt).square().sum(dim=1).argsort()[:self.voxel_max].sort()[0]  # sort to preserve locality
            xyz = xyz[condition]
            indices = indices[condition]

        col = col[indices].float()
        rgb = col.clone()
        lbl = lbl[indices]
        norm = norm[indices]

        if random.random() < 0.2:
            col.fill_(0.)
        else:
            if random.random() < 0.2:
                colmin = col.min(dim=0, keepdim=True)[0]
                colmax = col.max(dim=0, keepdim=True)[0]
                scale = 255 / (colmax - colmin)
                alpha = random.random()
                col = (1 - alpha + alpha * scale) * col - alpha * colmin * scale
            col.mul_(1 / 250.)
        if self.train and random.random() < 0.2:
            norm.fill_(0.)

        height = xyz[:, 2:]
        feature = torch.cat([col, height, norm], dim=1)

        cam_helper = CameraHelper(self.cam_opts, batch_size=self.batch_size, device=xyz.device)
        cam_helper.projects(xyz)
        cam_helper.cam_points.__update_attr__('p', xyz)
        cam_helper.cam_points = make_cam_points(cam_helper.cam_points, self.k, self.grid_size,
                                                None, up_sample=True, alpha=self.alpha)
        cam_helper.cam_points.__update_attr__('f', feature)
        cam_helper.cam_points.__update_attr__('y', lbl)
        cam_helper.cam_points.__update_attr__('rgb', rgb)
        return cam_helper.cam_points

    def get_test_item(self, idx):
        rotations = [0, 0.5, 1, 1.5]
        scales = [0.95, 1, 1.05]
        augs = len(rotations) * len(scales)
        aug = idx % self.loop
        pick = aug // augs
        aug %= augs

        idx //= self.loop
        xyz, col, norm, lbl = self.datas[idx]

        angle = math.pi * rotations[aug // len(scales)]
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        norm = norm @ rotmat
        rotmat *= scales[aug % len(scales)]
        xyz = xyz @ rotmat
        xyz -= xyz.min(dim=0)[0]

        indices = grid_subsampling_test(xyz, 0.02, 2.5 / 1.5, pick=pick)
        xyz = xyz[indices]
        lbl = lbl[indices]
        col = col[indices].float()
        rgb = col.clone()
        norm = norm[indices]

        col.mul_(1 / 250.)

        xyz -= xyz.min(dim=0)[0]
        feature = torch.cat([col, xyz[:, 2:], norm], dim=1)

        cam_helper = CameraHelper(self.cam_opts, batch_size=self.batch_size, device=xyz.device)
        cam_helper.projects(xyz)
        cam_helper.cam_points.__update_attr__('p', xyz)
        cam_helper.cam_points = make_cam_points(cam_helper.cam_points, self.k, self.grid_size,
                                                None, up_sample=True, alpha=self.alpha)
        cam_helper.cam_points.__update_attr__('f', feature)
        cam_helper.cam_points.__update_attr__('y', lbl)
        cam_helper.cam_points.__update_attr__('rgb', rgb)
        return cam_helper.cam_points


def scannetv2_collate_fn(batch):
    cam_list = list(batch)
    new_cam_points = merge_cam_points(cam_list)
    return new_cam_points
