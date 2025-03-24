import __init__

import torch

from backbone.camera import CameraOptions
from backbone.mamba_ssm.models import MambaConfig
from utils.config import EasyConfig


class ScanNetV2Config(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ScanNetV2Config'
        self.k = [24, 24, 24, 24, 24]
        self.grid_size = [0.04, 0.08, 0.16, 0.32]
        self.voxel_max = 80000
        self.cam_opts = CameraOptions.default(n_cameras=64)
        self.alpha = 0.1
        if self.alpha == 0:
            self.cam_opts.n_cameras = 1  # whatever


class ScanNetV2WarmupConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ScanNetV2WarmupConfig'
        self.k = [24, 24, 24, 24, 24]
        self.grid_size = [0.04, 0.08, 0.16, 0.32]
        self.voxel_max = 80000
        self.cam_opts = CameraOptions.default(n_cameras=64)
        self.alpha = 0.1
        if self.alpha == 0:
            self.cam_opts.n_cameras = 1  # whatever


class ModelConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelConfig'
        self.train_cfg = ScanNetV2Config()
        self.warmup_cfg = ScanNetV2WarmupConfig()
        self.num_classes = 20
        self.bn_momentum = 0.02
        drop_path = 0.1
        backbone_cfg = EasyConfig()
        backbone_cfg.name = 'CamPointModelConfig'
        backbone_cfg.in_channels = 7
        backbone_cfg.channel_list = [64, 128, 256, 384, 512]
        backbone_cfg.head_channels = 384
        backbone_cfg.mamba_blocks = [1, 1, 1, 1, 1]
        backbone_cfg.res_blocks = [4, 4, 4, 8, 4]
        backbone_cfg.mlp_ratio = 2.
        backbone_cfg.bn_momentum = self.bn_momentum
        drop_rates = torch.linspace(0., drop_path, sum(backbone_cfg.res_blocks)).split(backbone_cfg.res_blocks)
        backbone_cfg.drop_paths = [d.tolist() for d in drop_rates]
        backbone_cfg.head_drops = torch.linspace(0., 0.2, len(backbone_cfg.res_blocks)).tolist()
        backbone_cfg.mamba_config = MambaConfig.default()
        backbone_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}
        backbone_cfg.cam_opts = self.train_cfg.cam_opts
        backbone_cfg.diff_factor = 60.
        backbone_cfg.diff_std = [1.6, 2.5, 5, 10, 20]
        # backbone_cfg.diff_std = None
        self.backbone_cfg = backbone_cfg
