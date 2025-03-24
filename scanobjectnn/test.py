import __init__

import argparse
import logging
import os
import sys
from glob import glob

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone import CamPointModel, ClsHead, merge_cam_points
from scanobjectnn.configs import model_configs
from scanobjectnn.dataset import ScanObjectNN, scanobjectnn_collate_fn
from utils.config import EasyConfig
from utils.logger import setup_logger_dist, format_dict
from utils.metrics import Timer, AverageMeter, Metric
from utils.misc import set_random_seed, resume_state, cal_model_params, cal_model_flops


def prepare_exp(cfg):
    exp_root = 'exp-test'
    exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
    cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
    cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
    cfg.log_path = f'{cfg.exp_dir}/test.log'

    os.makedirs(cfg.exp_dir, exist_ok=True)
    setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)
    logfile = open(cfg.log_path, "a", 1)
    sys.stdout = logfile


def cal_flops(cfg, model):
    cfg.model_cfg.train_cfg.num_points = 1024
    cfg.model_cfg.train_cfg.n_samples = [1024, 256, 64]
    ds = ScanObjectNN(
        dataset_dir=cfg.dataset,
        train=True,
        warmup=False,
        num_points=cfg.model_cfg.train_cfg.num_points,
        k=cfg.model_cfg.train_cfg.k,
        n_samples=cfg.model_cfg.train_cfg.n_samples,
        alpha=cfg.model_cfg.train_cfg.alpha,
        batch_size=1,
        cam_opts=cfg.model_cfg.train_cfg.cam_opts
    )
    cam_points = merge_cam_points([ds[0]], up_sample=False)
    cam_points.to_cuda(non_blocking=True)
    flops, macs, params = cal_model_flops(model, inputs=(cam_points,))
    logging.info(f'FLOPs = {flops / 1e9:.4f}G, Macs = {macs / 1e9:.4f}G, Params = {params / 1e6:.4f}')


def warmup(cfg, model, test_loader):
    steps_per_epoch = 10
    pbar = tqdm(enumerate(test_loader), total=steps_per_epoch, desc='Warmup')
    i = 0
    for idx, cam_points in pbar:
        cam_points.to_cuda(non_blocking=True)
        with autocast():
            model(cam_points)
        pbar.set_description(f"Warmup [{idx}/{steps_per_epoch}]")

        i += 1
        if i > steps_per_epoch:
            return


@torch.no_grad()
def main(cfg):
    torch.cuda.set_device(0)
    set_random_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'Config:\n{cfg.__str__()}')

    test_loader = DataLoader(
        ScanObjectNN(
            dataset_dir=cfg.dataset,
            train=False,
            warmup=False,
            num_points=cfg.model_cfg.train_cfg.num_points,
            k=cfg.model_cfg.train_cfg.k,
            n_samples=cfg.model_cfg.train_cfg.n_samples,
            alpha=cfg.model_cfg.train_cfg.alpha,
            batch_size=cfg.batch_size,
            cam_opts=cfg.model_cfg.train_cfg.cam_opts
        ),
        batch_size=cfg.batch_size,
        collate_fn=scanobjectnn_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )

    backbone = CamPointModel(
        **cfg.model_cfg.backbone_cfg,
        task_type='cls',
    ).to('cuda')
    model = ClsHead(
        backbone=backbone,
        num_classes=cfg.model_cfg.num_classes,
        bn_momentum=cfg.model_cfg.bn_momentum,
        cls_type='max',
    ).to('cuda')
    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    model.eval()
    cal_flops(cfg, model)

    if cfg.ckpt == '':
        logging.warning(f'Checkpoint path is empty, quit now...')
        return
    resume_state(model, cfg.ckpt)

    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()
    m = Metric(cfg.num_classes)
    steps_per_epoch = len(test_loader)

    warmup(cfg, model, test_loader)
    pbar = tqdm(enumerate(test_loader), total=test_loader.__len__(), desc='Testing')
    for idx, cam_points in pbar:
        cam_points.to_cuda(non_blocking=True)
        target = cam_points.y
        timer.record(f'I{idx}')
        with autocast():
            pred = model(cam_points)
        time_cost = timer.record(f'I{idx}')
        timer_meter.update(time_cost)
        m.update(pred, target)
        pbar.set_description(f"Testing [{idx}/{steps_per_epoch}] "
                             + f"mACC {m.calc_macc():.4f}")
        if writer is not None and idx % cfg.metric_freq == 0:
            writer.add_scalar('time_cost_avg', timer_meter.avg, idx)
            writer.add_scalar('time_cost', time_cost, idx)
    acc, macc, miou, iou = m.calc()
    test_info = {
        'macc': macc,
        'oa': acc,
        'time_cost_avg': f"{timer_meter.avg:.8f}s",
    }
    logging.info(f'Summary:'
                 + f'\ntest: \n{format_dict(test_info)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('scanobjectnn testing')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='scanobjectnn')
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='s',
                        choices=['s', 'l', 'c'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='dataset_link')
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--num_workers', type=int, required=False, default=16)

    # for test
    parser.add_argument("--metric_freq", type=int, required=False, default=1)

    # for model
    parser.add_argument("--use_cp", action='store_true')

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load_args(args)

    model_cfg = model_configs[cfg.model_size]
    cfg.model_cfg = model_cfg
    cfg.model_cfg.backbone_cfg.use_cp = cfg.use_cp
    if cfg.use_cp:
        cfg.model_cfg.backbone_cfg.bn_momentum = 1 - (1 - cfg.model_cfg.bn_momentum) ** 0.5

    # scanobjectnn
    cfg.model_cfg.backbone_cfg.mamba_config.scan_method = 'random'

    cfg.num_classes = 15
    cfg.ignore_index = -100

    prepare_exp(cfg)
    main(cfg)
