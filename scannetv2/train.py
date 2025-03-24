import __init__

import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone.models import SegSemHead, CamPointModel
from scannetv2.configs import model_configs
from scannetv2.dataset import ScanNetV2, scannetv2_collate_fn
from utils.config import EasyConfig
from utils.logger import setup_logger_dist, format_dict, format_list
from utils.metrics import Timer, Metric, AverageMeter
from utils.misc import set_random_seed, load_state, save_state, resume_state, cal_model_params


def prepare_exp(cfg):
    exp_root = 'exp'
    exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
    if cfg.mode == 'resume':
        exp_id -= 1
    cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
    cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
    cfg.log_path = f'{cfg.exp_dir}/train.log'
    cfg.best_small_ckpt_path = f'{cfg.exp_dir}/best_small.ckpt'
    cfg.best_ckpt_path = f'{cfg.exp_dir}/best.ckpt'
    cfg.last_ckpt_path = f'{cfg.exp_dir}/last.ckpt'

    os.makedirs(cfg.exp_dir, exist_ok=True)
    # cfg.save(f'{cfg.exp_dir}/config.yaml')
    setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)


def warmup(cfg, model: nn.Module, warmup_loader):
    if cfg.mode != 'train':
        logging.warning(f'only warmup when train model: {cfg.mode} != "train"')
        return
    model.train()
    pbar = tqdm(enumerate(warmup_loader), total=warmup_loader.__len__(), desc='Warmup')
    for idx, cam_points in pbar:
        cam_points.to_cuda(non_blocking=True)
        target = cam_points.y
        with autocast():
            pred, diff = model(cam_points)
            loss = F.cross_entropy(pred, target, ignore_index=cfg.ignore_index) + diff
        loss.backward()


def train(cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc='Train')
    m = Metric(cfg.num_classes)
    loss_meter = AverageMeter()
    diff_meter = AverageMeter()
    steps_per_epoch = len(train_loader)
    for idx, cam_points in pbar:
        lam = scheduler_steps/(epoch*steps_per_epoch)
        lam = 3e-3 ** lam * 0.2
        scheduler.step(scheduler_steps)
        scheduler_steps += 1
        cam_points.to_cuda(non_blocking=True)
        target = cam_points.y
        mask = target != 20
        with autocast():
            pred, diff = model(cam_points)
            loss = F.cross_entropy(pred, target, label_smoothing=cfg.ls, ignore_index=cfg.ignore_index)
        optimizer.zero_grad(set_to_none=True)
        if cfg.use_amp:
            scaler.scale(loss + diff*lam).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss + diff*lam
            loss.backward()
            optimizer.step()

        m.update(pred[mask], target[mask])
        loss_meter.update(loss)
        diff_meter.update(diff)
        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                             + f"Loss {loss_meter.avg:.4f} "
                             + f"Diff {diff_meter.avg:.4f} "
                             + f"mACC {m.calc_macc():.4f}")
    acc, macc, miou, iou = m.calc()
    return loss_meter.avg, diff_meter.avg, miou, macc, iou, acc, scheduler_steps


def validate(cfg, model, val_loader, epoch):
    model.eval()
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    m = Metric(cfg.num_classes)
    loss_meter = AverageMeter()
    for idx, cam_points in pbar:
        cam_points.to_cuda(non_blocking=True)
        target = cam_points.y
        mask = target != cfg.ignore_index
        with autocast():
            pred = model(cam_points)
            loss = F.cross_entropy(pred, target, label_smoothing=cfg.ls, ignore_index=cfg.ignore_index)
        m.update(pred[mask], target[mask])
        loss_meter.update(loss)
        pbar.set_description(f"Val Epoch [{epoch}/{cfg.epochs}] "
                             + f"Loss {loss_meter.avg:.4f} "
                             + f"mACC {m.calc_macc():.4f}")
    acc, macc, miou, iou = m.calc()
    return loss_meter.avg, miou, macc, iou, acc


def main(cfg):
    torch.cuda.set_device(0)
    set_random_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'Config:\n{cfg.__str__()}')

    warmup_loader = DataLoader(
        ScanNetV2(
            dataset_dir=cfg.dataset,
            loop=cfg.batch_size,
            train=True,
            warmup=True,
            voxel_max=cfg.model_cfg.warmup_cfg.voxel_max,
            k=cfg.model_cfg.warmup_cfg.k,
            grid_size=cfg.model_cfg.warmup_cfg.grid_size,
            alpha=cfg.model_cfg.warmup_cfg.alpha,
            batch_size=1,
            cam_opts=cfg.model_cfg.warmup_cfg.cam_opts
        ),
        batch_size=1,
        collate_fn=scannetv2_collate_fn,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    train_loader = DataLoader(
        ScanNetV2(
            dataset_dir=cfg.dataset,
            loop=cfg.train_loop,
            train=True,
            warmup=False,
            voxel_max=cfg.model_cfg.train_cfg.voxel_max,
            k=cfg.model_cfg.train_cfg.k,
            grid_size=cfg.model_cfg.train_cfg.grid_size,
            alpha=cfg.model_cfg.train_cfg.alpha,
            batch_size=cfg.batch_size,
            cam_opts=cfg.model_cfg.train_cfg.cam_opts
        ),
        batch_size=cfg.batch_size,
        collate_fn=scannetv2_collate_fn,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        ScanNetV2(
            dataset_dir=cfg.dataset,
            loop=cfg.val_loop,
            train=False,
            warmup=False,
            voxel_max=cfg.model_cfg.train_cfg.voxel_max,
            k=cfg.model_cfg.train_cfg.k,
            grid_size=cfg.model_cfg.train_cfg.grid_size,
            alpha=cfg.model_cfg.train_cfg.alpha,
            batch_size=1,
            cam_opts=cfg.model_cfg.train_cfg.cam_opts
        ),
        batch_size=1,
        collate_fn=scannetv2_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=cfg.num_workers,
    )

    backbone = CamPointModel(
        **cfg.model_cfg.backbone_cfg,
    ).to('cuda')
    model = SegSemHead(
        backbone=backbone,
        num_classes=cfg.model_cfg.num_classes,
        bn_momentum=cfg.model_cfg.bn_momentum,
    ).to('cuda')
    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    scaler = GradScaler()
    start_epoch = 1
    best_epoch = 0
    best_miou = 0
    if cfg.mode == 'resume':
        cfg.ckpt = cfg.last_ckpt_path
        model_dict = resume_state(model, cfg.ckpt, optimizer=optimizer, scaler=scaler)
        start_epoch = model_dict['last_epoch'] + 1
        best_epoch = model_dict['best_epoch']
        best_miou = model_dict['best_miou']
        logging.info(f"Resume model from {cfg.ckpt}, best_miou={best_miou:.4f}, best_epoch={best_epoch}, start_epoch={start_epoch}")
    if cfg.mode == 'finetune':
        model_dict = load_state(model, cfg.ckpt, optimizer=optimizer, scaler=scaler)
        best_epoch = model_dict['best_epoch']
        best_miou = model_dict['best_miou']
        logging.info(f"Finetune model from {cfg.ckpt}, best_miou={best_miou:.4f}, best_epoch={best_epoch}, start_epoch={start_epoch}")

    steps_per_epoch = len(train_loader)
    scheduler_steps = steps_per_epoch * (start_epoch - 1)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=cfg.epochs * steps_per_epoch,
                                  lr_min=cfg.lr / 10000,
                                  cycle_decay=cfg.lr_decay,
                                  warmup_t=cfg.warmup_epochs * steps_per_epoch,
                                  warmup_lr_init=cfg.lr / 20)

    val_miou, val_macc, val_ious, val_accs = 0., 0., [], []
    macc_when_best = 0.
    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()

    warmup(cfg, model, warmup_loader)
    for epoch in range(start_epoch, cfg.epochs + 1):
        timer.record(f'E{epoch}_start')
        train_loss, train_diff, train_miou, train_macc, train_ious, train_accs, scheduler_steps = train(
            cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps,
        )
        lr = optimizer.param_groups[0]['lr']
        time_cost = timer.record(f'E{epoch}_end')
        timer_meter.update(time_cost)
        logging.info(
            f'@E{epoch} train:    '
            + f'miou={train_miou:.4f} macc={train_macc:.4f} oa={train_accs:.4f} loss={train_loss:.4f} '
            + f'diff={train_diff:.4f} lr={lr:.6f}')

        is_best = False
        if epoch % cfg.val_freq == 0 or epoch >= cfg.epochs:
            with torch.no_grad():
                val_loss, val_miou, val_macc, val_ious, val_accs = validate(
                    cfg, model, val_loader, epoch,
                )
            logging.info(f'@E{epoch} val:      '
                         + f'miou={val_miou:.4f} macc={val_macc:.4f} oa={val_accs:.4f} loss={val_loss:.4f}')
            if val_miou > best_miou:
                logging.info(f'@E{epoch} new best: miou {best_miou:.4f} => {val_miou:.4f}')
                is_best = True
                best_miou = val_miou
                macc_when_best = val_macc
            else:
                logging.info(f'@E{epoch} cur best: miou {best_miou:.4f}')
        if is_best:
            train_info = {
                'miou': train_miou,
                'macc': train_macc,
                'oa': train_accs,
                'loss': train_loss,
                'diff': train_diff,
                'lr': f"{lr:.6f}",
                'time_cost': f"{time_cost:.2f}s",
                'time_cost_avg': f"{timer_meter.avg:.2f}s",
            }
            val_info = {
                'miou': val_miou,
                'macc': val_macc,
                'oa': val_accs,
                'loss': val_loss,
            }
            logging.info(f'@E{epoch} summary:'
                         + f'\ntrain: \n{format_dict(train_info)}'
                         + f'\nval: \n{format_dict(val_info)}'
                         + f'\nious: \n{format_list(ScanNetV2.get_classes(), val_ious)}')
            best_epoch = epoch
            save_state(cfg.best_small_ckpt_path, model=model)
            save_state(cfg.best_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                       best_epoch=best_epoch, last_epoch=epoch, best_miou=best_miou)
        save_state(cfg.last_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                   best_epoch=best_epoch, last_epoch=epoch, best_miou=best_miou)
        if writer is not None and (epoch % cfg.val_freq == 0 or epoch >= cfg.epochs):
            writer.add_scalar('best_miou', best_miou, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('macc_when_best', macc_when_best, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_diff', train_diff, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('time_cost_avg', timer_meter.avg, epoch)
            writer.add_scalar('time_cost', time_cost, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('scannetv2 training')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='scannetv2')
    parser.add_argument('--mode', type=str, required=False, default='train', choices=['train', 'finetune', 'resume'])
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='s',
                        choices=['s', 'l', 'c'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='dataset_link')
    parser.add_argument('--train_loop', type=int, required=False, default=6)
    parser.add_argument('--val_loop', type=int, required=False, default=1)
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--num_workers', type=int, required=False, default=16)
    parser.add_argument("--strain", action='store_true')

    # for train
    parser.add_argument('--epochs', type=int, required=False, default=200)
    parser.add_argument("--warmup_epochs", type=int, required=False, default=20)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--lr_decay", type=float, required=False, default=0)
    parser.add_argument("--decay", type=float, required=False, default=0.05)
    parser.add_argument("--ls", type=float, required=False, default=0.2)
    parser.add_argument("--no_amp", action='store_true')

    # for validate
    parser.add_argument('--val_freq', type=int, required=False, default=1)

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

    if cfg.mode == 'finetune':
        assert cfg.ckpt != ''
    cfg.use_amp = not cfg.no_amp

    # scannetv2
    cfg.num_classes = 20
    cfg.ignore_index = 20

    prepare_exp(cfg)

    if cfg.strain:
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
    main(cfg)
