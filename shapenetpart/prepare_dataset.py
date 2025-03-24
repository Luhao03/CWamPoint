import __init__

import argparse
import os

import torch
from pointnet2_ops import pointnet2_utils

from shapenetpart.configs import model_configs
from shapenetpart.dataset import ShapeNetPartNormal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=False, default='dataset_link')
    parser.add_argument('-o', type=str, required=False, default='shapenetpart_presample.pt')

    args, opts = parser.parse_known_args()

    presample_path = os.path.join(args.i, args.o)
    model_cfg = model_configs['s']
    testset = ShapeNetPartNormal(
            dataset_dir=args.i,
            presample=True,
            train=False,
            voxel_max=model_cfg.train_cfg.voxel_max,
            k=model_cfg.train_cfg.k,
            n_samples=model_cfg.train_cfg.n_samples,
            alpha=model_cfg.train_cfg.alpha,
            batch_size=1,
            cam_opts=model_cfg.train_cfg.cam_opts,
        )

    cnt = 0
    xyzs = []
    shapes = []
    segs = []
    normals = []
    for cam_points, shape in testset:
        cam_points.to_cuda(False)
        xyz, normal, seg = cam_points.p, cam_points.f, cam_points.y
        xyz = xyz.float().unsqueeze(0)
        idx = pointnet2_utils.furthest_point_sample(xyz, 2048).long()
        xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        xyz = xyz.cpu()
        idx = idx.cpu()
        normal = normal.cpu()
        seg = seg.cpu()
        xyzs.append(xyz)
        shapes.append(shape)
        seg = seg.long()[idx.squeeze()]
        segs.append(seg)
        normal = normal.float().unsqueeze(0)
        normal = torch.gather(normal, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        normals.append(normal)
        cnt += 1
        if cnt % 250 == 0:
            print(f"{cnt * 100 / len(testset):.1f}% done")

    xyzs = torch.cat(xyzs)
    shapes = torch.tensor(shapes, dtype=torch.int64)
    segs = torch.cat(segs).view(len(testset), -1)
    normals = torch.cat(normals)
    print(xyzs.shape, normals.shape, shapes.shape, segs.shape)
    torch.save((xyzs, normals, shapes, segs), presample_path)
    print(presample_path)
