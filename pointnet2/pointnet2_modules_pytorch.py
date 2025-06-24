import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from typing import List

import pytorch_utils as pt_utils

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist = -2 * torch.bmm(src, dst.permute(0, 2, 1)) # torch.Size([1, 2048, 4090])
    src_sum = torch.sum(src ** 2, -1).reshape(B, N, 1).repeat(1, 1, dist.shape[2]) # torch.Size([1, 2048, 4090])
    dist += src_sum
    # dst_sum = torch.sum(dst ** 2, -1).reshape(B, 1, M).repeat(1, dist.shape[1], 1)
    dst_sum = torch.sum(dst ** 2, -1) # torch.Size([1, 4090])
    dst_sum = dst_sum.reshape(B, M, 1) # torch.Size([1, 4090, 1])
    dst_sum = dst_sum.repeat(1, 1, dist.shape[1]) # torch.Size([1, 4090, 2048])
    dist = dist.permute(0, 2, 1)

    dist += dst_sum
    dist = dist.permute(0, 2, 1)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # batch_indices = torch.arange(B, dtype=torch.long).to(device).reshape(view_shape).repeat(repeat_shape)
    batch_indices = torch.range(0, B-1, dtype=torch.int32).to(device).reshape(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.int(), :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.int32).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.int32).to(device)
    # batch_indices = torch.arange(B, dtype=torch.long).to(device)
    batch_indices = torch.range(0, B-1, dtype=torch.int32).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    group_idx = torch.range(0, N-1, dtype=torch.int32).to(device).reshape(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)

    # group_idx[sqrdists > radius ** 2] = N

    # mask = sqrdists > radius ** 2
    # not_mask = torch.sub(0, mask.int()) + 1

    iden_conv = IdentityConv(1).to(device)
    radius_sq = radius ** 2
    mask = (((sqrdists - radius_sq).sign() + 1)) // 2
    not_mask = 1 - mask
    # mask = iden_conv(mask.unsqueeze(1)).squeeze(1)
    # not_mask = iden_conv(not_mask.unsqueeze(1)).squeeze(1)

    group_idx = (not_mask * group_idx + mask * N) # TODO()维度太大不支持的问题
    # group_idx = group_idx.int() # int32

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # group_idx = torch.topk(group_idx.long(), group_idx.shape[2], dim=-1, largest=False).values[:, :, :nsample]

    group_first = group_idx[:, :, 0].reshape(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N

    # group_idx[mask] = group_first[mask]
    not_mask = torch.sub(0, mask.int()) + 1
    group_idx = not_mask * group_idx + mask * group_first

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    fps_idx = torch.from_numpy(np.random.choice(xyz.shape[1], npoint, replace=False)).unsqueeze(0).int()
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.reshape(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.reshape(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.reshape(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstractionVotes(nn.Module):
    def __init__(self,
                 *,
                 mlp: List[int],
                 npoint: int = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 group_all: bool = False,
                 use_xyz: bool = True,
                 pooling: str = 'max',
                 sigma: float = None, # for RBF pooling
                 normalize_xyz: bool = False, # noramlize local XYZ with radius
                 sample_uniformly: bool = False,
                 ret_unique_cnt: bool = False):
        super(PointNetSetAbstractionVotes, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.pooling = pooling
        self.mlp_module = None

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor,
                points: torch.Tensor = None,
                inds: torch.Tensor = None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        
        xyz -- xyz
        point - feature
        indice - farthest point sample
        """

        if points is not None:
            points = points.permute(0, 2, 1)

        # if self.group_all:
        #     new_xyz, new_points = sample_and_group_all(xyz, points)
        # else:
        #     new_xyz, new_points, _, inds = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, returnfps=True)
        new_xyz, new_points, _, inds = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, returnfps=True)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 1, 2) # [B, C+D, nsample,npoint]
        new_features = self.mlp_module(
            new_points
        )  # (B, mlp[-1], npoint, nsample)

        if self.pooling == 'max':
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)

        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        return new_xyz, new_features, inds

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, *, mlp: List[int], bn: bool = True):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, mlp[-1], N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)

            dists, idx = dists.sort(dim=-1)
            # dists, idx = torch.topk(dists, dists.shape[2], dim=-1, largest=False)

            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            ex_min = torch.from_numpy(np.array(1e-8)).float().to(dists.device).repeat(dists.shape[0], dists.shape[1], dists.shape[2])
            dist_recip = 1.0 / (dists + ex_min)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx.int()) * weight.reshape(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1).unsqueeze(-1)
        new_features = self.mlp(new_points)
        return new_features.squeeze(-1)

class IdentityUnitConv(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.identity_conv = torch.nn.Conv2d(   
            channels, channels, 1, groups=channels, bias=False).cuda()
        torch.nn.init.dirac_(
            self.identity_conv.weight.data, groups=channels)
        self.check_equal()

    def check_equal(self):
        random_data = torch.randn(1, self.channels, 32, 32).cuda()
        result = self.forward(random_data)
        np.testing.assert_allclose(
            random_data.cpu().detach().numpy(), result.cpu().detach().numpy(),
            rtol=1e-02, atol=1e-03)
        print("check Identity, pass!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity with 4-D dataflow, input == output."""
        return self.identity_conv(x)

class IdentityConv(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.identity_conv = torch.nn.Conv2d(   
            channels, channels, 1, stride=1, padding=0, groups=channels, bias=True)
        with torch.no_grad():
            self.identity_conv.weight = torch.nn.Parameter(torch.eye(channels).view(channels, channels, 1, 1))
            self.identity_conv.bias = torch.nn.Parameter(torch.zeros(channels))
        self.check_equal()

    def check_equal(self):
        random_data = torch.randn(1, self.channels, 32, 32)
        result = self.forward(random_data)
        np.testing.assert_allclose(
            random_data.cpu().detach().numpy(), result.cpu().detach().numpy(),
            rtol=1e-02, atol=1e-03)
        print("check Identity, pass!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity with 4-D dataflow, input == output."""
        return self.identity_conv(x)
