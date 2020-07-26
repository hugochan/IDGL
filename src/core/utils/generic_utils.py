import yaml
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import VERY_SMALL_NUMBER, INF


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x

def create_mask(x, N, device=None):
    x = x.data
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[i, :x[i]] = 1
    return to_cuda(torch.Tensor(mask), device)

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


def batch_normalize_adj(mx, mask=None):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    # mx: shape: [batch_size, N, N]

    rowsum = torch.clamp(mx.sum(1), min=VERY_SMALL_NUMBER)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    if mask is not None:
        r_inv_sqrt = r_inv_sqrt * mask

    r_mat_inv_sqrt = []
    for i in range(r_inv_sqrt.size(0)):
        r_mat_inv_sqrt.append(torch.diag(r_inv_sqrt[i]))

    r_mat_inv_sqrt = torch.stack(r_mat_inv_sqrt, 0)
    return torch.matmul(torch.matmul(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
