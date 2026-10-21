"""Geometry, projection, sparse flow, batching, and logging utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Bilinear sampler.

    Args:
        img: Img.
        coords: Coords.
        mode: Mode.
        mask: Validity mask; nonzero entries participate in the operation.
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def forward_interpolate(flow):
    """Forward interpolate.

    Args:
        flow: Flow.
    """
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='cubic', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='cubic', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def coords_grid(batch, ht, wd):
    """Coords grid.

    Args:
        batch: Batch.
        ht: Ht.
        wd: Wd.
    """
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    """Upflow8.

    Args:
        flow: Flow.
        mode: Mode.

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def upflow2(flow, mode='bilinear'):
    """Upflow2.

    Args:
        flow: Flow.
        mode: Mode.

    Geometry:
        Extrinsics map LiDAR column vectors to the camera frame, with translation
        in meters. The left residual is delta = Ti @ inverse(T0), so delta @ T0 = Ti.
    """
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return  2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
