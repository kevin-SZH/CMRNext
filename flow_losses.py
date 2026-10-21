"""RAFT flow supervision and uncertainty losses.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def NegativeLogLikelihoodLoss(predicted_flow, predicted_uncertainty, target_flow, mask):
    """Compute the negative log likelihood loss for flow uncertainty.

    Args:
        predicted_flow: Predicted flow.
        predicted_uncertainty: Predicted uncertainty.
        target_flow: Target flow.
        mask: Validity mask; nonzero entries participate in the operation.
    """
    u_component_mean = predicted_flow[:, 0, :, :]
    u_component_sigma = predicted_uncertainty[:, 0, :, :]
    u_component_distribution = torch.distributions.normal.Normal(u_component_mean, u_component_sigma)
    u_component_loss = -u_component_distribution.log_prob(target_flow[:, 0, :, :])
    u_component_loss = u_component_loss * mask

    v_component_mean = predicted_flow[:, 1, :, :]
    v_component_sigma = predicted_uncertainty[:, 1, :, :]
    v_component_distribution = torch.distributions.normal.Normal(v_component_mean, v_component_sigma)
    v_component_loss = -v_component_distribution.log_prob(target_flow[:, 1, :, :])
    v_component_loss = v_component_loss * mask

    return u_component_loss.sum() / mask.sum() + v_component_loss.sum() / mask.sum()


def RAFT_loss2(flow_preds, uncertainty_preds, flow_gt, valid, upsample=True, gamma=0.8, weight_nll=-1.0,
               unc_type="DER"):
    """Compute weighted multi-iteration RAFT flow supervision over valid correspondences.

    Args:
        flow_preds: Flow preds.
        uncertainty_preds: Uncertainty preds.
        flow_gt: Flow gt.
        valid: Valid.
        upsample: Upsample.
        gamma: Gamma.
        weight_nll: Weight nll.
        unc_type: Unc type.
    """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # Exclude invalid pixels and extreme displacements.
    valid = valid[:, 0]
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 1000)
    up_preds = []

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        if upsample:
            new_size = (4 * flow_preds[i].shape[2], 4 * flow_preds[i].shape[3])
            up_preds.append(4 * F.interpolate(flow_preds[i], size=new_size, mode='bilinear', align_corners=True))
            i_loss = (up_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        elif uncertainty_preds[i] is None:
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        else:
            if weight_nll == -1.0:
                if unc_type == "NLL":
                    i_loss = NegativeLogLikelihoodLoss(flow_preds[i], uncertainty_preds[i], flow_gt, valid)
            else:
                if unc_type == "NLL":
                    i_loss = NegativeLogLikelihoodLoss(flow_preds[i], uncertainty_preds[i], flow_gt, valid) * weight_nll
                i_loss += (valid[:, None] * (flow_preds[i] - flow_gt).abs()).mean()
            flow_loss += i_weight * i_loss

    if upsample:
        epe = torch.sum((up_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    else:
        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'f1': ((epe > 3.0) & ((epe / mag.view(-1)[valid.view(-1)]) > 0.05)).float().mean() * 100.
    }

    return flow_loss, metrics
