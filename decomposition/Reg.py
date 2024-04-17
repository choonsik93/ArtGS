import torch
import torch.nn as nn
import numpy as np


class DeformSymLoss(nn.Module):
    def __init__(self, diag_weight=1.0, upper_weight=1.0, norm_degree=1):
        super(DeformSymLoss, self).__init__()
        self.diag_weight = diag_weight
        self.upper_weight = upper_weight
        if norm_degree == 1:
            self.l1_loss = torch.nn.L1Loss()
        elif norm_degree == 2:
            self.l1_loss = torch.nn.MSELoss()

    def forward(self, model):
        deform_diag = model.deform_plane[0][0, :, range(model.time_grid), range(model.time_grid)].transpose(1, 0)
        deform_diag = model.deform_feature_out(deform_diag).reshape(model.time_grid, -1, 7)

        gt_deform = torch.zeros_like(deform_diag)
        gt_deform[:, :, 0] = 1.0
        deform_loss = self.l1_loss(deform_diag - gt_deform, torch.zeros_like(deform_diag))
        total_loss = self.diag_weight * deform_loss

        return total_loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight_dim1=1.0, TVLoss_weight_dim2=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight_dim1 = TVLoss_weight_dim1
        self.TVLoss_weight_dim2 = TVLoss_weight_dim2

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = (
            torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :].detach()), 2).sum()
            * self.TVLoss_weight_dim1
        )
        w_tv = (
            torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1].detach()), 2).sum()
            * self.TVLoss_weight_dim2
        )
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]