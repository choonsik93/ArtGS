import numpy as np
import math
import torch
import torch.nn as nn
#import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from .visualizer_trimesh import TestVisualizer
#from .visualizer_open3d import TestVisualizer
from .Reg import TVLoss, DeformSymLoss
import pytorch3d
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.points_to_volumes import add_points_features_to_volume_densities_features
from pytorch3d.loss import chamfer_distance
import os
#import scene.tools as tools


class VoxelDataset(Dataset):
    def __init__(self, time_grid, one_source=True, source_idx=None):
        self.batch_index_list = []
        if one_source:
            for i in range(time_grid):
                if source_idx is None:
                    self.batch_index_list.append([time_grid // 2, i])
                else:
                    self.batch_index_list.append([source_idx, i])
        else:
            for i in range(time_grid):
                for j in range(time_grid):
                    self.batch_index_list.append([i, j])

    def __getitem__(self, index):
        return self.batch_index_list[index]

    def __len__(self):
        return len(self.batch_index_list)


class Sampler3D:
    def __init__(
        self,
        model,
        cfg, 
        model_path="",
        device="cuda",
    ):
        self.model = model
        self.cfg = cfg
        self.n_iters = cfg.optim.n_iters
        self.step = 0
        self.logfolder = os.path.join(model_path, cfg.optim.logfolder)
        self.batch_size = cfg.optim.batch_size
        self.one_source = cfg.optim.one_source

        self.lr_decay_type = cfg.optim.lr_decay_type
        self.lr_decay_step = cfg.optim.n_iters if cfg.optim.lr_decay_step == -1 else cfg.optim.lr_decay_step
        self.lr_decay_target_ratio = cfg.optim.lr_decay_target_ratio
        self.device = device
        
        self.sym_loss_weight = cfg.optim.sym_loss_weight
        self.tv_deform_loss_weight = cfg.optim.tv_deform_loss_weight
        self.tv_label_loss_weight = cfg.optim.tv_label_loss_weight
        self.recon_loss_weight = cfg.optim.recon_loss_weight
        self.diag_loss_weight = cfg.optim.diag_loss_weight
        self.nst_loss_weight = cfg.optim.nst_loss_weight
        self.num_nst_points = cfg.optim.num_nst_points
        self.init_loss_metric()

        self.shrink_num_thresold = cfg.optim.shrink_num_thresold
        self.group_merge_threshold = cfg.optim.group_merge_threshold 

        self.gumbel = cfg.optim.gumbel
        self.hard = cfg.optim.hard
        self.tau = cfg.optim.tau
        self.eval = cfg.optim.eval
        self.turn_to_softmax = cfg.optim.turn_to_softmax

        self.loss_mode = cfg.optim.loss_mode
        self.adj_loss_version = cfg.optim.adj_loss_version

        self.vis = cfg.optim.vis
        self.vis_every = cfg.optim.vis_every
        if self.vis is True:
            vis_time_step = cfg.optim.vis_time_step
            deform_arg = self.deform_arg
            self.visualizer = TestVisualizer(self, time_step=vis_time_step, folder_name=self.logfolder,
                                             deform_arg=deform_arg, one_source=self.one_source, vis_seg=True)

    @torch.no_grad()
    def init_emptiness_from_input(self, dense_xyz, emptiness_all, time_set, source_idx=None):

        self.time_set = time_set
        self.xyzs = dense_xyz
        self.emptiness_all = emptiness_all

        self.num_data = emptiness_all.shape[0]
        if source_idx is None:
            self.source_idx = self.num_data // 2
        else:
            self.source_idx = source_idx
        self.batch_data = VoxelDataset(self.num_data, one_source=self.one_source, source_idx=source_idx)
        self.dataloader = DataLoader(self.batch_data, batch_size=self.batch_size, shuffle=True)

        self.source_nst_idx = []
        for i in range(self.num_data):
            emptiness_i = self.emptiness_all[i]
            xyzs_i = self.xyzs[emptiness_i]
            _, nst_idx, _ = knn_points(xyzs_i[None, ...], xyzs_i[None, ...], K=self.num_nst_points)
            self.source_nst_idx.append(nst_idx[0][:, :])

    """@torch.no_grad()
    def init_gaussians_from_input(self, gaussians_all, time_set, source_idx=None):
        self.time_set = time_set
        self.num_data = gaussians_all.shape[0]
        if source_idx is None:
            self.source_idx = self.num_data // 2
        else:
            self.source_idx = source_idx
        self.batch_data = VoxelDataset(self.num_data, one_source=self.one_source, source_idx=source_idx)
        self.dataloader = DataLoader(self.batch_data, batch_size=self.batch_size, shuffle=True)

    @torch.no_grad()
    def set_gaussian_source(self, gaussian_source):
        self.gaussian_source = gaussian_source
        _, nst_idx, _ = knn_points(gaussian_source[None, ...], gaussian_source[None, ...], K=self.num_nst_points)
        self.gaussian_source_nst_idx = nst_idx[0]"""

    def get_lr_decay_factor(self, step):
        """
        Calculate the learning rate decay factor = current_lr / initial_lr.
        """
        lr_decay_type = self.lr_decay_type
        lr_decay_step = self.lr_decay_step
        lr_decay_target_ratio = self.lr_decay_target_ratio

        if lr_decay_type == "exp":  # exponential decay
            lr_factor = lr_decay_target_ratio ** (
                step / lr_decay_step
            )
        elif lr_decay_type == "linear":  # linear decay
            lr_factor = lr_decay_target_ratio + (
                1 - lr_decay_target_ratio
            ) * (1 - step / lr_decay_step)
        elif lr_decay_type == "cosine":  # consine decay
            lr_factor = lr_decay_target_ratio + (
                1 - lr_decay_target_ratio
            ) * 0.5 * (1 + math.cos(math.pi * step / lr_decay_step))

        return max(lr_decay_target_ratio, lr_factor)
    
    @property
    def lr_factor(self):
        return self.get_lr_decay_factor(self.step)
    
    @property
    def deform_arg(self):
        deform_arg = {'gumbel': self.gumbel, 'hard': self.hard, 'tau': self.tau * self.lr_factor, 'eval': self.eval}
        """if self.step > self.turn_to_softmax:
            deform_arg = {'gumbel': False, 'hard': self.hard, 'tau': self.tau, 'eval': self.eval}
            self.model.deform_arg = deform_arg"""
        return deform_arg

    def init_loss_metric(self):
        self.symreg = DeformSymLoss(diag_weight=1.0, upper_weight=0.0, norm_degree=2)
        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()
        self.cross_loss = nn.CrossEntropyLoss()
        self.tvreg_s = TVLoss()
        self.tvreg_s_t = TVLoss(1.0, 0.0)

    def raw_label_to_vector(self, label, deform_arg=None):
        deform_arg = self.deform_arg if deform_arg is None else deform_arg
        if deform_arg["eval"] is True:
            _, label_ind = torch.max(label, 1)
            label = torch.eye(label.shape[-1], dtype=label.dtype, device=label.device)[label_ind]
        else:
            if deform_arg["gumbel"]:
                label = torch.nn.functional.gumbel_softmax(label, tau=deform_arg["tau"], hard=deform_arg["hard"])
            else:
                label = torch.nn.functional.softmax(label / deform_arg["tau"], dim=-1)
        return label

    @torch.no_grad()
    def group_merge(self):
        frame_time_source = self.time_set[self.source_idx]
        frame_time_all = self.time_set
        _, quaternion, translation = self.model.compute_deformfeature(frame_time_source.expand(frame_time_all.shape[0]),
                                                                      frame_time_all)
        deform_all = torch.cat((quaternion, translation), -1)
        K = deform_all.shape[1]
        deform_all = deform_all.transpose(0, 1).reshape(K, -1) # [K, T*7]
        dist_mtx = deform_all[:, None, :] - deform_all[None, :, :] # [K, K, T*7]
        dist_mtx = dist_mtx.reshape(K, K, -1, 7)
        dist_mtx = torch.sum(dist_mtx ** 2, -1) # [K, K, T]
        dist_mtx = dist_mtx.detach().cpu().numpy()
        dist_mtx = np.max(dist_mtx, -1)
        ind_0s, ind_1s = np.unravel_index(np.argsort(dist_mtx, axis=None), dist_mtx.shape)
        ind_0, ind_1 = ind_0s[K], ind_1s[K]
        print(dist_mtx[ind_0][ind_1])
        if dist_mtx[ind_0][ind_1] < self.group_merge_threshold:
            print("shrink %d to %d"%(K, K - 1))
            part_shrink_matrix = torch.zeros((K, K - 1), device=self.device)
            deform_shrink_matrix = torch.zeros((K, K - 1), device=self.device)
            count = 0
            for i in range(K):
                if i != ind_0 and i != ind_1:
                    part_shrink_matrix[i, count] = 1
                    deform_shrink_matrix[i, count] = 1
                    count += 1
            part_shrink_matrix[ind_0, -1] = 1
            part_shrink_matrix[ind_1, -1] = 1

            emptiness_source = self.emptiness_all[self.source_idx]
            xyzs_source = self.xyzs[emptiness_source, :]
            time_source = self.time_set[self.source_idx]
            frame_time_source = time_source * torch.ones((xyzs_source.shape[0], 1), device=self.device) # [N, 1]
            label_raw = self.model.compute_labelfeature(xyzs_source, frame_time_source) # [N, K]
            _, label_ind = torch.max(label_raw, 1)
            debug_label = torch.eye(label_raw.shape[-1], dtype=label_raw.dtype, device=label_raw.device)[label_ind]
            count_label = torch.sum(debug_label, 0)
            if count_label[ind_0] > count_label[ind_1]:
                deform_shrink_matrix[ind_0, -1] = 1
            else:
                deform_shrink_matrix[ind_1, -1] = 1

            deform_weight = self.model.deform_feature_out[-1].weight.data.transpose(0, 1)
            deform_bias = self.model.deform_feature_out[-1].bias.data
            label_weight = self.model.label_feature_out[-1].weight.data.transpose(0, 1)
            label_bias = self.model.label_feature_out[-1].bias.data
            
            label_weight = torch.matmul(label_weight, part_shrink_matrix)
            label_bias = torch.matmul(label_bias[None, :], part_shrink_matrix)[0]
            deform_weight = torch.matmul(deform_weight.reshape(-1, K, 7).transpose(1, 2), deform_shrink_matrix).transpose(1, 2)
            deform_weight = deform_weight.reshape(deform_weight.shape[0], -1)
            deform_bias = torch.matmul(deform_bias.reshape(K, 7).transpose(0, 1), deform_shrink_matrix).transpose(0, 1)
            deform_bias = deform_bias.reshape(-1)

            self.model.deform_feature_out[-1].weight = nn.Parameter(deform_weight.transpose(0, 1))
            self.model.deform_feature_out[-1].bias = nn.Parameter(deform_bias)
            self.model.label_feature_out[-1].weight = nn.Parameter(label_weight.transpose(0, 1))
            self.model.label_feature_out[-1].bias = nn.Parameter(label_bias)

            self.group_merge()
        else:
            return
        
    @torch.no_grad()
    def group_merge_v1(self):
        frame_time_source = self.time_set[self.source_idx]
        frame_time_all = self.time_set
        _, quaternion, translation = self.model.compute_deformfeature(frame_time_source.expand(frame_time_all.shape[0]),
                                                                      frame_time_all)
        deform_all = torch.cat((quaternion, translation), -1)
        K = deform_all.shape[1]
        deform_all = deform_all.transpose(0, 1).reshape(K, -1) # [K, T*7]
        dist_mtx = deform_all[:, None, :] - deform_all[None, :, :] # [K, K, T*7]
        dist_mtx = torch.sum(dist_mtx ** 2, -1) / frame_time_all.shape[0] #[K, K]
        dist_mtx = dist_mtx.detach().cpu().numpy()
        ind_0s, ind_1s = np.unravel_index(np.argsort(dist_mtx, axis=None), dist_mtx.shape)
        ind_0, ind_1 = ind_0s[K], ind_1s[K]
        if dist_mtx[ind_0][ind_1] < self.group_merge_threshold:
            print("shrink %d to %d"%(K, K - 1))
            part_shrink_matrix = torch.zeros((K, K - 1), device=self.device)
            deform_shrink_matrix = torch.zeros((K, K - 1), device=self.device)
            count = 0
            for i in range(K):
                if i != ind_0 and i != ind_1:
                    part_shrink_matrix[i, count] = 1
                    deform_shrink_matrix[i, count] = 1
                    count += 1
            part_shrink_matrix[ind_0, -1] = 1
            part_shrink_matrix[ind_1, -1] = 1

            emptiness_source = self.emptiness_all[self.source_idx]
            xyzs_source = self.xyzs[emptiness_source, :]
            time_source = self.time_set[self.source_idx]
            frame_time_source = time_source * torch.ones((xyzs_source.shape[0], 1), device=self.device) # [N, 1]
            label_raw = self.model.compute_labelfeature(xyzs_source, frame_time_source) # [N, K]
            _, label_ind = torch.max(label_raw, 1)
            debug_label = torch.eye(label_raw.shape[-1], dtype=label_raw.dtype, device=label_raw.device)[label_ind]
            count_label = torch.sum(debug_label, 0)
            if count_label[ind_0] > count_label[ind_1]:
                deform_shrink_matrix[ind_0, -1] = 1
            else:
                deform_shrink_matrix[ind_1, -1] = 1

            deform_weight = self.model.deform_basis_mat.weight.data.transpose(0, 1)
            deform_weight_shape = deform_weight.shape
            label_weight = self.model.label_basis_mat.weight.data.transpose(0, 1)
            
            label_weight = torch.matmul(label_weight, part_shrink_matrix)
            deform_weight = torch.matmul(deform_weight.reshape(deform_weight_shape[0], -1, 7).transpose(1, 2), deform_shrink_matrix).transpose(1, 2)
            deform_weight = deform_weight.reshape(deform_weight_shape[0], -1)

            self.model.deform_basis_mat.weight = nn.Parameter(deform_weight.transpose(0, 1))
            self.model.label_basis_mat.weight = nn.Parameter(label_weight.transpose(0, 1))

            self.group_merge()
        else:
            return

    @torch.no_grad()
    def shrink_label(self, buffer=1):
        emptiness_source = self.emptiness_all[self.source_idx]
        xyzs_source = self.xyzs[emptiness_source, :]
        time_source = self.time_set[self.source_idx]
        frame_time_source = time_source * torch.ones((xyzs_source.shape[0], 1), device=self.device) # [N, 1]
        label_raw = self.model.compute_labelfeature(xyzs_source, frame_time_source) # [N, K]
        _, label_ind = torch.max(label_raw, 1)

        prev_label_dim = label_raw.shape[1]

        debug_label = torch.eye(prev_label_dim, dtype=label_raw.dtype, device=label_raw.device)[label_ind]
        print(torch.sum(debug_label, 0))
        unique_label = torch.argwhere(torch.sum(debug_label, 0) > self.shrink_num_thresold)
        unique_label = unique_label.squeeze()
        if label_raw.shape[-1] == unique_label.shape[0]: 
            return
        part_shrink_matrix = torch.zeros((prev_label_dim, (unique_label.shape[0] + buffer)), device=self.device)
        print("shrink size: ", part_shrink_matrix.shape)
        for i in range(unique_label.shape[0]):
            part_shrink_matrix[unique_label[i], i] = 1

        deform_weight = self.model.deform_feature_out[-1].weight.data.transpose(0, 1)
        deform_bias = self.model.deform_feature_out[-1].bias.data
        label_weight = self.model.label_feature_out[-1].weight.data.transpose(0, 1)
        label_bias = self.model.label_feature_out[-1].bias.data
        
        label_weight = torch.matmul(label_weight, part_shrink_matrix)
        label_bias = torch.matmul(label_bias[None, :], part_shrink_matrix)[0]
        deform_weight = torch.matmul(deform_weight.reshape(-1, prev_label_dim, 7).transpose(1, 2), part_shrink_matrix).transpose(1, 2)
        deform_weight = deform_weight.reshape(deform_weight.shape[0], -1)
        deform_bias = torch.matmul(deform_bias.reshape(prev_label_dim, 7).transpose(0, 1), part_shrink_matrix).transpose(0, 1)
        deform_bias = deform_bias.reshape(-1)

        self.model.deform_feature_out[-1].weight = nn.Parameter(deform_weight.transpose(0, 1))
        self.model.deform_feature_out[-1].bias = nn.Parameter(deform_bias)
        self.model.label_feature_out[-1].weight = nn.Parameter(label_weight.transpose(0, 1))
        self.model.label_feature_out[-1].bias = nn.Parameter(label_bias)
    
    def get_sym_loss(self, deform):
        gt_deform = torch.zeros_like(deform)
        gt_deform[:, :, 0] = 1.0
        sym_loss = self.l1_loss(deform - gt_deform, torch.zeros_like(deform))
        return sym_loss
    
    def get_tv_loss(self):
        loss_tv = self.tv_deform_loss_weight * self.lr_factor * self.model.TV_loss_deform(self.tvreg_s)
        loss_tv += self.tv_label_loss_weight * self.lr_factor * self.model.TV_loss_label(self.tvreg_s)
        return loss_tv
        
    def get_loss(self, batch_idxs=None):

        model = self.model
        decay_factor = self.lr_factor

        """if self.step == self.turn_to_softmax:
            self.gumbel = False
            self.hard = False
            print(self.deform_arg)
            #self.loss_mode = "chamfer"
            model.deform_arg = self.deform_arg
            if self.vis:
                self.visualizer.deform_arg = self.deform_arg"""

        batch_idxs = next(iter(self.dataloader)) if batch_idxs is None else batch_idxs
        source_idxs = batch_idxs[0]
        target_idxs = batch_idxs[1]

        frame_time_source, frame_time_target = self.time_set[source_idxs], self.time_set[target_idxs]
        emptiness_source = self.emptiness_all[self.source_idx]
        emptiness_target = self.emptiness_all[target_idxs]
        xyzs_source = self.xyzs[emptiness_source, :]
        deform, quaternion, translation = model.compute_deformfeature(frame_time_source, frame_time_target)

        time_temp_input = frame_time_source[0] * torch.ones((xyzs_source.shape[0], 1), device=self.device) # [N, 1]
        label_raw = model.compute_labelfeature(xyzs_source, time_temp_input) # [N, K]
        valid_labels = self.raw_label_to_vector(label_raw) # [N, K]

        batch_time_target = frame_time_target[0] * torch.ones((xyzs_source.shape[0], 1), device=self.device) # [N, 1]
        soft_deform = model.compute_soft_deformfeature(xyzs_source, batch_time_target)

        if self.loss_mode == "voxel":
            xyzs_source_deform = pytorch3d.transforms.quaternion_apply(quaternion[:, :, None, :], xyzs_source[None, None, :, :]) \
                + translation[:, :, None, :] # [B, K, N, 3]
        
            B = self.batch_size
            K = quaternion.shape[1]
            N = xyzs_source.shape[0]
            grid_shape = emptiness_source.shape

            points_3d = xyzs_source_deform # [B, K, N, 3]
            points_3d = points_3d + soft_deform[None, None, :, :]
            points_3d = points_3d.reshape(B*K, N, 3)
            points_features = valid_labels.transpose(0, 1).unsqueeze(0).repeat(B, 1, 1).reshape(B*K, N, 1) # [B*K, N, 1]
            volume_densities = torch.zeros((B*K, 1, *grid_shape), device=model.device)
            volume_features = torch.zeros((B*K, 1, *grid_shape), device=model.device)
            add_points_features_to_volume_densities_features(points_3d, points_features, volume_densities, volume_features, rescale_features=False)

            emptiness_deformed = volume_features.reshape(B, K, *grid_shape)
            emptiness_deformed = torch.sum(emptiness_deformed, 1).transpose(1, 3) # [B, G, G, G]
            density_deformed = volume_densities.reshape(B, K, *grid_shape)
            density_deformed = torch.sum(density_deformed, 1).transpose(1, 3) # [B, G, G, G]
            
            valid_idx = density_deformed.bool() | emptiness_target
            #loss_3d = self.l2_loss(emptiness_deformed[valid_idx], emptiness_target.float()[valid_idx])
            loss_3d = self.l1_loss(emptiness_deformed[valid_idx], emptiness_target.float()[valid_idx])
            """if self.step < 7000:
                loss_3d = self.l1_loss(emptiness_deformed[valid_idx], emptiness_target.float()[valid_idx])
            else:
                loss_3d = self.l2_loss(emptiness_deformed[valid_idx], emptiness_target.float()[valid_idx])"""
            """if self.step < 5000:
                loss_3d = self.l1_loss(emptiness_deformed, emptiness_target.float())
            else:
                loss_3d = self.l1_loss(emptiness_deformed, emptiness_target.float())"""
            total_loss = loss_3d
        elif self.loss_mode == "merge_voxel":
            B = self.batch_size
            K = quaternion.shape[1]
            N = xyzs_source.shape[0]
            grid_shape = emptiness_source.shape

            quaternion_concat = torch.sum(valid_labels[None, :, :, None] * quaternion[:, None, :, :], 2) # [:, N, K, :] x [B, :, K, 4] -> [B, N, 4]
            translation_concat = torch.sum(valid_labels[None, :, :, None] * translation[:, None, :, :], 2)
            xyzs_source_deform = pytorch3d.transforms.quaternion_apply(quaternion_concat, xyzs_source[None, :, :]) + translation_concat

            points_3d = xyzs_source_deform # [B, N, 3]
            points_features = torch.ones_like(points_3d[:, :, 0:1]) # [B, N, 1]
            volume_densities = torch.zeros((B, 1, *grid_shape), device=model.device)
            volume_features = torch.zeros((B, 1, *grid_shape), device=model.device)
            add_points_features_to_volume_densities_features(points_3d, points_features, volume_densities, volume_features, rescale_features=False)

            emptiness_deformed = volume_features.reshape(B, *grid_shape).transpose(1, 3) # [B, G, G, G]
            density_deformed = volume_densities.reshape(B, *grid_shape).transpose(1, 3) # [B, G, G, G]
            
            valid_idx = density_deformed.bool() | emptiness_target
            loss_3d = self.l2_loss(emptiness_deformed[valid_idx], emptiness_target.float()[valid_idx])
            total_loss = loss_3d
        elif self.loss_mode == "chamfer":
            B = self.batch_size
            K = quaternion.shape[1]
            N = xyzs_source.shape[0]
            loss_3d = 0.0
            quaternion_concat = torch.sum(valid_labels[None, :, :, None] * quaternion[:, None, :, :], 2) # [:, N, K, :] x [B, :, K, 4] -> [B, N, 4]
            translation_concat = torch.sum(valid_labels[None, :, :, None] * translation[:, None, :, :], 2)
            xyzs_source_deform = pytorch3d.transforms.quaternion_apply(quaternion_concat, xyzs_source[None, :, :]) + translation_concat
            for i in range(self.batch_size):
                xyzs_target = self.xyzs[emptiness_target[i]]
                loss, _ = chamfer_distance(xyzs_source_deform[i].unsqueeze(0), xyzs_target.unsqueeze(0))
                loss_3d += loss
            total_loss = loss_3d
        else:
            raise ValueError('Invalid 3D Loss type')

        if self.adj_loss_version == 4: # and self.step > 5000: #and self.step > self.n_iters * 0.5: #and self.step > self.n_iters * 0.3:
            nst_idx_flat = self.source_nst_idx[self.source_idx].reshape(-1) # [N, nst_idx] -> [N*nst_idx]
            max_idx = torch.argmax(label_raw[nst_idx_flat, :], -1, keepdim=True).detach()
            nst_label_target = torch.eye(label_raw.shape[-1], dtype=label_raw.dtype, device=label_raw.device)[max_idx] # [N*nst_idx, K]
            nst_label_target = nst_label_target.reshape(N, self.num_nst_points, K)
            nst_label_target = torch.mean(nst_label_target, 1) # [N, K]
            nst_label_target = torch.argmax(nst_label_target, 1)
            adj_loss = self.nst_loss_weight * self.cross_loss(label_raw, nst_label_target)
            total_loss += adj_loss
        elif self.adj_loss_version == 5:
            nst_idx_flat = self.source_nst_idx[self.source_idx].reshape(-1) # [N, nst_idx] -> [N*nst_idx]
            nst_label_target = F.softmax(label_raw[nst_idx_flat, :], -1).detach().reshape(N, self.num_nst_points, K) # [N, nst_idx, K]
            nst_label_target = torch.mean(nst_label_target, 1) # [N, K]
            loss_nst = self.nst_loss_weight * (1.0/decay_factor) * self.cross_loss(label_raw, nst_label_target)
            total_loss += loss_nst

        """deform_source, _, _ = model.compute_deformfeature(frame_time_source, frame_time_source)
        loss_sym = self.sym_loss_weight * decay_factor * self.get_sym_loss(deform_source)
        total_loss += loss_sym"""

        loss_sym = self.sym_loss_weight * decay_factor * self.symreg(model)
        total_loss += loss_sym
        
        loss_tv = self.tv_deform_loss_weight * decay_factor * model.TV_loss_deform(self.tvreg_s)
        total_loss += loss_tv

        """loss_tv = self.tv_deform_loss_weight * decay_factor * model.TV_loss_soft_deform(self.tvreg_s)
        total_loss += loss_tv"""

        loss_tv = self.tv_label_loss_weight * decay_factor * model.TV_loss_label(self.tvreg_s) #(1.0/decay_factor)**3
        total_loss += loss_tv

        if self.step < 5000:
            loss_soft_deform_regul = 1e2 * self.l1_loss(soft_deform, torch.zeros_like(soft_deform))
            #loss_soft_deform_regul = 1e0 / decay_factor * torch.mean(torch.sum(soft_deform ** 2, -1))
            total_loss += loss_soft_deform_regul

        """if self.step > 5000:
            loss_tv = self.tv_label_loss_weight * 1000.0 * model.TV_loss_label(self.tvreg_s) 
            total_loss += loss_tv
        else:
            loss_tv = self.tv_label_loss_weight * decay_factor * model.TV_loss_label(self.tvreg_s) #(1.0/decay_factor)**3
            total_loss += loss_tv"""

        self.step = self.step + 1

        if self.step % self.vis_every == 0 and self.vis:
            self.visualizer.vis(model, iter=self.step, interactive=False)

        return total_loss

    def init_zero_deform(self, vis=True, save=True):

        model = self.model

        # Initialize the optimizer
        beta1 = 0.9
        beta2 = 0.99
        grad_vars = model.get_optparam_groups(lr_scale=1.0)
        optimizer = torch.optim.Adam(grad_vars, betas=(beta1, beta2))
        l1_loss = torch.nn.L1Loss()

        total_loss = 1.0
        while total_loss > 1e-3:
            deform_all = model.deform_plane[0][0, :, :, :].reshape(-1, model.time_grid * model.time_grid)
            deform_all = model.deform_feature_out(deform_all.T).reshape(model.time_grid * model.time_grid, -1, 7)
            gt_deform = torch.zeros_like(deform_all)
            gt_deform[:, :, 0] = 1.0
            total_loss = l1_loss(deform_all - gt_deform, torch.zeros_like(deform_all))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print("init deform loss: ", total_loss)

        if vis:
            self.visualizer.vis(model, iter=0, interactive=False)

        if save:
            save_path = f"{self.logfolder}/init_zero_deform.th"
            torch.save(model, save_path)

    def init_gmm_seg(self, vis=True, save=True):

        model = self.model
        
        emptiness_source = self.emptiness_all[self.source_idx]
        xyzs_source = self.xyzs[emptiness_source, :]
        frame_time_source = torch.zeros((xyzs_source.shape[0], 1), device=self.xyzs.device)
        valid_xyzs = self.xyzs[emptiness_source, :].detach().cpu().numpy()

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=model.label_dim)
        label = kmeans.fit_predict(valid_xyzs)
        gt_label = torch.eye(model.label_dim, dtype=self.xyzs.dtype, device=self.xyzs.device)[label]

        # Initialize the optimizer
        beta1 = 0.9
        beta2 = 0.99
        grad_vars = model.get_optparam_groups(self.cfg.optim, lr_scale=1.0)
        optimizer = torch.optim.Adam(grad_vars, betas=(beta1, beta2))

        time = self.time_set[self.source_idx] * torch.ones((1, 1), device=self.xyzs.device)
        frame_time = time.view((1, 1)).expand(self.xyzs.shape[0], 1)
        label_acc = 0.0
        iteration = 0
        while label_acc < 0.5:
            # Calculate the learning rate decay factor

            total_loss = 0.0
            for t in self.time_set:
                frame_time = t * torch.ones((xyzs_source.shape[0], 1), device=self.xyzs.device)
                label_features = model.compute_labelfeature(
                    xyzs_source, frame_time
                )
                pred_label = torch.nn.functional.softmax(label_features, 1)
                label_loss = - gt_label * torch.log(pred_label + 1e-8)
                label_loss = torch.sum(label_loss, 1).mean()
                _, pred_label_ind = torch.max(pred_label, 1)
                _, gt_label_ind = torch.max(gt_label, 1)
                label_acc = (pred_label_ind == gt_label_ind).float().mean()
                total_loss += label_loss 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iteration += 1

        if vis:
            self.visualizer.vis(model, iter=1, interactive=False)

        if save:
            save_path = f"{self.logfolder}/init_seg.th"
            torch.save(model, save_path)



