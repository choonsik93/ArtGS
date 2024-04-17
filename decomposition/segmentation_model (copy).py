import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
#import scene.tools as tools
from pytorch3d.transforms import quaternion_apply, quaternion_multiply


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


class HexPlane(torch.nn.Module):

    def __init__(self, args, cfg, device="cuda"):
        super().__init__()
        bounds = args.bounds
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]])
        self.aabb = torch.nn.Parameter(aabb, requires_grad=False)
        #self.grid_config = [args.planeconfig]
        self.grid_config = args.kplanes_config
        self.multiscale_res_multipliers = args.multires
        self.concat_features = True

        self.W = args.net_width
        self.D = args.defor_depth

        self.cfg = cfg
        self.device = device

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.num_matMode = len(self.matMode)

        self.label_dim = cfg.model.max_part_num
        self.deform_dim = 7 * self.label_dim
        self.align_corners = cfg.model.align_corners
        self.init_scale = cfg.model.init_scale
        self.init_shift = cfg.model.init_shift
        #self.device = device
        #self.init_planes(self.device)
        self.init_planes()

        self.deform_arg = {'gumbel': cfg.optim.gumbel, 'hard': cfg.optim.hard, 'tau': cfg.optim.tau, 'eval': cfg.optim.eval}
        self.time_source = None

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def init_planes(self):
        self.label_plane = self.init_one_triplane()
        self.deform_plane = self.init_one_plane()
        self.soft_deform_plane = self.init_one_hexplane()

        config = self.grid_config.copy()
        out_dim = config["output_coordinate_dim"]
        res_dim = len(self.multiscale_res_multipliers)

        self.label_feature_out = [nn.Linear(3 * res_dim * out_dim, self.W)]
        for i in range(self.D - 1):
            self.label_feature_out.append(nn.ReLU())
            self.label_feature_out.append(nn.Linear(self.W, self.W))
        self.label_feature_out.append(nn.ReLU())
        self.label_feature_out.append(nn.Linear(self.W, self.label_dim))
        self.label_feature_out = nn.Sequential(*self.label_feature_out)
        print(self.label_feature_out)

        self.deform_feature_out = [nn.Linear(out_dim, self.W)]
        for i in range(self.D - 1):
            self.deform_feature_out.append(nn.ReLU())
            self.deform_feature_out.append(nn.Linear(self.W, self.W))
        self.deform_feature_out.append(nn.ReLU())
        self.deform_feature_out.append(nn.Linear(self.W, self.deform_dim))
        self.deform_feature_out = nn.Sequential(*self.deform_feature_out)
        print(self.deform_feature_out)

        self.soft_deform_feature_out = [nn.Linear(6 * res_dim * out_dim, self.W)]
        for i in range(self.D - 1):
            self.soft_deform_feature_out.append(nn.ReLU())
            self.soft_deform_feature_out.append(nn.Linear(self.W, self.W))
        self.soft_deform_feature_out.append(nn.ReLU())
        self.soft_deform_feature_out.append(nn.Linear(self.W, 3))
        self.soft_deform_feature_out[-1].weight.data.fill_(0.0)
        self.soft_deform_feature_out[-1].bias.data.fill_(0.0)
        self.soft_deform_feature_out = nn.Sequential(*self.soft_deform_feature_out)
        print(self.soft_deform_feature_out)

    def init_one_plane(self):
        plane_coef = []
        config = self.grid_config.copy()
        out_dim = config["output_coordinate_dim"]
        gridSize = config["resolution"][3]
        self.time_grid = gridSize
        plane_coef.append(
            torch.nn.Parameter(
                self.init_scale
                * torch.randn(
                    (1, out_dim, gridSize, gridSize)
                )
                + self.init_shift
            ) 
        )
        return torch.nn.ParameterList(plane_coef)
        #return torch.nn.ParameterList(plane_coef).to(self.device)
    
    def init_one_triplane(self):
        plane_coef = []
        for res in self.multiscale_res_multipliers:
            config = self.grid_config.copy()
            # upsample resolution for multires hexplane, last dimension is for time
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            gridSize = config["resolution"][0:3]
            out_dim = config["output_coordinate_dim"]
            for i in range(len(self.matMode)):
                mat_id_0, mat_id_1 = self.matMode[i]
                plane_coef.append(
                    torch.nn.Parameter(
                        self.init_scale
                        * torch.randn(
                            (1, out_dim, gridSize[mat_id_1], gridSize[mat_id_0])
                        )
                        + self.init_shift
                    )
                )
        return torch.nn.ParameterList(plane_coef)
    
    def init_one_hexplane(self):
        plane_coef = []
        for res in self.multiscale_res_multipliers:
            config = self.grid_config.copy()
            # upsample resolution for multires hexplane, last dimension is for time
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            gridSize = config["resolution"][0:3]
            out_dim = config["output_coordinate_dim"]
            time_grid = config["resolution"][3]
            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]
                plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, out_dim, gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
                )
                plane_coef.append(
                    torch.nn.Parameter(
                        self.init_scale
                        * torch.randn((1, out_dim, gridSize[vec_id], time_grid))
                        + self.init_shift
                    )
                )
        return torch.nn.ParameterList(plane_coef)
    
    def compute_labelfeature(self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor, xyz_normalize=False) -> torch.Tensor:

        xyz_sampled = normalize_aabb(xyz_sampled, self.aabb) if xyz_normalize else xyz_sampled

        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].

        plane_feat = []
        for i_res in range(len(self.multiscale_res_multipliers)):
            for idx_plane in range(self.num_matMode):
                plane_feat.append(
                    F.grid_sample(
                        self.label_plane[idx_plane + i_res * self.num_matMode],
                        plane_coord[[idx_plane]],
                        align_corners=self.align_corners,
                    ).view(-1, *xyz_sampled.shape[:1])
                )
        plane_feat = torch.cat(plane_feat, 0)
        label_feat = self.label_feature_out(plane_feat.T)
        return label_feat
    
    def compute_soft_deformfeature(self, xyz, time):

        plane_coord = (
            torch.stack(
                (
                    xyz[..., self.matMode[0]],
                    xyz[..., self.matMode[1]],
                    xyz[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz[..., self.vecMode[0]],
                xyz[..., self.vecMode[1]],
                xyz[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat = []
        for i_res in range(len(self.multiscale_res_multipliers)):
            for idx_plane in range(self.num_matMode):
                plane_feat.append(
                    F.grid_sample(
                        self.soft_deform_plane[idx_plane * 2 + i_res * self.num_matMode * 2],
                        plane_coord[[idx_plane]],
                        align_corners=self.align_corners,
                    ).view(-1, *xyz.shape[:1])
                )
                plane_feat.append(
                    F.grid_sample(
                        self.soft_deform_plane[1 + idx_plane * 2 + i_res * self.num_matMode * 2],
                        line_time_coord[[idx_plane]],
                        align_corners=self.align_corners,
                    ).view(-1, *xyz.shape[:1])
                )
        plane_feat = torch.cat(plane_feat, 0)
        soft_deform = self.soft_deform_feature_out(plane_feat.T)
        return soft_deform
    
    def compute_deformfeature(self, source_time, target_time):
        B = source_time.shape[0]
        source_time = source_time.view(1, -1, 1, 1)
        target_time = target_time.view(1, -1, 1, 1)

        time_coord = torch.cat((source_time, target_time), -1).detach() # (1, B, 1, 2)
        
        plane_feat = F.grid_sample(
            self.deform_plane[0],
            time_coord,
            align_corners=self.align_corners,
        )[0, :, :, 0] #.view(-1, 1) # (1, C, B, 1) -> (C, B)

        deform = self.deform_feature_out(plane_feat.T) # (C, B) -> (B, C) -> (B, K*9)
        deform = deform.reshape(B, -1, 7)
        quaternion = torch.nn.functional.normalize(deform[:, :, 0:4], dim=-1)
        translation = deform[:, :,4:7]

        return deform, quaternion, translation
    
    def label_feat_to_label(self, label, deform_arg=None):
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
    
    def compute_batch_deform_once(
        self, xyz_sampled: torch.Tensor, frame_time_source: torch.Tensor, frame_time_target: torch.Tensor, 
        gumbel=True, hard=False, tau=0.1, eval=False, merge=True
    ) -> torch.Tensor:
        assert frame_time_source.shape == frame_time_target.shape

        label_raw = self.compute_labelfeature(xyz_sampled, frame_time_source) # (N, K)

        _, quaternion, translation = self.compute_deformfeature(frame_time_source, frame_time_target)
        xyz_sampled_deform = quaternion_apply(quaternion, xyz_sampled[:, None, :]) + translation # [N, K, 3]

        deform_arg = {"gumbel": gumbel, "hard": hard, "tau": tau, "eval": eval}
        label = self.label_feat_to_label(label_raw, deform_arg=deform_arg)

        xyz_sampled_deform = torch.sum(xyz_sampled_deform * label[:, :, None], 1) if merge else xyz_sampled_deform
        soft_deform = self.compute_soft_deformfeature(xyz_sampled, frame_time_target)
        if merge:
            xyz_sampled_deform += soft_deform
        else:
            xyz_sampled_deform += soft_deform[:, None, :]

        return xyz_sampled_deform, label_raw, label, quaternion, translation
    
    def forward(self, rays_pts_emb, rotations_emb, time_feature_source, time_feature_target, eval=False):  
        pts = rays_pts_emb[:, :3]
        rotations = rotations_emb[:, :4]
        
        _, quaternion, translation = self.compute_deformfeature(time_feature_source, time_feature_target)
        label_feat = self.compute_labelfeature(pts, time_feature_source, xyz_normalize=True) # [N, K]
        if eval:
            _, label_ind = torch.max(label_feat, 1)
            label = torch.eye(label_feat.shape[-1], dtype=label_feat.dtype, device=label_feat.device)[label_ind]
        else:
            label = self.label_feat_to_label(label_feat) # [N, K]

        quaternion_concat = torch.sum(quaternion * label[:, :, None], 1) # [N, 4]
        translation_concat = torch.sum(translation * label[:, :, None], 1) # [N, 3]

        pts = quaternion_apply(quaternion_concat, pts) + translation_concat
        rotations = quaternion_multiply(quaternion_concat, rotations)

        return pts, rotations, label
    
    def TV_loss_label(self, reg):
        total = 0
        for idx in range(len(self.label_plane)):
            total = total + reg(self.label_plane[idx])
        return total
    
    def TV_loss_deform(self, reg):
        total = reg(self.deform_plane[0])
        return total
    
    def TV_loss_soft_deform(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(0, len(self.soft_deform_plane), 2):
            total = total + reg(self.soft_deform_plane[idx]) + reg2(self.soft_deform_plane[idx + 1])
        return total
    
    def get_optparam_groups(self, lr_scale=1.0):
        cfg = self.cfg
        grad_vars = [
            {
                "params": self.deform_plane,
                "lr": lr_scale * cfg.optim.lr_deform_grid,
                "lr_org": cfg.optim.lr_deform_grid,
                "name": "grid",
            },
            {
                "params": self.deform_feature_out.parameters(),
                "lr": lr_scale * cfg.optim.lr_deform_nn,
                "lr_org": cfg.optim.lr_deform_nn,
            },
            {
                "params": self.soft_deform_plane,
                "lr": lr_scale * cfg.optim.lr_deform_grid,
                "lr_org": cfg.optim.lr_deform_grid,
                "name": "grid",
            },
            {
                "params": self.soft_deform_feature_out.parameters(),
                "lr": lr_scale * cfg.optim.lr_deform_nn,
                "lr_org": cfg.optim.lr_deform_nn,
            },
            {
                "params": self.label_plane,
                "lr": lr_scale * cfg.optim.lr_label_grid,
                "lr_org": cfg.optim.lr_label_grid,
                "name": "grid",
            },
            {
                "params": self.label_feature_out.parameters(),
                "lr": lr_scale * cfg.optim.lr_label_nn,
                "lr_org": cfg.optim.lr_label_nn,
            },
        ]
        return grad_vars
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "mat" in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "mat" not in name:
                parameter_list.append(param)
        return parameter_list

    def init_zero_deform(self):

        # Initialize the optimizer
        grad_vars = self.get_optparam_groups(self.cfg.optim, lr_scale=1.0)
        optimizer = torch.optim.Adam(grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2))
        l1_loss = torch.nn.L1Loss()

        total_loss = 1.0
        while total_loss > 1e-3:
            deform_all = self.deform_plane[0][0, :, :, :].reshape(-1, self.time_grid * self.time_grid)
            deform_all = self.deform_basis_mat(deform_all.T).reshape(self.time_grid * self.time_grid, -1, 9)
            gt_deform = torch.zeros_like(deform_all)
            gt_deform[:, :, 0] = 1.0
            gt_deform[:, :, 4] = 1.0
            total_loss = l1_loss(deform_all - gt_deform, torch.zeros_like(deform_all))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print("init deform loss: ", total_loss)