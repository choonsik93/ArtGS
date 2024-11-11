#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
from utils.sh_utils import RGB2SH

from PIL import Image, ImageColor
#color_list = torch.rand((40, 3), device="cuda")
colormap = sorted(set(ImageColor.colormap.values()))
color_list = torch.FloatTensor([ImageColor.getrgb(color) for color in colormap[:-1]]).to("cuda") / 255.0

source_cam = None

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array


def render_articulation(viewpoint_camera, pc:GaussianModel, pipe, bg_color:torch.Tensor, seg_model, source_scene=None, eval=False,
                        scaling_modifier=1.0, override_color=None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    """global source_cam
    if source_cam is None:
        source_cam = viewpoint_camera

    viewpoint_camera_time = viewpoint_camera.time
    viewpoint_camera = source_cam
    viewpoint_camera.time = viewpoint_camera_time"""

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) 
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        source_time = torch.tensor(seg_model.time_source).to(means3D.device).repeat(means3D.shape[0], 1)
    else:
        raster_settings = viewpoint_camera['camera']
        source_time = torch.tensor(seg_model.time_source).to(means3D.device).repeat(means3D.shape[0], 1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    if source_scene is None:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, rotations, opacity, shs, source_time)
        """means3D_final = means3D_final.detach()
        scales_final = scales_final.detach()
        rotations_final = rotations_final.detach()
        opacity_final = opacity_final.detach()
        shs_final = shs_final.detach()"""
    else:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = source_scene
        """means3D_final = means3D_final.detach()
        scales_final = scales_final.detach()
        rotations_final = rotations_final.detach()
        opacity_final = opacity_final.detach()
        shs_final = shs_final.detach()"""

    means3D_final_copy = means3D_final.clone()
    target_time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
    means3D_final, rotations_final, _ = seg_model(means3D_final, rotations_final, source_time, target_time, eval=eval)

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    frame_time_source = torch.zeros_like(means3D_final_copy[:, 0:1])
    label_feat = seg_model.compute_labelfeature(means3D_final_copy, frame_time_source, xyz_normalize=True)
    _, label_ind = torch.max(label_feat, 1)
    label_final = torch.eye(label_feat.shape[-1], dtype=label_feat.dtype, device=label_feat.device)[label_ind]
    label_sort = torch.argsort(torch.sum(label_final, 0), descending=True)
    max_label = label_sort[2]
    max_points_idx = label_final[:, max_label] == 1
    """means3D_final[max_points_idx, 0] += 0.5 * math.cos(2 * math.pi * viewpoint_camera.time)
    means3D_final[max_points_idx, 1] += 0.5 * math.sin(2 * math.pi * viewpoint_camera.time)
    means3D_final[max_points_idx, 2] += 0.2 * math.cos(2 * math.pi * viewpoint_camera.time)"""

    """means3D_final = means3D_final[max_points_idx]
    rotations_final = rotations_final[max_points_idx]
    opacity = opacity[max_points_idx]
    scales_final = scales_final[max_points_idx]
    shs_final = shs_final[max_points_idx]"""

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=shs_final,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp)
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "means3D": means3D_final,
            "opacity": opacity,}


def render_label(viewpoint_camera, pc:GaussianModel, pipe, bg_color:torch.Tensor, seg_model, eval=False,
                 scaling_modifier=1.0, override_color=None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration

    bg_color = torch.zeros_like(bg_color)
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        source_time = torch.tensor(seg_model.time_source).to(means3D.device).repeat(means3D.shape[0], 1)
    else:
        raster_settings = viewpoint_camera['camera']
        source_time = torch.tensor(seg_model.time_source).to(means3D.device).repeat(means3D.shape[0], 1)

    #rasterizer = DecompGaussianRasterizer(raster_settings=raster_settings)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, rotations, opacity, shs, source_time)
    means3D_final = means3D_final.detach()
    means3D_final_copy = means3D_final.clone()
    scales_final = scales_final.detach()
    rotations_final = rotations_final.detach()
    rotations_final_copy = rotations_final.clone()
    opacity_final = opacity_final.detach()
    shs_final = shs_final.detach()
    
    target_time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
    means3D_final, rotations_final, _ = seg_model(means3D_final, rotations_final, source_time, target_time, eval=eval)

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    label_sh = False
    if label_sh:
        frame_time_source = torch.zeros_like(means3D_final_copy[:, 0:1])
        label_feat = seg_model.compute_labelfeature(means3D_final_copy, frame_time_source, xyz_normalize=True)
        _, label_ind = torch.max(label_feat, 1)
        RGB = color_list[label_ind]
        colors_precomp = RGB
        #shs_final = RGB2SH(RGB)[:, None, :].repeat(1, 16, 1)
        shs_final = None

        rendered_image, radii, depth = rasterizer(
            means3D=means3D_final,
            means2D=means2D,
            shs=shs_final,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales_final,
            rotations=rotations_final,
            cov3D_precomp=cov3D_precomp)
    else:
        frame_time_source = torch.zeros_like(means3D_final_copy[:, 0:1])
        label_feat = seg_model.compute_labelfeature(means3D_final_copy, frame_time_source, xyz_normalize=True)
        _, label_ind = torch.max(label_feat, 1)
        label_final = torch.eye(label_feat.shape[-1], dtype=label_feat.dtype, device=label_feat.device)[label_ind]
        label_sort = torch.argsort(torch.sum(label_final, 0), descending=True)
        max_label = label_sort[1]
        max_points_idx = label_final[:, max_label] == 1
        #means3D_final[max_points_idx, 2] += 0.5
        
        """colors_precomp = label_final[:, 0:3]
        shs_final = None
        rendered_image, radii, depth = rasterizer(
            means3D=means3D_final,
            means2D=means2D,
            shs=shs_final,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales_final,
            rotations=rotations_final,
            cov3D_precomp=cov3D_precomp)
        
        colors_precomp = label_final[:, 3:6]
        shs_final = None
        rendered_image_2, radii, depth = rasterizer(
            means3D=means3D_final,
            means2D=means2D,
            shs=shs_final,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales_final,
            rotations=rotations_final,
            cov3D_precomp=cov3D_precomp)
        
        colors_precomp = label_final[:, 6:9]
        shs_final = None
        rendered_image_3, radii, depth = rasterizer(
            means3D=means3D_final,
            means2D=means2D,
            shs=shs_final,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales_final,
            rotations=rotations_final,
            cov3D_precomp=cov3D_precomp)
        
        colors_precomp = label_final[:, 9:12]
        shs_final = None
        rendered_image_4, radii, depth = rasterizer(
            means3D=means3D_final,
            means2D=means2D,
            shs=shs_final,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales_final,
            rotations=rotations_final,
            cov3D_precomp=cov3D_precomp)
        
        colors_precomp = label_final[:, 12:15]
        shs_final = None
        rendered_image_5, radii, depth = rasterizer(
            means3D=means3D_final,
            means2D=means2D,
            shs=shs_final,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales_final,
            rotations=rotations_final,
            cov3D_precomp=cov3D_precomp)
        
        rendered_image = torch.cat((rendered_image, rendered_image_2, rendered_image_3, rendered_image_4, rendered_image_5), 0)"""

        rendered_image_set = []
        label_dim = label_final.shape[1]
        for i in range(0, label_dim, 3):
            #print(i)
            colors_precomp = label_final[:, i:i+3]
            if colors_precomp.shape[1] != 3:
                colors_precomp_pad = torch.zeros((colors_precomp.shape[0], 3 - colors_precomp.shape[1]), device=colors_precomp.device)
                colors_precomp = torch.cat((colors_precomp, colors_precomp_pad), 1)
            shs_final = None
            rendered_image, radii, depth = rasterizer(
                means3D=means3D_final,
                means2D=means2D,
                shs=shs_final,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales_final,
                rotations=rotations_final,
                cov3D_precomp=cov3D_precomp)
            """rendered_image, radii, depth = rasterizer(
                means3D=means3D_final_copy,
                means2D=means2D,
                shs=shs_final,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales_final,
                rotations=rotations_final_copy,
                cov3D_precomp=cov3D_precomp)"""
            rendered_image_set.append(rendered_image)
            
        rendered_image = torch.cat(rendered_image_set, 0)

        """label_dim = label_final.shape[1]
        if label_dim % 3 != 0:
            colors_precomp = torch.cat((label_final, torch.zeros_like(label_final[:, 0:3-label_dim%3])), 1)
        else:
            colors_precomp = label_final
        #colors_precomp = torch.zeros_like(label_final)
        #print(colors_precomp.shape, torch.sum(colors_precomp, 1))
        #print(torch.max(label_final, 1))
        shs_final = None

        render_image_set = []
        for i in range(0, colors_precomp.shape[1], 3):
            rendered_image, radii, depth = rasterizer(
                means3D=means3D_final,
                means2D=means2D,
                shs=shs_final,
                colors_precomp=colors_precomp[i:i+3],
                opacities=opacity,
                scales=scales_final,
                rotations=rotations_final,
                cov3D_precomp=cov3D_precomp)
            render_image_set.append(rendered_image.detach())
        rendered_image = torch.cat(render_image_set, 0)
        rendered_image = rendered_image[0:label_dim, ...]"""
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}