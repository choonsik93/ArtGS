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
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

from pytorch3d.ops.points_to_volumes import add_points_features_to_volume_densities_features
import torchvision

import importlib
from decomposition.segmentation_model import HexPlane
from decomposition.sampler_3d import Sampler3D
from decomposition.scene_utils import render_training_image_and_label_articulation
from decomposition.gaussian_renderer import render_label, render_articulation

import open3d as o3d
from open3d.visualization import rendering
import math

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def part_reconstruction_3d(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):

    @torch.no_grad()
    def get_state_at_time(pc, viewpoint_camera):    
        means3D = pc.get_xyz
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
        opacity = pc._opacity
        shs = pc.get_features

        scales = pc._scaling
        rotations = pc._rotation
        cov3D_precomp = None

        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, rotations, opacity, shs, time)
                
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity_final = pc.opacity_activation(opacity_final)

        return means3D_final, scales_final, rotations_final, opacity_final, shs_final
    
    device = "cuda"

    seg_model = HexPlane(Config, device=device)
    first_iter = 0
    sampler_3d = Sampler3D(seg_model, Config, model_path=scene.model_path, device=device)
    sampler_3d.init_zero_deform(vis=False, save=False)
    grad_vars = seg_model.get_optparam_groups(Config.optim, lr_scale=1.0)
    """gaussians_grad_vars = gaussians.training_setup(opt)
    for param in gaussians_grad_vars:
        param["lr_org"] = 1e-5
        param["lr"] = 1e-5"""
    #optimizer = torch.optim.Adam(grad_vars + gaussians_grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))
    optimizer = torch.optim.Adam(grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))
    
    xyz_max, xyz_min = gaussians.get_aabb
    seg_model.aabb[0] = xyz_min
    seg_model.aabb[1] = xyz_max

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None

    final_iter = Config.optim.n_iters
    batch_size = Config.optim.batch_size
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    num_data = len(train_cams)
    source_idx = num_data // 2 if Config.source_idx is -1 else Config.source_idx
    sampled_scenes = [get_state_at_time(gaussians, viewpoint_cam) for viewpoint_cam in train_cams]
    points_list = [scene[0] for scene in sampled_scenes]
    points_set = torch.stack(points_list, 0)

    opacity_source = sampled_scenes[source_idx][3]
    threshold = Config.data.opacity_threshold
    ys_valid_idx = torch.where(opacity_source > threshold)[0]
    points_set = points_set[:, ys_valid_idx, :]
    normalized_points_set = (points_set - xyz_min) * (2.0 / (xyz_max - xyz_min)) - 1.0

    time_set = [viewpoint_cam.time for viewpoint_cam in train_cams]
    time_set = torch.FloatTensor(time_set).to(device)
    emptiness_gridSize = [Config.data.emptiness_map_size, Config.data.emptiness_map_size, Config.data.emptiness_map_size]
    emptiness_threshold = Config.data.emptiness_threshold
    dense_xyz, emptiness_all = generate_emptiness(normalized_points_set, device=device, gridSize=emptiness_gridSize, threshold=emptiness_threshold)
    
    sampler_3d.init_emptiness_from_input(dense_xyz, emptiness_all, time_set, source_idx=source_idx)
    seg_model.time_source = time_set[source_idx]
    source_cam = train_cams[source_idx]
    target_cam = train_cams[0]

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    def get_closest_cam_idx(query_time, time_set):
        return min(range(len(time_set)), key=lambda i: abs(time_set[i] - query_time))
    
    W, H = source_cam.image_width, source_cam.image_height
    render_o3d = rendering.OffscreenRenderer(W, H)
    render_o3d.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
    render_o3d.scene.scene.enable_sun_light(False)
    render_o3d.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA
    render_o3d.scene.show_axes(False)

    test_render_scene_idx = Config.test_render_scene_idx
    train_render_scene_idx = Config.train_render_scene_idx
    
    render_training_image_and_label_articulation(scene, gaussians, [test_cams[test_render_scene_idx]], render_articulation, render_label, pipe, background, seg_model, stage+"arttestlabel", 0, timer.get_elapsed_time(), scene.dataset_type)
    render_training_image_and_label_articulation(scene, gaussians, [train_cams[train_render_scene_idx]], render_articulation, render_label, pipe, background, seg_model, stage+"arttrainlabel", 0, timer.get_elapsed_time(), scene.dataset_type)
    train_case = "full"
    for iteration in range(first_iter, final_iter + 1):
        idx = 0
        viewpoint_cams = []
        while idx < batch_size:    
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            if not viewpoint_stack:
                viewpoint_stack = temp_list.copy()
            viewpoint_cams.append(viewpoint_cam)
            idx += 1
        if len(viewpoint_cams) == 0:
            continue

        if train_case == "debug":
            psnr_ = 0.0
            sampler_loss = sampler_3d.get_loss()
            loss = sampler_loss
        else:
            time_idx_source = []
            time_idx_target = []
            images = []
            gt_images = []
            for viewpoint_cam in viewpoint_cams:
                render_pkg = render_articulation(viewpoint_cam, gaussians, pipe, background, seg_model, cam_type=scene.dataset_type)
                image = render_pkg["render"]
                images.append(image.unsqueeze(0))
                if scene.dataset_type!="PanopticSports":
                    gt_image = viewpoint_cam.original_image.cuda()
                else:
                    gt_image = viewpoint_cam['image'].cuda()
                gt_images.append(gt_image.unsqueeze(0))
                time_idx_source.append(get_closest_cam_idx(source_cam.time, time_set))
                time_idx_target.append(get_closest_cam_idx(viewpoint_cam.time, time_set))      

            if iteration % 50 == 0 or iteration == 1:
                gt_target_redner_pkg = render(target_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
                xyz_target = gt_target_redner_pkg["means3D"]
                render_pkg = render_articulation(target_cam, gaussians, pipe, background, seg_model, cam_type=scene.dataset_type)
                xyz = render_pkg["means3D"]
                opacity = render_pkg["opacity"]
                xyz = xyz[opacity[:, 0] > 0.5]
                xyz_target = xyz_target[opacity[:, 0] > 0.5]
                vp = target_cam.world_view_transform
                cam2ego = vp.detach().cpu().numpy().transpose()
                W, H = target_cam.image_width, target_cam.image_height
                fov = target_cam.FoVx
                fx =  W / (2 * math.tan(fov / 2))

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
                pcd.paint_uniform_color([0.0, 0.0, 0.0])

                material = rendering.MaterialRecord()
                if render_o3d.scene.has_geometry("voxel_grid"):
                    render_o3d.scene.remove_geometry("voxel_grid")
                render_o3d.scene.add_geometry("voxel_grid", pcd, material)

                o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fx, W // 2, H // 2)
                render_o3d.setup_camera(o3d_intrinsic, cam2ego)

                img = render_o3d.render_to_image()
                occ_save_path = os.path.join(args.model_path, "images", '{0:05d}'.format(iteration) + "_" + stage + "_gaussians.png")
                print(f"Saving image to {occ_save_path}")
                o3d.io.write_image(occ_save_path, img)

                pcd = o3d.geometry.PointCloud()
                canonical_redner_pkg = render(source_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
                xyz_canonical = canonical_redner_pkg["means3D"]
                xyz_canonical = xyz_canonical[opacity[:, 0] > 0.5]
                pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
                #pcd.points = o3d.utility.Vector3dVector(xyz_canonical.detach().cpu().numpy())
                #pcd.paint_uniform_color([0.0, 0.0, 0.0])

                material = rendering.MaterialRecord()
                if render_o3d.scene.has_geometry("voxel_grid"):
                    render_o3d.scene.remove_geometry("voxel_grid")
                render_o3d.scene.add_geometry("voxel_grid", pcd, material)

                o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fx, W // 2, H // 2)
                render_o3d.setup_camera(o3d_intrinsic, cam2ego)

                img = render_o3d.render_to_image()
                occ_save_path = os.path.join(args.model_path, "images", '{0:05d}'.format(iteration) + "_" + stage + "_gaussians_canonical.png")
                print(f"Saving image to {occ_save_path}")
                o3d.io.write_image(occ_save_path, img)

                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible = False)
                vis.add_geometry(voxel_grid)
                img = vis.capture_screen_float_buffer(True)

                material = rendering.MaterialRecord()
                if render_o3d.scene.has_geometry("voxel_grid"):
                    render_o3d.scene.remove_geometry("voxel_grid")
                render_o3d.scene.add_geometry("voxel_grid", voxel_grid, material)

                o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fx, W // 2, H // 2)
                render_o3d.setup_camera(o3d_intrinsic, cam2ego)

                img = render_o3d.render_to_image()
                occ_save_path = os.path.join(args.model_path, "images", '{0:05d}'.format(iteration) + "_" + stage + ".png")
                print(f"Saving image to {occ_save_path}")
                o3d.io.write_image(occ_save_path, img)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_target.detach().cpu().numpy())
                pcd.paint_uniform_color([0.0, 0.0, 0.0])
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible = False)
                vis.add_geometry(voxel_grid)
                img = vis.capture_screen_float_buffer(True)

                material = rendering.MaterialRecord()
                if render_o3d.scene.has_geometry("voxel_grid"):
                    render_o3d.scene.remove_geometry("voxel_grid")
                render_o3d.scene.add_geometry("voxel_grid", voxel_grid, material)

                img = render_o3d.render_to_image()
                occ_save_path = os.path.join(args.model_path, "images", '{0:05d}'.format(iteration) + "_" + stage + "_target_voxel.png")
                print(f"Saving image to {occ_save_path}")
                o3d.io.write_image(occ_save_path, img)

                material = rendering.MaterialRecord()
                if render_o3d.scene.has_geometry("voxel_grid"):
                    render_o3d.scene.remove_geometry("voxel_grid")
                render_o3d.scene.add_geometry("voxel_grid", pcd, material)

                img = render_o3d.render_to_image()
                occ_save_path = os.path.join(args.model_path, "images", '{0:05d}'.format(iteration) + "_" + stage + "_target_pcd.png")
                print(f"Saving image to {occ_save_path}")
                o3d.io.write_image(occ_save_path, img)

                target_image = render_pkg["render"]
                save_path = os.path.join(args.model_path, "images", '{0:05d}'.format(iteration) + "_" + stage + "_image.png")
                torchvision.utils.save_image(target_image, save_path)

            image_tensor = torch.cat(images, 0)
            gt_image_tensor = torch.cat(gt_images, 0)

            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            sampler_loss = sampler_3d.get_loss(batch_idxs=[time_idx_source, time_idx_target])
            #loss = 1e-2 * Ll1 + sampler_loss
            loss = Ll1 + sampler_loss
            #loss = Ll1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            if "lr_org" in param_group:
                param_group["lr"] = param_group["lr_org"] * sampler_3d.lr_factor
        
        if iteration in Config.optim.shrink_list:
            sampler_3d.shrink_label()
            grad_vars = seg_model.get_optparam_groups(Config.optim, lr_scale=1.0)
            optimizer = torch.optim.Adam(grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))

        if iteration in Config.optim.group_merge_list:
            sampler_3d.group_merge()
            grad_vars = seg_model.get_optparam_groups(Config.optim, lr_scale=1.0)
            optimizer = torch.optim.Adam(grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))

        if iteration in Config.model.upsample_list:
            voxel_grid = seg_model.voxel_grid_list.pop(0)
            time_grid = seg_model.time_grid_list.pop(0)
            print("Upsample Grid: ", voxel_grid, time_grid)
            seg_model.upsample_volume_grid(voxel_grid, time_grid)
            grad_vars = seg_model.get_optparam_groups(Config.optim, lr_scale=1.0)
            optimizer = torch.optim.Adam(grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))
        
        if iteration in Config.data.emptiness_downsample_list:
            dense_xyz, emptiness_all = generate_emptiness(normalized_points_set, device=device, gridSize=emptiness_gridSize, threshold=Config.data.emptiness_threshold_list[0])
            sampler_3d.init_emptiness_from_input(dense_xyz, emptiness_all, time_set, source_idx=source_idx)

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss:.{7}f}", "L1Loss": f"{Ll1:.{7}f}", "SamplerLoss": f"{sampler_loss:.{7}f}", "psnr": f"{psnr_:.{2}f}"})
                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            if True: #Config.optim.vis:
                if iteration % Config.optim.vis_every == 0 or iteration == 1:
                    render_training_image_and_label_articulation(scene, gaussians, [test_cams[test_render_scene_idx]], render_articulation, render_label, pipe, background, 
                                                                 seg_model, stage+"arttestlabel", iteration, timer.get_elapsed_time(), scene.dataset_type, eval=False)
                    render_training_image_and_label_articulation(scene, gaussians, [train_cams[train_render_scene_idx]], render_articulation, render_label, pipe, background, 
                                                                 seg_model, stage+"arttrainlabel", iteration, timer.get_elapsed_time(), scene.dataset_type, eval=False)                  
    
    torch.save(seg_model, os.path.join(scene.model_path, "seg_model.th"))


def generate_emptiness(points_set, gridSize=[101, 101, 101], device="cuda", threshold=0.5):
    samples = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ),
        -1,
    ).to(device)
    dense_xyz = samples * 2.0 - 1.0

    # Generate emptiness voxel
    num_data = points_set.shape[0]
    points_features = torch.clone(points_set)
    volume_densities = torch.zeros((num_data, 1, *gridSize), device=device)
    volume_features = torch.zeros((num_data, 3, *gridSize), device=device)
    add_points_features_to_volume_densities_features(points_set, points_features, volume_densities, volume_features, rescale_features=False)

    emptiness_all = (volume_densities.squeeze() > threshold).transpose(1, 3)

    return dense_xyz, emptiness_all


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    timer.start()
    part_reconstruction_3d(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations, timer)


def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500*i for i in range(100)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3000, 4000, 5000, 6000, 7_000, 9000, 10000, 12000, 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--decomp_configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])

    print(args.iterations)
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    if args.decomp_configs:
        from config.config import Config
        from omegaconf import OmegaConf
        base_cfg = OmegaConf.structured(Config())
        yaml_cfg = OmegaConf.load(args.decomp_configs)
        Config = OmegaConf.merge(base_cfg, yaml_cfg)
    else:
        from config.config import Config

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
