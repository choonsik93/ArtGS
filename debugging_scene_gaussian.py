import os
import sys
import numpy as np
import random
import torch
import torch.nn
from decomposition.segmentation_model import HexPlane
from decomposition.sampler_3d import Sampler3D
from config.config import Config
from pytorch3d.ops.points_to_volumes import add_points_features_to_volume_densities_features
from tqdm import tqdm

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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

def train(hyper):
    Config.optim.vis = True
    Config.optim.vis_every = 1000

    device = "cuda"

    seg_model = HexPlane(args, Config)
    seg_model.cuda()

    points_list = []
    opacity_list = []
    for i in range(0, 160, 1):
        #path = "output/dnerf/hellwarrior/point_numpy_pertimestamp/points_%d.npz"%(i)
        #path = "dnerf_results/lego/point_numpy_pertimestamp/points_%d.npz"%(i)
        #path = "/home/jaegu/Desktop/2024IROS/4DGaussians/standup/point_numpy_pertimestamp/points_%d.npz"%(i)
        path = f"{args.object}/point_numpy_pertimestamp/points_%d.npz"%(i)
        points_list.append(np.load(path)["points"])
        opacity_list.append(np.load(path)["opacity"])

    points_list = np.stack(points_list, 0)
    opacity_list = np.stack(opacity_list, 0)

    num_data = points_list.shape[0]
    source_idx = num_data // 2
    threshold = Config.data.opacity_threshold # 0.01
    ys_valid_idx = np.where(opacity_list[source_idx] > threshold)[0]
    points_list = points_list[:, ys_valid_idx, :]

    """import trimesh
    for i in range(0, 160, 10):
        pc = trimesh.PointCloud(points_list[i])
        pc.show()"""

    if args.object == "cookie":
        aabb = torch.Tensor([[15.0510, 13.7514, 69.6490], [-24.5438, -41.5278, 9.5826]]).to(device)
    else:
        aabb = torch.Tensor([[-1.2998, -1.2998, -1.2987], [1.2998, 1.2999, 1.2999]]).to(device)
    #aabb = torch.Tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).to(device)
    xyz_min = aabb[0]
    xyz_max = aabb[1]

    points_set = torch.FloatTensor(points_list).to(device)
    time_set = torch.linspace(0.0, 1.0, num_data).to(device)
        
    """seg_model = HexPlane(Config, device=device)
    seg_model.aabb[0] = xyz_min
    seg_model.aabb[1] = xyz_max"""

    #seg_model = torch.load("seg_model_3000.th")
    #print(seg_model.aabb[0], seg_model.aabb[1])

    gridSize = [Config.data.emptiness_map_size, Config.data.emptiness_map_size, Config.data.emptiness_map_size]\

    points_set = (points_set - xyz_min) * (2.0 / (xyz_max - xyz_min)) - 1.0

    dense_xyz, emptiness_all = generate_emptiness(points_set, gridSize=gridSize, device="cuda", threshold=Config.data.emptiness_threshold)

    sampler_3d = Sampler3D(seg_model, Config, model_path=args.object, device=device)
    sampler_3d.init_emptiness_from_input(dense_xyz, emptiness_all, time_set, source_idx=source_idx)
    sampler_3d.init_zero_deform(vis=False, save=False)
    #sampler_3d.init_gmm_seg(vis=False, save=False)
    #torch.save(seg_model, "init_zero_deform_40_degug.th")
    grad_vars = seg_model.get_optparam_groups(lr_scale=1.0)
    optimizer = torch.optim.Adam(grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))


    """scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Config.optim.max_lr, 
                                                    steps_per_epoch=Config.optim.lr_decay_step, epochs=Config.optim.n_iters)"""

    """for i in range(0, emptiness_all.shape[0], 10):
        import trimesh
        from scipy.spatial.transform import Rotation as R
        emptiness = emptiness_all[i]
        valid_xyz = dense_xyz[emptiness, :].detach().cpu().numpy()
        pc = trimesh.PointCloud(valid_xyz)
        scene = trimesh.Scene()
        camera_rotation = np.eye(4)
        rotation = R.from_euler('xyz', [70, 0, -30], degrees=True).as_matrix() # hellwarrior
        camera_rotation[0:3, 0:3] = rotation
        camera = trimesh.scene.Camera(fov=(np.pi/4.0, np.pi/4.0))
        transform = camera.look_at([[0, 0.0, 0.0]], rotation=camera_rotation, distance=2.5)
        scene.camera_transform = transform
        scene.add_geometry(pc)
        scene.show()
        file_name = f"gaussian_%d.png"%(i)
        data = scene.save_image(resolution=(1024, 1024), visible=True)
        with open(file_name, 'wb') as f:
            f.write(data)
            f.close()"""

    progress_bar = tqdm(range(0, Config.optim.n_iters), desc="Training progress")

    color_list = np.random.randint(255, size=(40, 3))
    import trimesh

    #sampler_3d.step = 2999

    for iteration in range(0, Config.optim.n_iters):
        loss = sampler_3d.get_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr_org"] * sampler_3d.lr_factor
        """if iteration > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr_org"] * 0.1"""

        """if (iteration + 1) % Config.optim.lr_decay_step == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * Config.optim.lr_decay_ratio"""
        
        """if (iteration + 1) % 1000 == 0:
            sampler_3d.hierarchical_clustering()"""

        """if (iteration + 1) % 2000 == 0 and iteration + 1 != Config.optim.n_iters:
            sampler_3d.shrink_label()
            sampler_3d.visualizer.vis(seg_model, iter=iteration+2, interactive=False)
            grad_vars = seg_model.get_optparam_groups(Config.optim, lr_scale=1.0)
            optimizer = torch.optim.Adam(grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))"""
        
        if iteration in Config.optim.shrink_list:
            sampler_3d.shrink_label()
            sampler_3d.visualizer.vis(seg_model, iter=iteration+2, interactive=False)
            grad_vars = seg_model.get_optparam_groups(lr_scale=1.0)
            optimizer = torch.optim.Adam(grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))

        if iteration in Config.optim.group_merge_list:
            sampler_3d.group_merge()
            sampler_3d.visualizer.vis(seg_model, iter=iteration+3, interactive=False)
            grad_vars = seg_model.get_optparam_groups(lr_scale=1.0)
            optimizer = torch.optim.Adam(grad_vars, betas=(Config.optim.beta1, Config.optim.beta2))

        if iteration in Config.data.emptiness_downsample_list:
            dense_xyz, emptiness_all = generate_emptiness(points_set, gridSize=gridSize, device="cuda", threshold=Config.data.emptiness_threshold_list[0])
            sampler_3d.init_emptiness_from_input(dense_xyz, emptiness_all, time_set, source_idx=source_idx)

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)
        if iteration == Config.optim.n_iters - 1:
            progress_bar.close()
        
    torch.save(seg_model, f"{args.object}/debug.th")


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--object", type=str, default="hellwarrior")
    parser.add_argument("--decomp_configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
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
        
    hyper = hp.extract(args)
    print(hyper)
    train(hyper)