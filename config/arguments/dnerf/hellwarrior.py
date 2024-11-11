_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 50]
    }
)

"""OptimizationParams = dict(
    position_lr_init = 0.0,
    deformation_lr_init = 0.0,
    grid_lr_init = 0.0,
    grid_lr_final = 0.0,
    feature_lr = 0.0025,
    opacity_lr = 0.00,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
)"""