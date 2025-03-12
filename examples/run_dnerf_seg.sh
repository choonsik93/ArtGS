python train.py -s data/dnerf/$1 --expname "dnerf/$1" --configs arguments/dnerf/$1.py &&
python train_seg.py -s data/dnerf/$1 --expname "dnerf/$1" --configs arguments/dnerf/$1.py --decomp_configs config/dnerf/$1.yaml && 
python render_seg.py --model_path "output/dnerf/$1/" --skip_train --configs arguments/dnerf/$1.py