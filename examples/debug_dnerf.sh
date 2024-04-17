docker cp ../ArtGS/debugging_scene_gaussian.py $2:/workspace/ArtGS &&
docker cp ../ArtGS/decomposition $2:/workspace/ArtGS &&
docker cp ../ArtGS/config $2:/workspace/ArtGS &&
#docker cp ../ArtGS $2:/workspace &&
docker exec --workdir /workspace/ArtGS -it $2 python debugging_scene_gaussian.py --model_path "output/dnerf/$1/" --configs arguments/dnerf/$1.py --object $1
docker cp $2:/workspace/ArtGS/$1/test ./
#docker cp $2:/workspace/ArtGS/output/dnerf/$1 ./

#docker exec --workdir /workspace/ArtGS -it $2 python train.py -s data/dnerf/$1 --port 6017 --expname "dnerf/$1" --configs arguments/dnerf/$1.py
#docker exec --workdir /workspace/ArtGS -it $2 python train_seg.py -s data/dnerf/$1 --port 6017 --expname "dnerf/$1" --configs arguments/dnerf/$1.py --decomp_configs config/dnerf/$1.yaml && 
#docker exec --workdir /workspace/ArtGS -it $2 python render_seg.py --model_path "output/dnerf/$1/" --skip_train --configs arguments/dnerf/$1.py &&
#docker cp $2:/workspace/ArtGS/output/dnerf/$1 ./