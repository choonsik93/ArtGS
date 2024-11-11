# Unsupervised 3D Part Decomposition via Leveraged Gaussian Splatting
JaeGoo Choy*, Geonho Cha, Hogun Kee, Songhwai Oh
[Project](https://choonsik93.github.io/artnerf/) | [Full Paper]() | [Video](https://youtu.be/4B10ItDZNK0?si=KAZPVlScu1L7_YTB) 

This repository contains the official authors implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found here. We further provide the reference images used to create the error metrics reported in the paper, as well as recently created, pre-trained models.

## Environmental Setups

```bash
git clone https://github.com/rllab-snu/ArtGS.git
cd ArtGS
git submodule init
git submodule update
make build-image
make run
```

## Data Preparation

The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is used. You can download the dataset from [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0).


## Training

For training synthetic scenes such as `bouncingballs`, run

```
python train.py -s data/dnerf/bouncingballs --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
```

```
python generate_mesh.py -s data/dnerf/bouncingballs --expname "dnerf/bouncingballs" --model_path "output/dnerf/bouncingballs"
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> 
```
