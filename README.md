## Environmental Setups

```bash
git clone https://github.com/choonsik93/ArtGS.git
cd ArtGS
git submodule init
git submodule update
make build-image
make run
```

```bash
conda create -n artgs python=3.9
conda activate artgs
conda install pytorch=2.1.0 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c iopath iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
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
