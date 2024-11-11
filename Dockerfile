FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade

# install python3.9
RUN apt-get install -y keyboard-configuration
RUN apt install -y python3.9 python3.9-dev
RUN apt-get update && apt-get -y install wget python3-pip build-essential git curl lsb-release

# symlink
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN ln -s -f /usr/bin/python /usr/bin/python3

# Dependency
RUN echo "export CUDA_HOME=/usr/local/cuda && export PATH=/usr/local/cuda-11.8/bin:$PATH && LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 qt5-default

RUN python -m pip install --upgrade pip
RUN pip install open3d==0.18.0 plyfile pytorch_msssim imageio[ffmpeg] matplotlib lpips opencv-python mmcv==1.6.0
RUN apt-get install -y python3-tk
RUN pip install trimesh pyglet==v1.5.28 omegaconf 
RUN pip install -U 'sapien>=3.0.0b1'