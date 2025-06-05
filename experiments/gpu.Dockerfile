FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# install minicoda
RUN apt update
RUN apt install -y wget
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && ~/miniconda3/bin/conda init --all 
RUN rm ~/miniconda3/miniconda.sh

# set up environment
ADD environment.yaml environment.yaml
RUN ~/miniconda3/bin/conda env create -f environment.yaml

# make data and model directories
RUN mkdir data logs src

# copy assets into src
COPY assets src
