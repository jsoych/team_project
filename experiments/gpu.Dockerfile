FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# install minicoda
RUN apt update && apt install -y wget
RUN mkdir -p /root/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh
RUN bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && /root/miniconda3/bin/conda init --all 
RUN rm /root/miniconda3/miniconda.sh

# set up experiment environment
RUN mkdir /root/experiment
WORKDIR /root/experiment
ADD environment.yaml environment.yaml
RUN /root/miniconda3/bin/conda env create -f environment.yaml
RUN rm environment.yaml
RUN mkdir configs logs src

# copy assets into src
COPY assets src
