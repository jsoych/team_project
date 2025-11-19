FROM ubuntu:24.04

# install miniconda
RUN apt update && apt install -y wget
RUN mkdir -p /root/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh
RUN bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && /root/miniconda3/bin/conda init --all 
RUN rm /root/miniconda3/miniconda.sh 

# create worker directory
RUN mkdir /root/worker
WORKDIR /root/worker
RUN mkdir configs logs src

# create conda environment
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
ADD environment.yaml environment.yaml
RUN /root/miniconda3/bin/conda env create -f environment.yaml
RUN /root/miniconda3/bin/conda run -n worker python -m pip install --upgrade redis
RUN rm environment.yaml

COPY src src
WORKDIR src
CMD [ "/root/miniconda3/bin/conda", "run", "-n", "worker", "python", "main.py" ]