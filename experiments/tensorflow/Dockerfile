FROM continuumio/miniconda3

# set up environment
ADD environment.yaml environment.yaml
RUN conda env create -f environment.yaml

# make data and model directories
RUN mkdir data logs src

# copy assets into src
COPY src src

CMD ["conda", "-n", "experiment", "run", "python", "/src/main.py"]
