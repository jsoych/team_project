FROM experiment:gpu

# Update experiment environment
WORKDIR /root/experiment
ENV CONDA_OVERRIDE_CUDA=12.8.1
ADD environment.yaml environment.yaml
RUN /root/miniconda3/bin/conda env update -f environment.yaml 
COPY src src

# Run experiment
WORKDIR src
CMD ["/root/miniconda3/bin/conda", "run", "-n", "experiment", "python", "experiment.py"]
