FROM experiment:cpu

# set up experiment environment
RUN /root/miniconda3/bin/conda install -n experiment scikit-learn
WORKDIR /root/experiment
COPY src src

# run experiment
WORKDIR src
CMD [ "/root/miniconda3/bin/conda", "run", "-n", "experiment", "python", "experiment.py" ]